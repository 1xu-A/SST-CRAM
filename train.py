import os
import pickle
from shutil import copyfile

import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score
import numpy as np
import tensorflow as tf
import keras
from keras import backend as K
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, EarlyStopping

from eegemotion.utils import ReduceLRBacktrack, get_sample_weights
from eegemotion.model4 import create_MT_CNN


def train(x_all_subject, y_a_all_subject, y_v_all_subject, all_subject_id,
          subject_n=32, dropout_rate=0.2, number_of_inputs=1, model_dir='.', metrics_dir='.',
          model_name='MT_CNN', img_size=(8, 9, 8), lr_decay_factor=0.5,
          lr_decay_patience=5, epochs_n=2, seed=7, verbose=0, task='multi',
          fine_tuning=True):
    if task == 'multi':
        loss_weights = [1, 1]
    elif task == 'valence':
        loss_weights = [1, 0]
    elif task == 'arousal':
        loss_weights = [0, 1]
    else:
        raise NotImplementedError

    model_checkpoint_path_SD = f'{model_dir}/{model_name}-for-test.hdf5'

    lrate_silent = lambda: ReduceLRBacktrack(best_path=model_checkpoint_path_SD,
                                             monitor='val_loss', patience=lr_decay_patience,
                                             factor=lr_decay_factor, verbose=verbose)

    lrate = lambda model_checkpoint_path: ReduceLROnPlateau(best_path=model_checkpoint_path,
                                                            monitor="val_loss", patience=5,
                                                            factor=0.5, verbose=1)

    es = lambda: EarlyStopping(monitor='val_loss', mode='min',
                               verbose=1, patience=20)

    save_model_for_test = lambda: ModelCheckpoint(
        model_checkpoint_path_SD,
        monitor='val_loss', save_best_only=True,
        save_weights_only=False,
        mode='min', verbose=verbose)

    save_model = lambda model_checkpoint_path: ModelCheckpoint(model_checkpoint_path,
                                                               monitor='val_loss', save_best_only=True,
                                                               save_weights_only=False,
                                                               mode='min')

    scores_subject_independent_list = []

    np.random.seed(seed)
    n_splits = 5
    kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for fold, (train, test) in enumerate(kfold.split(x_all_subject, y_a_all_subject.argmax(1))):

        print(f'\n\nFold {fold + 1}/{n_splits}\n\n')
        # if fold > 1:
        #    continue
        K.clear_session()

        model_checkpoint_path_SI_unique = f'{model_dir}/{model_name}-weight_AV-fold{fold + 1:02d}' + \
                                          '-epoch{epoch:02d}-loss{val_loss:.2f}-A_accuracy{val_out_a_accuracy:.4f}-V_accuracy{val_out_v_accuracy:.4f}.hdf5'
        model_checkpoint_path_SI_for_load = f'{model_dir}/{model_name}-weight_AV-fold{fold + 1:02d}.hdf5'

        model = create_MT_CNN(img_size, dropout_rate, number_of_inputs)

        model.compile(loss=[keras.losses.categorical_crossentropy, keras.losses.categorical_crossentropy],
                      loss_weights=loss_weights,
                      optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                      metrics=['accuracy'])

        # Fit the model
        x_train = x_all_subject[train]
        y_train_a = y_a_all_subject[train]
        y_train_v = y_v_all_subject[train]
        subject_id_train = all_subject_id[train]

        x_test = x_all_subject[test]
        y_test_a = y_a_all_subject[test]
        y_test_v = y_v_all_subject[test]
        subject_id_test = all_subject_id[test]

        sample_weights = get_sample_weights([y_train_v, y_train_a])

        history = model.fit([x_train[:, i] for i in range(x_train.shape[1])],
                         [y_train_v, y_train_a], epochs=epochs_n,
                         batch_size=64, verbose=1,
                         sample_weight=sample_weights,
                         callbacks=[save_model(model_checkpoint_path_SI_unique),
                                    save_model(model_checkpoint_path_SI_for_load),
                                    lrate(model_checkpoint_path_SI_for_load),
                                    es()],
                         validation_data=([x_test[:, i] for i in range(x_test.shape[1])],
                                          [y_test_v, y_test_a]))




        scores = model.evaluate([x_test[:, i] for i in range(x_test.shape[1])],
                                [y_test_v, y_test_a], verbose=1)

        scores_subject_independent_list.append(scores)


        if fine_tuning:
            multi_scores_subject_dependent_list = []
            valence_scores_subject_dependent_list = []
            arousal_scores_subject_dependent_list = []
            scores_subject_dependent_list_before = []

            multi_scores_subject_dependent_per_fold = []
            valence_scores_subject_dependent_per_fold = []
            arousal_scores_subject_dependent_per_fold = []
            scores_subject_dependent_per_fold_before = []

            metrics = ['loss', 'valence loss', 'arousal loss', 'valence acc', 'arousal acc']
            f1_scores = []
            for i in range(subject_n):
                short_name = f'{i + 1:02}'
                K.clear_session()
                print("\nprocessing: ", short_name, "......")

                model = create_MT_CNN(img_size, dropout_rate, number_of_inputs)

                model.compile(loss=[keras.losses.categorical_crossentropy, keras.losses.categorical_crossentropy],
                              optimizer=tf.keras.optimizers.Adam(learning_rate=0.001 / 8),
                              metrics=['accuracy'])


                model.load_weights(model_checkpoint_path_SI_for_load)

                # Fit the model
                x_train_for_subject = x_train[subject_id_train == i]
                y_train_v_for_subject = y_train_v[subject_id_train == i]
                y_train_a_for_subject = y_train_a[subject_id_train == i]

                x_test_for_subject = x_test[subject_id_test == i]
                y_test_v_for_subject = y_test_v[subject_id_test == i]
                y_test_a_for_subject = y_test_a[subject_id_test == i]

                scores_for_subject = model.evaluate(
                    [x_test_for_subject[:, i] for i in range(x_test_for_subject.shape[1])],
                    [y_test_v_for_subject, y_test_a_for_subject], verbose=verbose)

                y_pred_v, y_pred_a = model.predict(
                    [x_test_for_subject[:, i] for i in range(x_test_for_subject.shape[1])])
                y_pred_v = np.argmax(y_pred_v, axis=1)
                y_pred_a = np.argmax(y_pred_a, axis=1)

                f1_v = f1_score(np.argmax(y_test_v_for_subject, axis=1), y_pred_v, average='weighted')
                f1_a = f1_score(np.argmax(y_test_a_for_subject, axis=1), y_pred_a, average='weighted')

                f1_scores.append({'Valence F1': f1_v, 'Arousal F1': f1_a})

                best_loss_SI = scores_for_subject[:3]


                scores_subject_dependent_per_fold_before.append(scores_for_subject[-2:])
                for score in scores_for_subject[:-2]:
                    print(round(score, 6), end=' ')
                print('Before fine-tuning Valence F1:', round(f1_v, 6), 'Before fine-tuning Arousal F1:', round(f1_a, 6))


                print('Before fine-tuning:', list(zip(metrics, [round(score, 6) for score in scores_for_subject])))


                # AV Multi-task fine-tuning
                copyfile(model_checkpoint_path_SI_for_load, model_checkpoint_path_SD)
                callbacks = [lrate_silent(),
                             save_model_for_test(),
                             es()]
                for callback in callbacks:
                    callback.best = best_loss_SI[0]

                sample_weights = get_sample_weights([y_train_v_for_subject, y_train_a_for_subject])
                hist_for_subject = model.fit([x_train_for_subject[:, i] for i in range(x_train_for_subject.shape[1])],
                                             [y_train_v_for_subject, y_train_a_for_subject], epochs=epochs_n,
                                             sample_weight=sample_weights,
                                             batch_size=64, verbose=verbose,
                                             callbacks=callbacks,
                                             validation_data=(
                                             [x_test_for_subject[:, i] for i in range(x_test_for_subject.shape[1])],
                                             [y_test_v_for_subject, y_test_a_for_subject]))

                model.load_weights(model_checkpoint_path_SD)

                scores_for_subject = model.evaluate(
                    [x_test_for_subject[:, i] for i in range(x_test_for_subject.shape[1])],
                    [y_test_v_for_subject, y_test_a_for_subject], verbose=verbose)

                y_pred_v, y_pred_a = model.predict(
                    [x_test_for_subject[:, i] for i in range(x_test_for_subject.shape[1])])
                y_pred_v = np.argmax(y_pred_v, axis=1)
                y_pred_a = np.argmax(y_pred_a, axis=1)

                f1_v = f1_score(np.argmax(y_test_v_for_subject, axis=1), y_pred_v, average='weighted')
                f1_a = f1_score(np.argmax(y_test_a_for_subject, axis=1), y_pred_a, average='weighted')

                f1_scores.append({'Valence F1': f1_v, 'Arousal F1': f1_a})

                multi_scores_subject_dependent_per_fold.append(scores_for_subject[-2:])

                for score in scores_for_subject[:-2]:
                    print(round(score, 6), end=' ')
                print('After fine-tuning on Multi-Task Valence F1:', round(f1_v, 6), 'After fine-tuning on Multi-Task Arousal F1:', round(f1_a, 6))
                std_dev = np.std(multi_scores_subject_dependent_per_fold, axis=0)
                print('After fine-tuning on Multi-Task',
                      list(zip(metrics, [round(score, 6) for score in scores_for_subject])))


                # For Valence
                K.clear_session()
                model = create_MT_CNN(img_size, dropout_rate, number_of_inputs)

                model.compile(loss=keras.losses.categorical_crossentropy, loss_weights=[1, 0],
                              optimizer=tf.keras.optimizers.Adam(learning_rate=0.001 / 4),
                              metrics=['accuracy'])

                model.load_weights(model_checkpoint_path_SI_for_load)

                copyfile(model_checkpoint_path_SI_for_load, model_checkpoint_path_SD)
                callbacks = [lrate_silent(),
                             save_model_for_test(),
                             es()]
                for callback in callbacks:
                    callback.best = best_loss_SI[1]

                sample_weights = get_sample_weights([y_train_v_for_subject])
                hist_for_subject = model.fit([x_train_for_subject[:, i] for i in range(x_train_for_subject.shape[1])],
                                             [y_train_v_for_subject, y_train_a_for_subject], epochs=epochs_n,
                                             sample_weight=sample_weights,
                                             batch_size=64, verbose=verbose,
                                             callbacks=callbacks,
                                             validation_data=(
                                             [x_test_for_subject[:, i] for i in range(x_test_for_subject.shape[1])],
                                             [y_test_v_for_subject, y_test_a_for_subject]))

                model.load_weights(model_checkpoint_path_SD)

                scores_for_subject = model.evaluate(
                    [x_test_for_subject[:, i] for i in range(x_test_for_subject.shape[1])],
                    [y_test_v_for_subject, y_test_a_for_subject], verbose=verbose)

                y_pred_v, y_pred_a = model.predict(
                    [x_test_for_subject[:, i] for i in range(x_test_for_subject.shape[1])])
                y_pred_v = np.argmax(y_pred_v, axis=1)
                y_pred_a = np.argmax(y_pred_a, axis=1)

                f1_v = f1_score(np.argmax(y_test_v_for_subject, axis=1), y_pred_v, average='weighted')
                f1_a = f1_score(np.argmax(y_test_a_for_subject, axis=1), y_pred_a, average='weighted')

                f1_scores.append({'Valence F1': f1_v, 'Arousal F1': f1_a})

                valence_scores_subject_dependent_per_fold.append(scores_for_subject[-2:])
                for score in scores_for_subject[:-2]:
                    print(round(score, 6), end=' ')
                print('After fine-tuning on VALENCE Valence F1:', round(f1_v, 6), 'After fine-tuning on VALENCE Arousal F1:', round(f1_a, 6))

                print('After fine-tuning on VALENCE',
                      list(zip(metrics, [round(score, 6) for score in scores_for_subject])))


                # Arousal
                K.clear_session()
                model = create_MT_CNN(img_size, dropout_rate, number_of_inputs)

                model.compile(loss=keras.losses.categorical_crossentropy, loss_weights=[0, 1],
                              optimizer=tf.keras.optimizers.Adam(learning_rate=0.001 / 4),
                              metrics=['accuracy'])

                model.load_weights(model_checkpoint_path_SI_for_load)

                copyfile(model_checkpoint_path_SI_for_load, model_checkpoint_path_SD)
                callbacks = [lrate_silent(),
                             save_model_for_test(),
                             es()]
                for callback in callbacks:
                    callback.best = best_loss_SI[2]

                sample_weights = get_sample_weights([y_train_a_for_subject])
                hist_for_subject = model.fit([x_train_for_subject[:, i] for i in range(x_train_for_subject.shape[1])],
                                             [y_train_v_for_subject, y_train_a_for_subject], epochs=epochs_n,
                                             sample_weight=sample_weights,
                                             batch_size=64, verbose=verbose,
                                             callbacks=callbacks,
                                             validation_data=(
                                             [x_test_for_subject[:, i] for i in range(x_test_for_subject.shape[1])],
                                             [y_test_v_for_subject, y_test_a_for_subject]))

                model.load_weights(model_checkpoint_path_SD)

                scores_for_subject = model.evaluate(
                    [x_test_for_subject[:, i] for i in range(x_test_for_subject.shape[1])],
                    [y_test_v_for_subject, y_test_a_for_subject], verbose=verbose)
                y_pred_v, y_pred_a = model.predict(
                    [x_test_for_subject[:, i] for i in range(x_test_for_subject.shape[1])])
                y_pred_v = np.argmax(y_pred_v, axis=1)
                y_pred_a = np.argmax(y_pred_a, axis=1)

                f1_v = f1_score(np.argmax(y_test_v_for_subject, axis=1), y_pred_v, average='weighted')
                f1_a = f1_score(np.argmax(y_test_a_for_subject, axis=1), y_pred_a, average='weighted')

                f1_scores.append({'Valence F1': f1_v, 'Arousal F1': f1_a})
                arousal_scores_subject_dependent_per_fold.append(scores_for_subject[-2:])

                for score in scores_for_subject[:-2]:
                    print(round(score, 6), end=' ')
                print('After fine-tuning on AROUSAL Valence F1:', round(f1_v, 6), 'After fine-tuning on AROUSAL Arousal F1:', round(f1_a, 6))
                std_dev = np.std(arousal_scores_subject_dependent_per_fold, axis=0)
                print('After fine-tuning on AROUSAL',
                      list(zip(metrics, [round(score, 6) for score in scores_for_subject])))

            valence_scores_subject_dependent_list.append(valence_scores_subject_dependent_per_fold)
            arousal_scores_subject_dependent_list.append(arousal_scores_subject_dependent_per_fold)
            multi_scores_subject_dependent_list.append(multi_scores_subject_dependent_per_fold)

            scores_subject_dependent_list_before.append(scores_subject_dependent_per_fold_before)

            with open(os.path.join(metrics_dir, f'{model_name}_scores_SD_before.pkl'), 'wb') as fl:
                pickle.dump(scores_subject_dependent_list_before, fl)

            with open(os.path.join(metrics_dir, f'{model_name}_valence_scores_SD.pkl'), 'wb') as fl:
                pickle.dump(valence_scores_subject_dependent_list, fl)

            with open(os.path.join(metrics_dir, f'{model_name}_arousal_scores_SD.pkl'), 'wb') as fl:
                pickle.dump(arousal_scores_subject_dependent_list, fl)

            with open(os.path.join(metrics_dir, f'{model_name}_multi_scores_SD.pkl'), 'wb') as fl:
                pickle.dump(multi_scores_subject_dependent_list, fl)

        with open(os.path.join(metrics_dir, f'{model_name}_scores_SI.pkl'), 'wb') as fl:
            pickle.dump(scores_subject_independent_list, fl)
