import tensorflow as tf
import keras
from keras import layers
from keras.layers import Input, Conv2D, MaxPooling2D, Dropout, BatchNormalization, Activation, Conv1D, GlobalAveragePooling2D, multiply, Add
from keras.layers import Flatten, Dense, Reshape, Average, Bidirectional, LSTM, GRU, Permute, Softmax, Dot, MultiHeadAttention
from keras.models import Sequential, Model
import math

def channel_attenstion(inputs, ratio=0.25):
    '''ratio代表第一个全连接层下降通道数的倍数'''

    channel = inputs.shape[-1]  # 获取输入特征图的通道数
    print(channel)

    # [h,w,c]==>[None,c]
    x_max = layers.GlobalMaxPooling2D()(inputs)
    x_avg = layers.GlobalAveragePooling2D()(inputs)

    # [None,c]==>[1,1,c]
    x_max = layers.Reshape([1, 1, -1])(x_max)
    x_avg = layers.Reshape([1, 1, -1])(x_avg)


    x_max = layers.Dense(channel * ratio)(x_max)
    x_avg = layers.Dense(channel * ratio)(x_avg)

    x_max = layers.Activation('relu')(x_max)
    x_avg = layers.Activation('relu')(x_avg)
    x_max = layers.Dense(channel)(x_max)
    x_avg = layers.Dense(channel)(x_avg)
    x = layers.Add()([x_max, x_avg])
    x = tf.nn.sigmoid(x)
    x = layers.Multiply()([inputs, x])

    return x

def spatial_attention(inputs):

    x_max = tf.reduce_max(inputs, axis=3, keepdims=True)
    x_avg = tf.reduce_mean(inputs, axis=3, keepdims=True)

    x = layers.concatenate([x_max, x_avg])

    x = layers.Conv2D(filters=1, kernel_size=(1, 1), strides=1, padding='same')(x)

    x = tf.nn.sigmoid(x)

    x = layers.Multiply()([inputs, x])

    return x
def CBAM_attention(inputs):
    x = channel_attenstion(inputs)
    x = spatial_attention(x)
    return x


def eca_block(inputs, b=1, gama=2):
    in_channel = inputs.shape[-1]

    kernel_size = int(abs((math.log(in_channel, 2) + b) / gama))

    if kernel_size % 2:
        kernel_size = kernel_size

    else:
        kernel_size = kernel_size + 1

    x = layers.GlobalAveragePooling2D()(inputs)

    # [None,c]==>[c,1]
    x = layers.Reshape(target_shape=(in_channel, 1))(x)

    # [c,1]==>[c,1]
    x = Conv1D(filters=1, kernel_size=kernel_size, padding='same', use_bias=False)(x)

    # sigmoid激活
    x = tf.nn.sigmoid(x)

    # [c,1]==>[1,1,c]
    x = layers.Reshape((1, 1, in_channel))(x)

    # 结果和输入相乘
    outputs = layers.multiply([inputs, x])

    return outputs


def create_base_network(img_size, dropout_rate):
    inputs = keras.Input(shape=img_size)

    x = CBAM_attention(inputs)
    x = Conv2D(64, 5, activation='relu', padding='same', name='conv1')(x)
    x = BatchNormalization()(x)
    x = Dropout(dropout_rate)(x)

    eca = eca_block(x)
    x = layers.add([x, eca])

    x = Conv2D(128, 4, activation='relu', padding='same', name='conv3')(x)
    x = BatchNormalization()(x)
    x = Dropout(dropout_rate)(x)

    eca = eca_block(x)
    x = layers.add([x, eca])

    x = Conv2D(256, 4, activation='relu', padding='same', name='conv5')(x)
    x = BatchNormalization()(x)
    x = Dropout(dropout_rate)(x)

    eca = eca_block(x)
    x = layers.add([x, eca])

    x = Conv2D(64, 1, activation='relu', padding='same', name='conv7')(x)
    x = MaxPooling2D(2, 2, name='pool1')(x)
    x = BatchNormalization()(x)
    x = Dropout(dropout_rate)(x)
    # print("x shape", x.shape)

    eca = eca_block(x)
    x = layers.add([x, eca])


    x = layers.Flatten(name='fla1')(x)
    x = Dense(512, activation='relu', name='dense1')(x)
    x = layers.Reshape((1, 512))(x)
    x = BatchNormalization()(x)
    x = Dropout(dropout_rate)(x)
    # x = MultiHeadAttention(num_heads=6, key_dim=3, attention_axes=(1, 2))(x, x)
    x = Bidirectional(LSTM(128, name='lstm'))(x)
    # x = GRU(128, name='gru')(x)
    model = keras.Model(inputs, x)
    return model


def create_MT_CNN(img_size=(8, 9, 8), dropout_rate=0.2, number_of_inputs=1):

    base_network = create_base_network(img_size, dropout_rate)

    inputs = [Input(shape=img_size) for i in range(number_of_inputs)]

    if number_of_inputs == 1:
       x = base_network(inputs[0])
    else:
       x = Average()([base_network(input_) for input_ in inputs])

    x = Flatten(name='flat')(x)

    out_v = Dense(2, activation='softmax', name='out_v')(x)
    out_a = Dense(2, activation='softmax', name='out_a')(x)

    model = Model(inputs, [out_v, out_a])
    # model.summary()
    return model