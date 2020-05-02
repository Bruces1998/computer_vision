from keras import Input
from keras.layers import Activation, BatchNormalization, Dense, Dropout, Conv2D, MaxPooling2D, Flatten
from keras.models import Model
import keras.backend as k
from keras.regularizers import l2

class AlexNet:
    def load(height, width, depth, classes, reg=0.0002):
        input_shape = (height, width, depth)
        channel_dim = -1

        if k.image_data_format=='channel_first':
            input_shape = (depth, height, width)
            channel_dim = 1
        input = Input(shape=input_shape)
        X = Conv2D(strides=(4, 4), filters=96, kernel_size=(11, 11), kernel_regularizer=l2(reg), padding='same')(input)
        X = Activation(activation="relu")(X)
        X = BatchNormalization(axis=channel_dim)(X)
        X = MaxPooling2D(pool_size=(3,3), strides=(2, 2))(X)
        X = Dropout(0.25)(X)

        X = Conv2D(filters=256, kernel_size=(5, 5), kernel_regularizer=l2(reg), padding='same')(X)
        X = Activation(activation="relu")(X)
        X = BatchNormalization(axis=channel_dim)(X)
        X = MaxPooling2D(pool_size=(3,3), strides=(2, 2))(X)
        X = Dropout(0.25)(X)

        X = Conv2D(filters=384, kernel_size=(3, 3), kernel_regularizer=l2(reg), padding='same')(X)
        X = Activation(activation="relu")(X)
        X = BatchNormalization(axis=channel_dim)(X)

        X = Conv2D(filters=384, kernel_size=(3, 3), kernel_regularizer=l2(reg), padding='same')(X)
        X = Activation(activation="relu")(X)
        X = BatchNormalization(axis=channel_dim)(X)

        X = Conv2D(filters=256, kernel_size=(3, 3), kernel_regularizer=l2(reg), padding='same')(X)
        X = Activation(activation="relu")(X)
        X = BatchNormalization(axis=channel_dim)(X)
        X = MaxPooling2D(pool_size=(3,3), strides=(2, 2))(X)
        X = Dropout(0.25)(X)

        X = Flatten()(X)

        X = Dense(4096, kernel_regularizer=l2(reg))(X)
        X = Activation(activation='relu')(X)
        X = BatchNormalization()(X)
        X = Dropout(0.5)(X)

        X = Dense(4096, kernel_regularizer=l2(reg))(X)
        X = Activation(activation='relu')(X)
        X = BatchNormalization()(X)
        X = Dropout(0.5)(X)

        X = Dense(classes, kernel_regularizer=l2(reg))(X)
        X = Activation(activation='softmax')(X)


        model = Model(inputs=[input], outputs=[X])

        return model
