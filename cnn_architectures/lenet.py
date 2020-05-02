from keras.models import Sequential
from keras.layers.core import Flatten, Dense, Activation
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras import backend as K
from keras import regularizers

class Lenet:

    def load(width, height, depth, classes ):
        model = Sequential()
        inputShape = (height, width, depth)
        if K.image_data_format=="channels_first":
            input_shape = (depth, height, width)

        model.add(Conv2D(20,(5,5), padding='same', input_shape=inputShape))
        model.add(Activation('tanh'))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        model.add(Conv2D(50, (5, 5), padding="same"))
        model.add(Activation("tanh"))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        model.add(Flatten())
        model.add(Dense(500))
        model.add(Activation("tanh"))
        model.add(Dense(classes, kernel_regularizer=regularizers.l1(0.01)))
        model.add(Activation("softmax"))
        return model
