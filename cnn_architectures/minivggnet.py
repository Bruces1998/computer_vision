from keras.layers import Conv2D, Dense, Activation, Flatten, MaxPool2D, BatchNormalization, Dropout
from keras.models import Sequential
from keras import backend as K



class VggNet():
    def load(height, width, depth, classes):
        inputShape=(height, width, depth)
        channel_dim = -1

        if K.image_data_format=='channel_first':
            inputShape=(depth, height, width)
            channel_dim = 1



        model= Sequential()
        model.add(Conv2D(filters=32, kernel_size=(3,3),padding="same", input_shape=inputShape))
        model.add(Activation("relu"))
        # model.add(BatchNormalization(axis=channel_dim))


        model.add(Conv2D(filters=32, kernel_size=(3,3),padding="same"))
        model.add(Activation("relu"))
        # model.add(BatchNormalization(axis=channel_dim))
        model.add(MaxPool2D(pool_size=(2, 2), strides=(2,2)))
        model.add(Dropout(0.25))


        model.add(Conv2D(filters=64, kernel_size=(3,3),padding="same"))
        model.add(Activation("relu"))
        # model.add(BatchNormalization(axis=channel_dim))

        model.add(Conv2D(filters=32, kernel_size=(3,3),padding="same"))
        model.add(Activation("relu"))
        # model.add(BatchNormalization(axis=channel_dim))

        model.add(MaxPool2D(pool_size=(2, 2), strides=(2,2)))
        model.add(Dropout(0.25))



        model.add(Flatten())
        model.add(Dense(512))
        model.add(Activation("relu"))
        # model.add(BatchNormalization())
        model.add(Dropout(0.5))
        model.add(Dense(classes))
        model.add(Activation("softmax"))


        return model
