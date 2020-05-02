from cnn_architectures.alexnet import AlexNet
from keras.utils import plot_model


model = AlexNet.load(227, 227, 3, 1000)
plot_model(model, to_file='alexnet.png', show_shapes=True)
