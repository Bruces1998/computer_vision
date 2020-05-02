from sklearn.preprocessing import LabelBinarizer
from keras.optimizers import SGD
from lenet import Lenet
import numpy as np
from sklearn.metrics import classification_report

data = np.load('Kannada_MNIST_datataset_paper/Kannada_MNIST_npz/Kannada_MNIST/X_kannada_MNIST_train.npz')
labels = np.load('Kannada_MNIST_datataset_paper/Kannada_MNIST_npz/Kannada_MNIST/y_kannada_MNIST_train.npz')
lst1 = data.files
lst2 = labels.files
for item in lst1:
    print(item)
#     print(data[item])

for item in lst2:
    print(item)

data1 = data[item]
label = labels[item]
data1 = data1.reshape(data1.shape[0],28, 28, 1)

from sklearn.model_selection import train_test_split
(trainX, testX, trainY, testY ) = train_test_split(data1, label, test_size=0.3)

testY = LabelBinarizer().fit_transform(testY)
trainY = LabelBinarizer().fit_transform(trainY)

sgd = SGD(lr=0.1)

model = Lenet.load(height=28, width=28, depth=1, classes=10)

print("[INFO] training network......")
model.compile(loss="categorical_crossentropy", optimizer=sgd,metrics=["accuracy"] )
H = model.fit(trainX, trainY, validation_data=(testX, testY), epochs=8, batch_size=128)
print("[INFO] evaluating network......")

predictions = model.predict(testX,batch_size=128)
print(classification_report(testY.argmax(axis=1), predictions.argmax(axis=1)))
