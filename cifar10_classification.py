import numpy as np
from keras.datasets import cifar10
from keras.layers import Dense
from keras.models import Sequential
from keras.optimizers import SGD
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report
import matplotlib.pyplot as pyplot


print("[INFO] loading CIFAR-10 data.....")
((trainX, trainY),(testX, testY))=cifar10.load_data()

trainX = trainX.astype("float") / 255.0
testX = testX.astype("float") / 255.0

trainX = trainX.reshape((trainX.shape[0], 3072))
testX = testX.reshape((testX.shape[0], 3072))

le = LabelBinarizer()

trainY = le.fit_transform(trainY)
testY = le.fit_transform(testY)


labelNames = ["airplane", "automobile", "bird", "cat", "deer","dog", "frog", "horse", "ship", "truck"]
model=Sequential()
model.add(Dense(1024, input_shape=(3072,), activation="relu"))
model.add(Dense(512, activation="relu"))
model.add(Dense(10, activation="softmax"))


print("[INFO] training network....")
sgd = SGD(0.1)

model.compile(loss = "categorical_crossentropy", optimizer=sgd, metrics=["accuracy"])

H = model.fit(trainX, trainY, validation_data=(testX, testY), epochs=50, batch_size=32)
print("[INFO] evaluating network....")
predictions = model.predict(testX, batch_size=32)
print(classification_report(testY.argmax(axis=1), predictions.argmax(axis=1), target_names=labelNames))
