import matplotlib
import numpy as np
import pandas as pd
import argparse
from keras.datasets import cifar10
import matplotlib.pyplot as plt
from keras.optimizers import SGD
from keras import layers
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report
from minivggnet import VggNet
from keras.callbacks import LearningRateScheduler
from trainingmonitor import TrainingMonitor
import os


def step_decay(epoch):
    initAlpha = 0.01
    factor = 0.25
    dropEvery = 5

    alpha = initAlpha * (factor ** np.floor((1+ epoch)/ dropEvery))
    return float(alpha)



ap = argparse.ArgumentParser()
ap.add_argument("-o", "--output", required=True, help="path to output loss/accuracy plot")
args = vars(ap.parse_args())
print("[INFO] process ID: {}".format(os.getpid()))

print("[INFO] Loading cifar10 Data.....")
((trainX, trainY), (testX, testY)) = cifar10.load_data()


trainX = trainX.astype("float") / 255.0
testX = testX.astype("float") / 255.0

lb = LabelBinarizer()

trainY = lb.fit_transform(trainY)

testY = lb.fit_transform(testY)

labelNames = ["airplane", "automobile", "bird", "cat", "deer",
"dog", "frog", "horse", "ship", "truck"]
figPath = os.path.sep.join([args["output"], "{}.png".format(os.getpid())])
jsonPath = os.path.sep.join([args["output"], "{}.json".format(os.getpid())])


callbacks = [TrainingMonitor(figPath, jsonPath = jsonPath),LearningRateScheduler(step_decay)]

opt = SGD(lr=0.01, momentum = 0.9, nesterov=True)
model = VggNet.load(width = 32, height = 32, depth = 3, classes = 10)
model.compile(loss = "categorical_crossentropy", optimizer = opt, metrics = ['accuracy'])


H = model.fit(trainX, trainY, validation_data = (testX, testY), batch_size = 64, epochs=20, callbacks = callbacks, verbose = 1)

print("[INFO] evaluating Model.....")

predictions = model.predict(testX, batch_size = 64)

print(classification_report(testY.argmax(axis=1), predictions.argmax(axis=1), target_names=labelNames))
