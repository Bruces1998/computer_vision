
from sklearn.preprocessing import LabelBinarizer
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from keras.layers import Dense
from keras.models import Sequential
import numpy as np
from keras.optimizers import SGD
import matplotlib.pyplot as plt
import pandas as pd



print("[INFO] Loading MNIST......")
# dataset = datasets.fetch_mldata("MNIST Original")
train = pd.read_csv('trainmnist.csv')
# test = pd.read_csv('Downloads/testmnist.csv')

trainXX = train.drop('label',axis=1)
trainXX = np.array(trainXX / 255.0)

# testX = test.drop('label', axis=1)
# testX = np.aaray(testX / 255.0)

trainYY = np.array(train['label'])
# testY = np.array(test['label'])




# data = dataset.data.astype("float") / 255.0

(trainX ,testX, trainY, testY)=train_test_split( trainXX, trainYY, test_size = 0.25)

lb = LabelBinarizer()

trainY = lb.fit_transform(trainY)

testY = lb.fit_transform(testY)

model = Sequential()

model.add(Dense(256, input_shape=(784,), activation="sigmoid"))
model.add(Dense(128, activation="sigmoid"))
model.add(Dense(10, activation="softmax"))

print("[INFO] training network.....")
sgd = SGD(0.1)
model.compile(loss="categorical_crossentropy", optimizer=sgd, metrics=["accuracy"])
H = model.fit(trainX, trainY, validation_data=(testX, testY), epochs=100, batch_size=128)
print("[INFO] evaluating network.....")
predictions = model.predict(testX,batch_size=128)
print(classification_report(testY.argmax(axis=1), predictions.argmax(axis=1), target_names=[str(x) for x in lb.classes_]))
