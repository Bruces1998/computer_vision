from keras.datasets import cifar10
from minivggnet import VggNet
from sklearn.preprocessing import LabelBinarizer
from keras.optimizers import SGD
from sklearn.metrics import classification_report



print("[INFO] Loading CIFAR-10 data.....")
((X_train, y_train), (X_test, y_test))=cifar10.load_data()
lb = LabelBinarizer()
y_train = lb.fit_transform(y_train)
y_test = lb.fit_transform(y_test)

X_test = X_test.astype("float") / 255.0
X_train = X_train.astype("float") / 255.0

labelNames = ["airplane", "automobile", "bird", "cat", "deer","dog", "frog", "horse", "ship", "truck"]

print("[INFO] compiling model......")
model = VggNet.load(height=32, width=32, depth=3, classes=10)

sgd = SGD(lr=0.1, decay=0.01/40 , momentum=0.9, nesterov=True)

model.compile(loss="categorical_crossentropy", optimizer=sgd, metrics=["accuracy"])

print("[INFO] trainig network......")

H = model.fit(X_train, y_train, validation_data=(X_test, y_test), batch_size=1024, epochs=20, verbose=1)

print("[INFO] evaluating network........")
predictions = model.predict(X_test, batch_size=1024)

print(classification_report(y_test.argmax(axis=1), predictions.argmax(axis=1), target_names=labelNames))
