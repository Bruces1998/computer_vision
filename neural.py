import numpy as np
class NeuralNetwork:
    def __init__(self, layers, alpha=0.1, gamma=0.01):

        self.W = []
        self.layers = layers

        self.alpha = alpha
        self.gamma = gamma

        for i in np.arange(0,len(layers)-2):
            w = np.random.randn(layers[i]+1, layers[i+1]+1)

            self.W.append(w/np.sqrt(layers[i]))


        w = np.random.randn(layers[-2]+1,layers[-1])

        self.W.append(w/np.sqrt(layers[-2]))




    def __repr__(self):

        return "NeuralNetwork: {}".format("-".join(str(l) for l in self.layers))


    def sigmoid(self, x):

        return 1.0 / (1 + np.exp(-x ))


    def sigmoid_deriv(self, x):

        return x * (1-x)



    def fit(self, X, y, epochs=1000, displayUpdate=100):


        X = np.c_[X, np.ones((X.shape[0]))]


        for epoch in np.arange(0, epochs):

            for (x, target) in zip(X, y):
                self.fit_partial(x, target)



            if epoch == 0 or (epoch+1)% displayUpdate == 0:
                loss = self.calculate_loss(X, y)
                print("[INFO] epoch={}, loss={:.7f}".format(epoch+1, loss))


    def fit_partial(self, x, y):


        A = [np.atleast_2d(x)]

        for layer in np.arange(0, len(self.W) ):
            # print("AW",A[layer],self.W[layer])
            net = A[layer].dot(self.W[layer])
            out = self.sigmoid(net)


            A.append(out)


        error = A[-1] - y



        D = [error * self.sigmoid_deriv(A[-1])]

        for layer in np.arange(len(A) - 2,0 ,-1):

            delta = D[-1].dot(self.W[layer].T)
            delta = delta * self.sigmoid_deriv(A[layer])
            # print(type(delta))
            D.append(delta)

        D = D[::-1]


        for layer in np.arange(0, len(self.W)):
            # penalty=0
            # for i in np.arange(0,self.W[layer].shape[0]):
            #     for j in np.arange(0, self.W[layer].shape[1]):
            #         penalty+=(self.W[layer][i][j])**2
            # + self.gamma * penalty

            self.W[layer] += -self.alpha * A[layer].T.dot(D[layer])

    def predict(self, X, addBias= True):

        p = np.atleast_2d(X)

        if addBias:
            p = np.c_[p, np.ones((p.shape[0]))]


        for layer in np.arange(0, len(self.W)):

            p= self.sigmoid(np.dot(p, self.W[layer]))


        return p


    def calculate_loss(self, X, targets):
        targets = np.atleast_2d(targets)
        predictions = self.predict(X, addBias=False)

        loss = 0.5 * np.sum((predictions - targets) ** 2)

        return loss



# X = np.array([[0, 0, 0], [0, 0, 1], [0, 1,0], [0, 1,1], [1,0,0], [1,0,1], [1,1,0], [1,1,1]])
# y = np.array([[0],[1],[1],[0],[1],[0],[0],[1]])
# nn = NeuralNetwork([3,8,2,1], alpha=0.5)
# nn.fit(X, y, epochs=10000)
#
# for (x, target) in zip(X, y):
#
#     pred = nn.predict(x)[0][0]
#
#     step = 1 if pred > 0.5 else 0
#     print("[INFO] data={}, ground-truth={}, pred={:.4f}, step={}".format(x, target[0], pred, step))






from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn import datasets


print("[INFO] loading MNIST (sample) dataset...")
digits = datasets.load_digits()
data = digits.data.astype("float")

data = (data - data.min()) / (data.max() - data.min())
print("[INFO] samples: {}, dim: {}".format(data.shape[0], data.shape[1]))


(trainX ,testX, trainY, testY)=train_test_split( data, digits.target, test_size = 0.25)

trainY = LabelBinarizer().fit_transform(trainY)
testY = LabelBinarizer().fit_transform(testY)

print("[INFO] training Network......")
nn = NeuralNetwork([trainX.shape[1],32, 16, 10])
print("[INFO] {}".format(nn))
nn.fit(trainX, trainY, epochs = 1000)


print("[INFO] evaluating network......")
predictions = nn.predict(testX)
predictions = predictions.argmax(axis=1)
print(classification_report(testY.argmax(axis=1), predictions))
