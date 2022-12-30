from Algrithm.methods import *


class LogisticClassification(object):
    def __init__(self):
        self.train_data = None
        self.train_label = None
        self.w = None

    def CE_loss(self, w):
        X = self.train_data
        Y = self.train_label
        Y = Y.reshape(Y.shape[0], 1)
        r = (np.sum(np.multiply(1.0 / (1 + np.exp(np.multiply(Y, np.dot(X, w)))), np.multiply(-Y, X)), axis=0) /
             X.shape[0]).reshape(w.shape[0], 1)
        # result = zeros((X.shape[1], 1))
        # for i in range(X.shape[0]):
        #     result += (1.0 / (1 + exp(Y[i] * dot(X[i], w))) * (-Y[i] * X[i])).T
        # nabla_E = 1.0 / X.shape[0] * result
        return r

    def train(self, x=None, y=None):
        if x is not None and y is not None:
            self.train_data = np.hstack((np.ones((x.shape[0], 1)), x))
            y[y == 0] = -1
            self.train_label = y
        trainData = np.mat(self.train_data)
        m, n = np.shape(trainData)
        W_train = np.zeros((n, 1))  # 初始W为0
        eta = 0.1
        step = 0
        while True:
            step += 1
            direction = self.CE_loss(W_train)
            if np.linalg.norm(direction) == 0:
                break
            else:
                W_train -= direction * eta
            if step >= 2000:
                break
        self.w = W_train.copy()
        print(f"W更新次数为{step}")
        print(f"最终W为: {W_train.flatten()}")
        return self

    def predict(self, dataTest):
        dataTest = np.hstack((np.ones((dataTest.shape[0], 1)), dataTest))
        return (dataTest.dot(self.w) > 0).astype(int)

    def predict_proba(self, dataTest):
        dataTest = np.hstack((np.ones((dataTest.shape[0], 1)), dataTest))
        proba = 1.0 / (1 + np.exp(-dataTest.dot(self.w)))
        return proba
