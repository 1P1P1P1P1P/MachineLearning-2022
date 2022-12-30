from Algrithm.methods import *


class PLA(object):
    def __init__(self):
        self.train_data = None
        self.train_label = None
        self.w = None

    def train(self, x=None, y=None):
        if x is not None and y is not None:
            self.train_data = np.hstack((x, np.ones((x.shape[0], 1))))
            y[y == 0] = -1
            self.train_label = y
        trainData = self.train_data.copy()
        trainLabel = self.train_label.copy()
        m, n = np.shape(trainData)
        W_train = np.zeros((n, 1))  # 初始W为0
        step = 0
        while True:
            is_OK = True  # 判断是否分对
            for i in range(m):
                if sign(np.dot(trainData[i], W_train)) == trainLabel[i]:
                    continue
                else:
                    is_OK = False
                    W_train += (trainLabel[i] * trainData[i]).reshape(n, 1)
                    step += 1
            if is_OK:
                break
        self.w = W_train
        print(f"W更新次数为{step}")
        print(f"最终W为: {W_train.flatten()}")
        return self

    def predict(self, dataTest):
        dataTest = np.c_[dataTest, np.ones(shape=(dataTest.shape[0],))]
        return (dataTest.dot(self.w) > 0).astype(int)
