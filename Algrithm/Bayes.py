from Algrithm.methods import *


class Bayes(object):
    def __init__(self):
        self.trainSet = None
        self.trainData = 0  # 训练集数据
        self.trainLabel = 0  # 训练集标记
        self.yProba = {}  # 先验概率
        self.xyProba = {}  # 条件概率
        self.ySet = {}  # 标记类别对应的数量
        self.ls = 1  # 加入的拉普拉斯平滑的系数
        self.n_samples = 0  # 训练集样本数量
        self.n_features = 0  # 训练集特征数量

    # 计算P(y)先验概率
    def calPy(self, y, LS=True):
        Py = {}
        yi = {}
        ySet = np.unique(y)
        for i in ySet:
            Py[i] = (sum(y == i) + self.ls) / (self.n_samples + len(ySet))
            yi[i] = sum(y == i)
        self.yProba = Py
        self.ySet = yi
        return

    # 计算P(y)先验概率
    def calPxy(self, X, y, LS=True):
        Pxy = {}
        for yi, yiCount in self.ySet.items():
            Pxy[yi] = {}  # 第一层是标签Y的分类
            for xIdx in range(self.n_features):
                Pxy[yi][xIdx] = {}  # 第二层是不同的特征
                # 下标为第xIdx的特征数据
                Xi = X[:, xIdx]
                XiSet = np.unique(Xi)
                XiSetCount = XiSet.size
                # 下标为第xIdx，并标签为yi的特征数据
                Xiyi = X[np.nonzero(y == yi)[0], xIdx]
                for xi in XiSet:
                    Pxy[yi][xIdx][xi] = self.classifyProba(xi, Xiyi, XiSetCount)  # 第三层是变量Xi的分类概率，离散变量
        self.xyProba = Pxy
        return

    # 如果是离散变量就直接计算概率
    def classifyProba(self, x, xArr, XiSetCount):
        Pxy = (sum(xArr == x) + self.ls) / (xArr.size + XiSetCount)  # 加入拉普拉斯修正的概率
        return Pxy

    def train(self, X, y):
        self.n_samples, self.n_features = X.shape
        self.calPy(y)
        self.calPxy(X, y)
        self.trainSet = X
        self.trainLabel = y
        return self

    def predict_proba(self, X):
        m, n = X.shape
        proba = np.zeros((m, len(self.yProba)))
        for i in range(m):
            for idx, (yi, Py) in enumerate(self.yProba.items()):
                proba_idx = Py
                for xIdx in range(n):
                    xi = X[i, xIdx]
                    proba_idx *= self.xyProba[yi][xIdx][xi]
                proba[i, idx] = proba_idx
        return proba

    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)

    def accuracy(self, X):
        pred = self.predict(X)
        accu = 1.0 * np.sum(np.where(self.trainLabel == pred, 1, 0))/pred.shape[0]
        return accu
