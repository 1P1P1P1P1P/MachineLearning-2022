from Algrithm.methods import *


class KMeans(object):
    def __init__(self, n_cluster=2, max_iter=10, n_init=10, tol=1e-4, distance_type='Euclidean', p=2):
        self.n_cluster = n_cluster
        self.max_iter = max_iter
        self.n_init = n_init  # 进行多次聚类，选择最好的一次
        self.tol = tol  # 停止聚类的阈值
        self.centers = None
        self.distance_type = distance_type
        self.p = p

    def _error(self, x, y_pred, centers):
        error = 0.0
        for cluster in range(self.n_cluster):
            if cluster in y_pred:
                index = y_pred == cluster
                error += np.sqrt(np.sum(np.power(x[index, :] - centers[cluster], 2)))
        return error

    def _init_center(self, x):
        if self.n_cluster > x.shape[0]:
            self.n_cluster = x.shape[0]
        # 初始化中心
        centers = x[np.random.randint(0, x.shape[0], self.n_cluster), :]
        return centers

    def getDistance(self, center, x):
        if self.distance_type == 'Manhattan':
            distances = np.r_[[Manhattan(center, xp) for xp in x]]
        elif self.distance_type == 'Minkowski':
            distances = np.r_[[Minkowski(center, xp, self.p) for xp in x]]
        elif self.distance_type == 'CosDistance':
            distances = np.r_[[CosDistance(center, xp) for xp in x]]
        else:
            distances = np.r_[[Euclidean(center, xp) for xp in x]]
        return distances

    def _kmeans(self, x):
        centers = self._init_center(x)
        y_pred = None
        # 迭代
        for i in range(self.max_iter):
            oldCenters = centers.copy()
            distances = self.getDistance(centers, x)
            y_pred = np.argmin(distances, axis=1)
            for cluster in range(self.n_cluster):
                if cluster in y_pred:
                    centers[cluster] = np.mean(x[y_pred == cluster, :], axis=0)
                else:
                    centers[cluster] = x[np.random.randint(0, x.shape[0], 1), :]
            centerShiftTotal = np.linalg.norm(oldCenters - centers) ** 2
            if centerShiftTotal <= self.tol:
                break
        error = self._error(x, y_pred, centers)
        return centers, error, y_pred

    def train(self, x):
        bestError = None
        bestCenter = None
        for i in range(self.n_init):
            centers, error, y_pred = self._kmeans(x)
            if bestError is None or bestError >= error:
                bestError = error
                bestCenter = centers
        self.centers = bestCenter
        return self

    def predict(self, x):
        distances = self.getDistance(self.centers, x)
        return np.argmin(distances, axis=1)
