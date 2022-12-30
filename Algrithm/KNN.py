import pandas as pd
from tqdm import tqdm
from Algrithm.methods import *


class KNN(object):
    def __init__(self, k, pred_type='classify', distance_type='euclidean', p=2, weight_type=None):
        self.y = None
        self.x = None
        self.k = k
        self.pred_type = pred_type
        self.distance_type = distance_type
        self.p = p
        self.weight_type = weight_type

    def train(self, x, y):
        self.x = x
        self.y = y

    def getDistance(self, x, xp):
        if self.distance_type == 'Manhattan':
            distances = Manhattan(x, xp)
        elif self.distance_type == 'Minkowski':
            distances = Minkowski(x, xp, self.p)
        elif self.distance_type == 'CosDistance':
            distances = CosDistance(x, xp)
        else:
            distances = Euclidean(x, xp)
        indexs = np.argsort(distances)
        distances.sort()
        return np.asarray(distances[:self.k]).reshape(1, -1), np.asarray(indexs[:self.k]).reshape(1, -1)

    def getWeight(self, distance, sigma=10.0):
        if self.weight_type == 'gaussian':
            weights = np.exp(-distance ** 2 / (2 * sigma ** 2))
        else:
            weights = np.ones(distance.shape)
        return weights

    def predict(self, xps):
        x = self.x
        y = self.y
        distances, indexs = self.getDistance(x, xps[0])
        weights = self.getWeight(distances)
        for i in tqdm(range(1, xps.shape[0])):
            distance, index = self.getDistance(x, xps[i])
            # distances = np.r_[distances, distance]
            indexs = np.r_[indexs, index]
            weight = self.getWeight(distance)
            weights = np.r_[weights, weight]
        y = y[indexs]
        if self.pred_type == 'regression':
            avg = np.sum(weights * y, axis=1, keepdims=True)
            total_weight = np.sum(weights, axis=1, keepdims=True)
            pred = avg / total_weight
        else:
            pred = np.empty((0, 1))
            for i in tqdm(range(y.shape[0])):
                y_weights = weights[i] * y[i]
                df = pd.DataFrame(np.c_[y_weights.T, y[i].T], columns=['y_weights', 'y'])
                df_class = df.groupby('y').sum()
                pred = np.append(pred, np.asarray(df_class.idxmax()).reshape(-1, 1), axis=0)
        return pred
