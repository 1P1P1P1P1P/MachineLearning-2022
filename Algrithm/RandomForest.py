from Algrithm.Bagging import BaggingClassifier, BaggingRegressor
from Algrithm.methods import *


class RandomForestClassifier(BaggingClassifier):
    def __init__(self, base_estimator=None, n_estimators=10, feature_sample=0.66):
        super().__init__(base_estimator=None, n_estimators=10)
        self.feature_sample = feature_sample
        # 记录每个基学习器选择的特征
        self.feature_indices = []

    def train(self, x, y):
        n_sample, n_feature = x.shape
        for estimator in self.base_estimator:
            # 重采样训练集
            indices = np.random.choice(n_sample, n_sample, replace=True)
            x_bootstrap = x[indices]
            y_bootstrap = y[indices]
            # 对特征抽样
            feature_indices = np.random.choice(n_feature, int(n_feature * self.feature_sample), replace=False)
            self.feature_indices.append(feature_indices)
            x_bootstrap = x_bootstrap[:, feature_indices]
            estimator.train(x_bootstrap, y_bootstrap)

    def predict_proba(self, x):
        probas = []
        for index, estimator in enumerate(self.base_estimator):
            probas.append(estimator.predict_proba(x[:, self.feature_indices[index]]))
        return np.mean(probas, axis=0)


class RandomForestRegressor(BaggingRegressor):
    def __init__(self, base_estimator=None, n_estimators=10, feature_sample=0.66):
        super().__init__(base_estimator=None, n_estimators=10)
        self.feature_sample = feature_sample
        # 记录每个基学习器选择的特征
        self.feature_indices = []

    def train(self, x, y):
        n_sample, n_feature = x.shape
        for estimator in self.base_estimator:
            # 重采样训练集
            indices = np.random.choice(n_sample, n_sample, replace=True)
            x_bootstrap = x[indices]
            y_bootstrap = y[indices]
            # 对特征抽样
            feature_indices = np.random.choice(n_feature, int(n_feature * self.feature_sample), replace=False)
            self.feature_indices.append(feature_indices)
            x_bootstrap = x_bootstrap[:, feature_indices]
            estimator.train(x_bootstrap, y_bootstrap)

    def predict(self, x):
        preds = []
        for index, estimator in enumerate(self.base_estimator):
            preds.append(estimator.predict(x[:, self.feature_indices[index]]))

        return np.mean(preds, axis=0)

