from Algrithm.methods import *
from Algrithm.DecisionTree import CARTClassifier, CARTRegressor
import copy


class AdaBoostClassifier(object):
    def __init__(self, base_estimator=None, n_estimators=10, learning_rate=1.0):
        self.base_estimator = base_estimator
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        if self.base_estimator is None:
            # 默认使用决策树桩
            self.base_estimator = CARTClassifier(max_depth=2)
        # 同质分类器
        if type(base_estimator) != list:
            estimator = self.base_estimator
            self.base_estimator = [copy.deepcopy(estimator) for _ in range(0, self.n_estimators)]
        # 异质分类器
        else:
            self.n_estimators = len(self.base_estimator)
        # 记录estimator权重
        self.estimator_weights = []

    def train(self, x, y):
        n_sample = x.shape[0]
        sample_weights = np.asarray([1.0] * n_sample)
        for index in range(0, self.n_estimators):
            self.base_estimator[index].train(x, y, sample_weight=sample_weights)

            indicates = (self.base_estimator[index].predict(x) == y).astype(int)
            # 计算误分率
            error_rate = np.sum([sample_weights[j] * (1.0 - indicates[j]) for j in range(0, n_sample)]) / n_sample

            # 计算权重系数
            alpha_rate = 1.0 / 2.0 * np.log((1 - error_rate) / (error_rate + 1e-7))
            # alpha_rate = min(10.0, alpha_rate)
            self.estimator_weights.append(alpha_rate)

            # 更新样本权重
            for j in range(0, n_sample):
                sample_weights[j] = sample_weights[j] * np.exp(-1.0 * alpha_rate * np.power(-1.0, 1 - indicates[j]))
            sample_weights = sample_weights / np.sum(sample_weights) * n_sample
        # 更新estimator权重
        for i in range(0, self.n_estimators):
            self.estimator_weights[i] *= np.power(self.learning_rate, i)

    def predict_proba(self, x):
        result = np.sum(
            [self.base_estimator[j].predict_proba(x) * self.estimator_weights[j] for j in range(0, self.n_estimators)],
            axis=0)
        return result / result.sum(axis=1, keepdims=True)

    def predict(self, x):
        return np.argmax(self.predict_proba(x), axis=1)


class AdaBoostRegressor(object):
    def __init__(self, base_estimator=None, n_estimators=10, learning_rate=1.0):
        self.base_estimator = base_estimator
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        if self.base_estimator is None:
            # 默认使用决策树桩
            self.base_estimator = CARTRegressor(max_depth=2)
        # 同质分类器
        if type(base_estimator) != list:
            estimator = self.base_estimator
            self.base_estimator = [copy.deepcopy(estimator) for _ in range(0, self.n_estimators)]
        # 异质分类器
        else:
            self.n_estimators = len(self.base_estimator)

        # 记录estimator权重
        self.estimator_weights = []

        # 记录最终中位数值弱学习器的index
        self.median_index = None

    def fit(self, x, y):
        n_sample = x.shape[0]
        sample_weights = np.asarray([1.0] * n_sample)
        for index in range(0, self.n_estimators):
            self.base_estimator[index].train(x, y, sample_weight=sample_weights)

            errors = np.abs(self.base_estimator[index].predict(x) - y)
            error_max = np.max(errors)

            # 计算线性误差，其他误差类型，可以自行扩展
            linear_errors = errors / error_max
            # 计算误分率
            error_rate = np.dot(linear_errors, sample_weights / n_sample)

            # 计算权重系数
            alpha_rate = error_rate / (1.0 - error_rate + 1e-10)
            self.estimator_weights.append(alpha_rate)

            # 更新样本权重
            for j in range(0, n_sample):
                sample_weights[j] = sample_weights[j] * np.power(alpha_rate, 1 - linear_errors[j])
            sample_weights = sample_weights / np.sum(sample_weights) * n_sample

        # 更新estimator权重
        self.estimator_weights = np.log(1 / np.asarray(self.estimator_weights))
        for i in range(0, self.n_estimators):
            self.estimator_weights[i] *= np.power(self.learning_rate, i)
        self.estimator_weights /= np.sum(self.estimator_weights)

    def predict(self, x):
        return np.sum(
            [self.estimator_weights[i] * self.base_estimator[i].predict(x) for i in range(0, self.n_estimators)],
            axis=0)
