from Algrithm.methods import *
from Algrithm.DecisionTree import CARTRegressor
import copy


class GradientBoostingRegressor(object):
    def __init__(self, base_estimator=None, n_estimators=10, learning_rate=1.0, loss='exp', huber_threshold=1e-1,
                 quantile_threshold=0.5):
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
        self.loss = loss
        self.huber_threshold = huber_threshold
        self.quantile_threshold = quantile_threshold

    def _get_gradient(self, y, y_pred):
        if self.loss == 'ls':
            return y - y_pred
        elif self.loss == 'exp':
            return y * np.exp(-1.0 * y * y_pred)
        elif self.loss == 'lae':
            return (y - y_pred > 0).astype(int) * 2 - 1
        elif self.loss == 'huber':
            return np.where(np.abs(y - y_pred) > self.huber_threshold,
                            self.huber_threshold * ((y - y_pred > 0).astype(int) * 2 - 1), y - y_pred)
        elif self.loss == "quantile":
            return np.where(y - y_pred > 0, self.quantile_threshold, self.quantile_threshold - 1)

    def train(self, x, y):
        # 拟合第一个模型
        self.base_estimator[0].train(x, y)
        y_pred = self.base_estimator[0].predict(x)
        new_y = self._get_gradient(y, y_pred)
        for index in range(1, self.n_estimators):
            self.base_estimator[index].train(x, new_y)
            y_pred += self.base_estimator[index].predict(x) * self.learning_rate
            new_y = self._get_gradient(y, y_pred)
        return self

    def predict(self, x):
        return np.sum(
            [self.base_estimator[0].predict(x)] +
            [self.learning_rate * self.base_estimator[i].predict(x) for i in range(1, self.n_estimators - 1)] +
            [self.base_estimator[self.n_estimators - 1].predict(x)], axis=0)


class GradientBoostingClassifier(object):
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

        # 扩展class_num组分类器
        self.expand_base_estimators = []

    def train(self, x, y):
        # 将y转one-hot编码
        class_num = np.amax(y) + 1
        y_cate = np.zeros(shape=(len(y), class_num))
        y_cate[np.arange(len(y)), y] = 1
        # 扩展分类器
        self.expand_base_estimators = [copy.deepcopy(self.base_estimator) for _ in range(class_num)]

        # 拟合第一个模型
        y_pred_score_ = []
        for class_index in range(0, class_num):
            self.expand_base_estimators[class_index][0].train(x, y_cate[:, class_index])
            y_pred_score_.append(self.expand_base_estimators[class_index][0].predict(x))
        y_pred_score_ = np.c_[y_pred_score_].T
        # 计算负梯度
        new_y = y_cate - softmax(y_pred_score_)
        # 训练后续模型
        for index in range(1, self.n_estimators):
            y_pred_score = []
            for class_index in range(0, class_num):
                self.expand_base_estimators[class_index][index].train(x, new_y[:, class_index])
                y_pred_score.append(self.expand_base_estimators[class_index][index].predict(x))
            y_pred_score_ += np.c_[y_pred_score].T * self.learning_rate
            new_y = y_cate - softmax(y_pred_score_)

    def predict_proba(self, x):
        y_pred_score = []
        for class_index in range(0, len(self.expand_base_estimators)):
            estimator_of_index = self.expand_base_estimators[class_index]
            y_pred_score.append(
                np.sum(
                    [estimator_of_index[0].predict(x)] +
                    [self.learning_rate * estimator_of_index[i].predict(x) for i in
                     range(1, self.n_estimators - 1)] +
                    [estimator_of_index[self.n_estimators - 1].predict(x)]
                    , axis=0)
            )
        return softmax(np.c_[y_pred_score].T)

    def predict(self, x):
        return np.argmax(self.predict_proba(x), axis=1)
