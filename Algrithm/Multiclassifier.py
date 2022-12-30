from Algrithm.methods import *
from Algrithm import LogisticClassification
import copy


class MultiClassifier:
    def __init__(self, base_classifier=LogisticClassification, train_data=None, train_label=None, mode='ova',
                 show_Process=True):
        """
        多分类实现
        :param train_data: 训练数据
        :param train_label: 训练数据标签
        :param base_classifier: 多分类器使用的基分类器的实例
        :param mode: 分类模式，'ova'表示One-VS-All, 'ovo'表示One-VS-One
        """
        self.predict_results = None
        self.classifiers = None
        self.n_class = None
        self.train_data = train_data
        if self.train_data is not None:
            self.train_data = np.hstack((np.ones((self.train_data.shape[0], 1)), self.train_data))
        self.train_label = train_label
        self.base_classifier = base_classifier()
        self.mode = mode
        self.show_Process = show_Process
        self.Process = 0

    def base_classifier_train(self, base_classifier, x, y):
        if self.show_Process:
            self.Process += 1
            cls = base_classifier.train(x, y)
            plot_decision_function(x, y, cls, fname=f"process_{self.Process}")
        return base_classifier.train(x, y)

    @staticmethod
    def predict_proba_base_classifier(base_classifier, x):
        return base_classifier.predict_proba(dataTest=x)

    def train(self, x=None, y=None):
        if x is not None and y is not None:
            self.train_data = np.hstack((np.ones((x.shape[0], 1)), x))
            self.train_label = y
        else:
            x = self.train_data.copy()
            y = self.train_label.copy()
        self.n_class = int(max(y)) + 1  # 确认数据类别数量
        if self.mode == 'ova':  # 判断分类方式为 'ova'
            self.classifiers = []
            for cls in range(0, self.n_class):
                # 对y进行了判断
                self.classifiers.append(self.base_classifier_train(copy.deepcopy(self.base_classifier), x,
                                                                   np.asarray([[1] if i == cls else [-1] for i in y])))
        elif self.mode == "ovo":  # 判断分类方式为 'ovo'
            self.classifiers = {}
            for first_cls in range(0, self.n_class - 1):  # 每一次将两个类别的数据进行训练
                for second_cls in range(first_cls + 1, self.n_class):
                    # 获取对应类别的数据
                    index = np.where(y == first_cls)[0].tolist() + np.where(y == second_cls)[0].tolist()
                    new_x = x[index, :]
                    new_y = y[index]
                    self.classifiers[(first_cls, second_cls)] = copy.deepcopy(
                        self.base_classifier_train(self.base_classifier, new_x,
                                                   np.asarray([[1] if i == first_cls else [-1] for i in new_y])))
        self.predict_results = self.predict(dataTest=x).reshape(x.shape[0], 1)
        return self

    def predict_proba(self, x=None):
        if x is None:
            x = self.train_data.copy()
        if self.mode == 'ova':
            probas = []
            for cls in self.classifiers:
                probas.append(self.predict_proba_base_classifier(cls, x))
            total_probas = np.concatenate(probas, axis=1)
            # 归一化
            return total_probas / total_probas.sum(axis=1, keepdims=True)
        elif self.mode == 'ovo':
            probas = {}
            for first_cls in range(0, self.n_class - 1):
                for second_cls in range(first_cls + 1, self.n_class):
                    probas[(first_cls, second_cls)] = self.predict_proba_base_classifier(
                        self.classifiers[(first_cls, second_cls)], x)
                    probas[(second_cls, first_cls)] = 1.0 - probas[(first_cls, second_cls)]
            # 统计概率
            total_probas = []
            for first_cls in range(0, self.n_class):  # 所有两两比较中，是first_class的概率之和
                temp = []
                for second_cls in range(0, self.n_class):
                    if first_cls != second_cls:
                        temp.append(probas[(first_cls, second_cls)])
                temp = np.concatenate(temp, axis=1).sum(axis=1, keepdims=True)
                total_probas.append(temp)
            # 归一化
            total_probas = np.concatenate(total_probas, axis=1)
            return total_probas / total_probas.sum(axis=1, keepdims=True)

    def predict(self, dataTest):
        results = np.argmax(self.predict_proba(x=dataTest), axis=1)
        return results

