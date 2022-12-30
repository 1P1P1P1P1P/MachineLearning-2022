from Algrithm.methods import *
from Algrithm.PLA import PLA


class Pocket(PLA):
    def __init__(self, steps=200):
        super().__init__()
        self.w = None
        self.steps = steps

    def errorPoints(self, w):  # 用来判断当前w值对应的分类错误
        error_Points = np.array([])
        X = self.train_data
        Y = self.train_label
        for i in range(X.shape[0]):
            if np.dot(X[i], w) * Y[i] <= 0:
                error_Points = np.append(error_Points, i)
        return error_Points.shape[0]

    def train(self, x=None, y=None):
        if x is not None and y is not None:
            self.train_data = np.hstack((x, np.ones((x.shape[0], 1))))
            y[y == 0] = -1
            self.train_label = y
        X = self.train_data.copy()
        Y = self.train_label.copy()
        w = np.zeros((X.shape[1], 1))
        self.w = w.copy()
        errors = self.errorPoints(w)
        step = 0
        steps = self.steps
        itr = 0
        while step <= steps:
            step += 1
            error_each = np.random.choice(errors)
            w += np.dot(Y[error_each], X[error_each]).reshape(X.shape[1], 1)
            errors = self.errorPoints(w)
            best_errors = self.errorPoints(self.w)
            if errors <= best_errors:
                self.w = w.copy()
                itr += 1
            if errors == 0:
                break
        print(f"更新次数为{step}")
        print(f"W_BEST更新次数为{itr}")
        print(f"最终W为{self.w.flatten()}")
        return self

