import os
import matplotlib.pyplot as plt
from sklearn import datasets
import math
import numpy as np


def sign(X):
    """
    符号函数
    :param X:
    :return:
    """
    if X > 0:
        return 1
    elif X < 0:
        return -1
    else:
        return 0


def sigmoid(z):
    return 1.0 / (1 + np.exp(-z))


def standardization(x):
    """
    标准化
    :param x:
    :return:
    """
    x = (x - np.mean(x, axis=0)) / np.std(x, axis=0)
    return x


def softmax(x):
    if x.ndim == 1:
        return np.exp(x) / np.exp(x).sum()
    else:
        return np.exp(x) / np.exp(x).sum(axis=1, keepdims=True)


def dataGenerated_for_2class(data_size, scale=0.5, random_state=None):
    """
    生成随机数据
    :param scale:
    :param data_size: 生成一类数据的大小, tuple
    :param random_state: 随机种子
    :return:
    """
    if isinstance(data_size, int):
        data_size = (data_size, data_size)
    np.random.seed(random_state)

    points_loc = [[1, 6], [10, 3]]
    data_x = np.vstack(
        (np.random.normal(points_loc[0], scale, data_size), np.random.normal(points_loc[1], scale, data_size)))
    data_y = np.vstack((np.ones((data_size[0], 1)), np.ones((data_size[0], 1)) * -1))
    return data_x, data_y


# 针对多分类
def dataGenerated_for_multiclass(data_size, scale=0.5, class_num=2, random_state=None):
    if isinstance(data_size, int):
        data_size = (data_size, data_size)
    np.random.seed(random_state)

    data_x = np.empty((0, data_size[1]))
    data_y = np.empty((0, 1))
    for i in range(class_num):
        data_x = np.vstack((data_x, np.random.normal([np.random.randint(0, 10) for _ in range(data_size[1])], scale,
                                                     data_size)))  # 添加常数项，对应的x为一
        data_y = np.vstack((data_y, np.ones((data_size[0], 1)) * i))
    return data_x, data_y


# 分类画图方法
def plot_decision_function(X, y, clf, support_vectors=None, fname=None, plot_step=0.02, centers=False):
    save_path = "./image"
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, plot_step),
                         np.arange(y_min, y_max, plot_step))

    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, alpha=0.4)
    plt.scatter(X[:, 0], X[:, 1], alpha=0.8, c=y, edgecolor='k')
    # 绘制支持向量
    if support_vectors is not None:
        plt.scatter(X[support_vectors, 0], X[support_vectors, 1], s=80, c='none', alpha=0.7, edgecolor='red')
    # 绘制质心
    if centers:
        plt.scatter(clf.centers[:, 0], clf.centers[:, 1], c='red', marker='o')
    if fname is not None:
        plt.title(fname)
        save_name = save_path + "/" + fname + '.svg'
        plt.savefig(save_name, format='svg')
    plt.show()


# 计算熵的函数
def entropy(x, sample_weight=None):
    x = np.asarray(x)
    # x中元素个数
    x_num = len(x)
    # 如果sample_weight为None设均设置一样
    if sample_weight is None:
        sample_weight = np.asarray([1.0] * x_num)
    x_counter = {}
    weight_counter = {}
    # 统计各x取值出现的次数以及其对应的sample_weight列表
    for index in range(0, x_num):
        x_value = x[index]
        if x_counter.get(x_value) is None:
            x_counter[x_value] = 0
            weight_counter[x_value] = []
        x_counter[x_value] += 1
        weight_counter[x_value].append(sample_weight[index])
    ent = .0
    for key, value in x_counter.items():
        p_i = 1.0 * value * np.mean(weight_counter.get(key)) / x_num
        ent += -p_i * math.log(p_i)
    return ent


# 计算条件熵:H(y|x)
def cond_entropy(x, y, sample_weight=None):
    x = np.asarray(x)
    y = np.asarray(y)
    # x中元素个数
    x_num = len(x)
    # 如果sample_weight为None设均设置一样
    if sample_weight is None:
        sample_weight = np.asarray([1.0] * x_num)
    # 计算
    ent = .0
    for x_value in set(x):
        x_index = np.where(x == x_value)
        new_x = x[x_index]
        new_y = y[x_index]
        new_sample_weight = sample_weight[x_index]
        p_i = 1.0 * len(new_x) / x_num
        ent += p_i * entropy(new_y, new_sample_weight)
    return ent


# 信息增益:H(y)-H(y|x) ID3算法采用
def muti_info(x, y, sample_weight=None):
    x_num = len(x)
    if sample_weight is None:
        sample_weight = np.asarray([1.0] * x_num)
    return entropy(y, sample_weight) - cond_entropy(x, y, sample_weight)


# 信息增益率 C4.5算法
def info_gain_rate(x, y, sample_weight=None):
    x_num = len(x)
    if sample_weight is None:
        sample_weight = np.asarray([1.0] * x_num)
    return 1.0 * muti_info(x, y, sample_weight) / (1e-12 + entropy(x, sample_weight))


# 计算基尼系数 Gini(D)
def gini(x, sample_weight=None):
    x_num = len(x)
    # 如果sample_weight为None设均设置一样
    if sample_weight is None:
        sample_weight = np.asarray([1.0] * x_num)
    x_counter = {}
    weight_counter = {}
    # 统计各x取值出现的次数以及其对应的sample_weight列表
    for index in range(0, x_num):
        x_value = x[index]
        if x_counter.get(x_value) is None:
            x_counter[x_value] = 0
            weight_counter[x_value] = []
        x_counter[x_value] += 1
        weight_counter[x_value].append(sample_weight[index])

    # 计算gini系数
    gini_value = 1.0
    for key, value in x_counter.items():
        p_i = 1.0 * value * np.mean(weight_counter.get(key)) / x_num
        gini_value -= p_i * p_i
    return gini_value


# 计算条件gini系数:Gini(y,x)
def cond_gini(x, y, sample_weight=None):
    x = np.asarray(x)
    y = np.asarray(y)
    # x中元素个数
    x_num = len(x)
    # 如果sample_weight为None设均设置一样
    if sample_weight is None:
        sample_weight = np.asarray([1.0] * x_num)
    # 计算
    gini_value = .0
    for x_value in set(x):
        x_index = np.where(x == x_value)
        new_x = x[x_index]
        new_y = y[x_index]
        new_sample_weight = sample_weight[x_index]
        p_i = 1.0 * len(new_x) / x_num
        gini_value += p_i * gini(new_y, new_sample_weight)
    return gini_value


# gini值的增益
def gini_gain(x, y, sample_weight=None):
    x_num = len(x)
    if sample_weight is None:
        sample_weight = np.asarray([1.0] * x_num)
    return gini(y, sample_weight) - cond_gini(x, y, sample_weight)


# 平方误差
def square_error(x, sample_weight=None):
    x = np.asarray(x)
    x_mean = np.mean(x)
    x_num = len(x)
    if sample_weight is None:
        sample_weight = np.asarray([1.0] * x_num)
    error = 0.0
    for index in range(0, x_num):
        error += (x[index] - x_mean) * (x[index] - x_mean) * sample_weight[index]
    return error


# 计算按x分组的y的误差值
def cond_square_error(x, y, sample_weight=None):
    x = np.asarray(x)
    y = np.asarray(y)
    # x中元素个数
    x_num = len(x)
    # 如果sample_weight为None设均设置一样
    if sample_weight is None:
        sample_weight = np.asarray([1.0] * x_num)
    # 计算
    error = .0
    for x_value in set(x):
        x_index = np.where(x == x_value)
        new_y = y[x_index]
        new_sample_weight = sample_weight[x_index]
        error += square_error(new_y, new_sample_weight)
    return error


# 平方误差带来的增益值
def square_error_gain(x, y, sample_weight=None):
    x_num = len(x)
    if sample_weight is None:
        sample_weight = np.asarray([1.0] * x_num)
    return square_error(y, sample_weight) - cond_square_error(x, y, sample_weight)


def Euclidean(x1, x2):
    return np.sqrt(np.sum(np.power((x1 - x2), 2), axis=1))


def Manhattan(x1, x2):
    return np.sum(np.abs(x1 - x2), axis=1)


def Minkowski(x1, x2, p):
    return np.power(np.sum(np.power(np.abs(x1 - x2), p), axis=1), 1 / p)


def Chebyshev(x1, x2):
    return np.max(np.abs(x1 - x2), axis=1)


def CosDistance(x1, x2):
    return np.sum(x1 * x2, axis=1, dtype=float) / (
                Euclidean(x1, np.zeros(x1.shape)) * Euclidean(x2, np.zeros(x2.shape)))
