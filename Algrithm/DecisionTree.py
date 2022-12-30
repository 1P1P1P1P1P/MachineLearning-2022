from Algrithm.methods import *


class DataBinWrapper(object):
    def __init__(self, max_bins=10):
        # 分段数
        self.max_bins = max_bins
        # 记录x各个特征的分段区间
        self.XrangeMap = None

    def fit(self, x):
        n_sample, n_feature = x.shape
        # 构建分段数据
        self.XrangeMap = [[] for _ in range(0, n_feature)]
        for index in range(0, n_feature):
            tmp = x[:, index]
            for percent in range(1, self.max_bins):
                percent_value = np.percentile(tmp, (1.0 * percent / self.max_bins) * 100.0 // 1)
                self.XrangeMap[index].append(percent_value)

    def transform(self, x):
        if x.ndim == 1:
            return np.asarray([np.digitize(x[i], self.XrangeMap[i]) for i in range(0, x.size)])
        else:
            return np.asarray([np.digitize(x[:, i], self.XrangeMap[i]) for i in range(0, x.shape[1])]).T


# ID3和C4.5决策树分类器
class DecisionTreeClassifier(object):
    class Node(object):  # 树节点，用于存储节点信息以及关联子节点
        def __init__(self, feature_index: int = None, target_distribute: dict = None, weight_distribute: dict = None,
                     children_nodes: dict = None, num_sample: int = None):
            self.feature_index = feature_index
            self.target_distribute = target_distribute
            self.weight_distribute = weight_distribute
            self.children_nodes = children_nodes
            self.num_sample = num_sample

    def __init__(self, criterion='c4.5', max_depth=None, min_samples_split=2, min_samples_leaf=1,
                 min_impurity_decrease=0, max_bins=10):
        """
        :param criterion:划分标准，包括id3,c4.5，默认为c4.5
        :param max_depth:树的最大深度
        :param min_samples_split:当对一个内部结点划分时，要求该结点上的最小样本数，默认为2
        :param min_samples_leaf:设置叶子结点上的最小样本数，默认为1
        :param min_impurity_decrease:打算划分一个内部结点时，只有当划分后不纯度(可以用criterion参数指定的度量来描述)减少值不小于该参数指定的值，才会对该结点进行划分，默认值为0
        """
        self.criterion = criterion
        if criterion == 'c4.5':
            self.criterion_func = info_gain_rate
        elif criterion == 'id3':
            self.criterion_func = muti_info
        else:
            self.criterion_func = info_gain_rate
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_impurity_decrease = min_impurity_decrease

        self.root_node = None
        self.sample_weight = None
        self.dbw = DataBinWrapper(max_bins=max_bins)

    def _build_tree(self, current_depth, current_node, x, y, sample_weight, prePrune=False):
        rows, cols = x.shape
        # 计算y分布以及其权重分布
        target_distribute = {}
        weight_distribute = {}
        for index, tmp_value in enumerate(y):
            if tmp_value not in target_distribute:
                target_distribute[tmp_value] = 0.0
                weight_distribute[tmp_value] = []
            target_distribute[tmp_value] += 1.0
            weight_distribute[tmp_value].append(sample_weight[index])
        for key, value in target_distribute.items():
            target_distribute[key] = value / rows
            weight_distribute[key] = np.mean(weight_distribute[key])
        current_node.target_distribute = target_distribute
        current_node.weight_distribute = weight_distribute
        current_node.num_sample = rows
        # 判断停止切分的条件

        if len(target_distribute) <= 1:
            return

        if rows < self.min_samples_split:
            return

        if self.max_depth is not None and current_depth > self.max_depth:
            return

        # 寻找最佳的特征
        best_index = None
        best_criterion_value = 0
        for index in range(0, cols):
            criterion_value = self.criterion_func(x[:, index], y)
            if criterion_value > best_criterion_value:
                best_criterion_value = criterion_value
                best_index = index

        # 如果criterion_value减少不够则停止
        if best_index is None:
            return
        if best_criterion_value <= self.min_impurity_decrease:
            return
        # 切分
        current_node.feature_index = best_index
        children_nodes = {}
        current_node.children_nodes = children_nodes
        selected_x = x[:, best_index]
        for item in set(selected_x):
            selected_index = np.where(selected_x == item)
            # 如果切分后的点太少，以至于都不能做叶子节点，则停止分割
            if len(selected_index[0]) < self.min_samples_leaf:
                continue
            child_node = self.Node()
            children_nodes[item] = child_node
            self._build_tree(current_depth + 1, child_node, x[selected_index], y[selected_index],
                             sample_weight[selected_index], prePrune)
        # 预剪枝实现，当prePrune is True是说明采用预剪枝
        if len(current_node.children_nodes) != 0 and prePrune:
            alpha = 1.5
            # 计算剪枝前的损失值
            pre_prune_value = alpha
            for key, value in current_node.target_distribute.items():
                pre_prune_value += -1 * current_node.num_sample * value * np.log(
                    value) * current_node.weight_distribute.get(key, 1.0)
            # 计算剪枝后的损失值
            after_prune_value = alpha * len(current_node.children_nodes)
            for child_node in current_node.children_nodes.values():
                for key, value in child_node.target_distribute.items():
                    after_prune_value += -1 * child_node.num_sample * value * np.log(
                        value) * child_node.weight_distribute.get(key, 1.0)
            if after_prune_value >= pre_prune_value:
                # 剪枝操作
                current_node.children_nodes = None
                current_node.feature_index = None

    def train(self, x, y, sample_weight=None, prePrune=False):
        n_sample = x.shape[0]
        if sample_weight is None:
            self.sample_weight = np.asarray([1.0] * n_sample)
        else:
            self.sample_weight = sample_weight
        # check sample_weight
        if len(self.sample_weight) != n_sample:
            raise Exception('sample_weight size error:', len(self.sample_weight))
        # 构建空的根节点
        self.root_node = self.Node()
        # 对x分箱
        self.dbw.fit(x)
        # 递归构建树
        self._build_tree(1, self.root_node, self.dbw.transform(x), y, self.sample_weight, prePrune)
        return self

    # 检索叶子节点的结果
    def _search_node(self, current_node: Node, x, class_num):
        if current_node.feature_index is None or current_node.children_nodes is None or len(
                current_node.children_nodes) == 0 or current_node.children_nodes.get(
            x[current_node.feature_index]) is None:
            result = []
            total_value = 0.0
            for index in range(0, class_num):
                value = current_node.target_distribute.get(index, 0) * current_node.weight_distribute.get(index, 1.0)
                result.append(value)
                total_value += value
            # 归一化
            for index in range(0, class_num):
                result[index] = result[index] / total_value
            return result
        else:
            return self._search_node(current_node.children_nodes.get(x[current_node.feature_index]), x, class_num)

    def _prune_node(self, current_node: Node, alpha):
        # 如果有子结点,先对子结点部分剪枝
        if current_node.children_nodes is not None and len(current_node.children_nodes) != 0:
            for child_node in current_node.children_nodes.values():
                self._prune_node(child_node, alpha)
        # 再尝试对当前结点剪枝
        if current_node.children_nodes is not None and len(current_node.children_nodes) != 0:
            # 避免跳层剪枝
            for child_node in current_node.children_nodes.values():
                # 当前剪枝的层必须是叶子结点的层
                if child_node.children_nodes is not None and len(child_node.children_nodes) > 0:
                    return
            # 计算剪枝前的损失值
            pre_prune_value = alpha * len(current_node.children_nodes)
            for child_node in current_node.children_nodes.values():
                for key, value in child_node.target_distribute.items():
                    pre_prune_value += -1 * child_node.num_sample * value * np.log(
                        value) * child_node.weight_distribute.get(key, 1.0)
            # 计算剪枝后的损失值
            after_prune_value = alpha
            for key, value in current_node.target_distribute.items():
                after_prune_value += -1 * current_node.num_sample * value * np.log(
                    value) * current_node.weight_distribute.get(key, 1.0)
            if after_prune_value <= pre_prune_value:
                # 剪枝操作
                current_node.children_nodes = None
                current_node.feature_index = None

    # 决策树剪枝
    def prune(self, alpha=0.01):
        self._prune_node(self.root_node, alpha)

    def predict_proba(self, x):
        # 计算结果概率分布
        x = self.dbw.transform(x)
        rows = x.shape[0]
        results = []
        class_num = len(self.root_node.target_distribute)
        for row in range(0, rows):
            results.append(self._search_node(self.root_node, x[row], class_num))
        return np.asarray(results)

    def predict(self, x):
        return np.argmax(self.predict_proba(x), axis=1)


# CART分类树
class CARTClassifier(object):
    class Node(object):
        def __init__(self, feature_index: int = None, feature_value=None, target_distribute: dict = None,
                     weight_distribute: dict = None,
                     left_child_node=None, right_child_node=None, num_sample: int = None):
            self.feature_index = feature_index
            self.feature_value = feature_value
            self.target_distribute = target_distribute
            self.weight_distribute = weight_distribute
            self.left_child_node = left_child_node
            self.right_child_node = right_child_node
            self.num_sample = num_sample

    def __init__(self, criterion='gini', max_depth=None, min_samples_split=2, min_samples_leaf=1,
                 min_impurity_decrease=0, max_bins=10):
        """
        :param criterion:划分标准，默认为gini,另外entropy表示用信息增益比
        :param max_depth:树的最大深度
        :param min_samples_split:当对一个内部结点划分时，要求该结点上的最小样本数，默认为2
        :param min_samples_leaf:设置叶子结点上的最小样本数，默认为1
        :param min_impurity_decrease:打算划分一个内部结点时，只有当划分后不纯度(可以用criterion参数指定的度量来描述)减少值不小于该参数指定的值，才会对该结点进行划分，默认值为0
        """
        self.criterion = criterion
        if criterion == 'gini':
            self.criterion_func = gini_gain
        else:
            self.criterion_func = info_gain_rate
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_impurity_decrease = min_impurity_decrease

        self.root_node = None
        self.dbw = DataBinWrapper(max_bins=max_bins)

    def _build_tree(self, current_depth, current_node: Node, x, y, sample_weight):
        rows, cols = x.shape
        # 计算y分布以及其权重分布
        target_distribute = {}
        weight_distribute = {}
        for index, tmp_value in enumerate(y):
            if tmp_value not in target_distribute:
                target_distribute[tmp_value] = 0.0
                weight_distribute[tmp_value] = []
            target_distribute[tmp_value] += 1.0
            weight_distribute[tmp_value].append(sample_weight[index])
        for key, value in target_distribute.items():
            target_distribute[key] = value / rows
            weight_distribute[key] = np.mean(weight_distribute[key])
        current_node.target_distribute = target_distribute
        current_node.weight_distribute = weight_distribute
        current_node.num_sample = rows
        # 判断停止切分的条件

        if len(target_distribute) <= 1:
            return

        if rows < self.min_samples_split:
            return

        if self.max_depth is not None and current_depth > self.max_depth:
            return

        # 寻找最佳的特征以及取值
        best_index = None
        best_index_value = None
        best_criterion_value = 0
        for index in range(0, cols):
            for index_value in set(x[:, index]):
                criterion_value = self.criterion_func((x[:, index] == index_value).astype(int), y, sample_weight)
                if criterion_value > best_criterion_value:
                    best_criterion_value = criterion_value
                    best_index = index
                    best_index_value = index_value

        # 如果criterion_value减少不够则停止
        if best_index is None:
            return
        if best_criterion_value <= self.min_impurity_decrease:
            return
        # 切分
        current_node.feature_index = best_index
        current_node.feature_value = best_index_value
        selected_x = x[:, best_index]

        # 创建左孩子结点
        left_selected_index = np.where(selected_x == best_index_value)
        # 如果切分后的点太少，以至于都不能做叶子节点，则停止分割
        if len(left_selected_index[0]) >= self.min_samples_leaf:
            left_child_node = self.Node()
            current_node.left_child_node = left_child_node
            self._build_tree(current_depth + 1, left_child_node, x[left_selected_index], y[left_selected_index],
                             sample_weight[left_selected_index])
        # 创建右孩子结点
        right_selected_index = np.where(selected_x != best_index_value)
        # 如果切分后的点太少，以至于都不能做叶子节点，则停止分割
        if len(right_selected_index[0]) >= self.min_samples_leaf:
            right_child_node = self.Node()
            current_node.right_child_node = right_child_node
            self._build_tree(current_depth + 1, right_child_node, x[right_selected_index], y[right_selected_index],
                             sample_weight[right_selected_index])

    def train(self, x, y, sample_weight=None):
        # check sample_weight
        n_sample = x.shape[0]
        if sample_weight is None:
            sample_weight = np.asarray([1.0] * n_sample)
        # check sample_weight
        if len(sample_weight) != n_sample:
            raise Exception('sample_weight size error:', len(sample_weight))

        # 构建空的根节点
        self.root_node = self.Node()
        # 对x分箱
        self.dbw.fit(x)
        # 递归构建树
        self._build_tree(1, self.root_node, self.dbw.transform(x), y, sample_weight)
        return self

    # 检索叶子节点的结果
    def _search_node(self, current_node: Node, x, class_num):
        if current_node.left_child_node is not None and x[current_node.feature_index] == current_node.feature_value:
            return self._search_node(current_node.left_child_node, x, class_num)
        elif current_node.right_child_node is not None and x[current_node.feature_index] != current_node.feature_value:
            return self._search_node(current_node.right_child_node, x, class_num)
        else:
            result = []
            total_value = 0.0
            for index in range(0, class_num):
                value = current_node.target_distribute.get(index, 0) * current_node.weight_distribute.get(index, 1.0)
                result.append(value)
                total_value += value
            # 归一化
            for index in range(0, class_num):
                result[index] = result[index] / total_value
            return result

    def _prune_node(self, current_node: Node, alpha):
        # 如果有子结点,先对子结点部分剪枝
        if current_node.left_child_node is not None:
            self._prune_node(current_node.left_child_node, alpha)
        if current_node.right_child_node is not None:
            self._prune_node(current_node.right_child_node, alpha)
        # 再尝试对当前结点剪枝
        if current_node.left_child_node is not None or current_node.right_child_node is not None:
            # 避免跳层剪枝
            for child_node in [current_node.left_child_node, current_node.right_child_node]:
                # 当前剪枝的层必须是叶子结点的层
                if child_node.left_child_node is not None or child_node.right_child_node is not None:
                    return
            # 计算剪枝的前的损失值
            pre_prune_value = alpha * 2
            for child_node in [current_node.left_child_node, current_node.right_child_node]:
                for key, value in child_node.target_distribute.items():
                    pre_prune_value += -1 * child_node.num_sample * value * np.log(
                        value) * child_node.weight_distribute.get(key, 1.0)
            # 计算剪枝后的损失值
            after_prune_value = alpha
            for key, value in current_node.target_distribute.items():
                after_prune_value += -1 * current_node.num_sample * value * np.log(
                    value) * current_node.weight_distribute.get(key, 1.0)

            if after_prune_value <= pre_prune_value:
                # 剪枝操作
                current_node.left_child_node = None
                current_node.right_child_node = None
                current_node.feature_index = None
                current_node.feature_value = None

    def prune(self, alpha=0.01):
        self._prune_node(self.root_node, alpha)

    def predict(self, x):
        return np.argmax(self.predict_proba(x), axis=1)

    def predict_proba(self, x):
        x = self.dbw.transform(x)
        rows = x.shape[0]
        results = []
        class_num = len(self.root_node.target_distribute)
        for row in range(0, rows):
            results.append(self._search_node(self.root_node, x[row], class_num))
        return np.asarray(results)


# CART回归树
class CARTRegressor(object):
    class Node(object):
        def __init__(self, feature_index: int = None, feature_value=None, y_hat=None, square_error=None,
                     left_child_node=None, right_child_node=None, num_sample: int = None):
            self.feature_index = feature_index
            self.feature_value = feature_value
            self.y_hat = y_hat
            self.square_error = square_error
            self.left_child_node = left_child_node
            self.right_child_node = right_child_node
            self.num_sample = num_sample

    def __init__(self, criterion='mse', max_depth=None, min_samples_split=2, min_samples_leaf=1, min_std=1e-3,
                 min_impurity_decrease=0, max_bins=10):
        self.criterion = criterion
        if criterion == 'mse':
            self.criterion_func = square_error_gain
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_std = min_std
        self.min_impurity_decrease = min_impurity_decrease

        self.root_node = None
        self.dbw = DataBinWrapper(max_bins=max_bins)

    def _build_tree(self, current_depth, current_node: Node, x, y, sample_weight):
        rows, cols = x.shape
        # 计算当前y的加权平均值
        current_node.y_hat = np.dot(sample_weight / sum(sample_weight), y)
        current_node.num_sample = rows
        # 判断停止切分的条件
        current_node.square_error = np.dot(y - np.mean(y), y - np.mean(y))
        if np.sqrt(current_node.square_error / rows) <= self.min_std:
            return

        if rows < self.min_samples_split:
            return

        if self.max_depth is not None and current_depth > self.max_depth:
            return

        # 寻找最佳的特征以及取值
        best_index = None
        best_index_value = None
        best_criterion_value = 0
        for index in range(0, cols):
            for index_value in sorted(set(x[:, index])):
                criterion_value = self.criterion_func((x[:, index] <= index_value).astype(int), y, sample_weight)
                if criterion_value > best_criterion_value:
                    best_criterion_value = criterion_value
                    best_index = index
                    best_index_value = index_value

        # 如果criterion_value减少不够则停止
        if best_index is None:
            return
        if best_criterion_value <= self.min_impurity_decrease:
            return
        # 切分
        current_node.feature_index = best_index
        current_node.feature_value = best_index_value
        selected_x = x[:, best_index]

        # 创建左孩子结点
        left_selected_index = np.where(selected_x <= best_index_value)
        # 如果切分后的点太少，以至于都不能做叶子节点，则停止分割
        if len(left_selected_index[0]) >= self.min_samples_leaf:
            left_child_node = self.Node()
            current_node.left_child_node = left_child_node
            self._build_tree(current_depth + 1, left_child_node, x[left_selected_index], y[left_selected_index],
                             sample_weight[left_selected_index])
        # 创建右孩子结点
        right_selected_index = np.where(selected_x > best_index_value)
        # 如果切分后的点太少，以至于都不能做叶子节点，则停止分割
        if len(right_selected_index[0]) >= self.min_samples_leaf:
            right_child_node = self.Node()
            current_node.right_child_node = right_child_node
            self._build_tree(current_depth + 1, right_child_node, x[right_selected_index], y[right_selected_index],
                             sample_weight[right_selected_index])

    def train(self, x, y, sample_weight=None):
        n_sample = x.shape[0]
        if sample_weight is None:
            sample_weight = np.asarray([1.0] * n_sample)
        # check sample_weight
        if len(sample_weight) != n_sample:
            raise Exception('sample_weight size error:', len(sample_weight))

        # 构建空的根节点
        self.root_node = self.Node()

        # 对x分箱
        self.dbw.fit(x)

        # 递归构建树
        self._build_tree(1, self.root_node, self.dbw.transform(x), y, sample_weight)
        return self

    # 检索叶子节点的结果
    def _search_node(self, current_node: Node, x):
        if current_node.left_child_node is not None and x[current_node.feature_index] <= current_node.feature_value:
            return self._search_node(current_node.left_child_node, x)
        elif current_node.right_child_node is not None and x[current_node.feature_index] > current_node.feature_value:
            return self._search_node(current_node.right_child_node, x)
        else:
            return current_node.y_hat

    def predict(self, x):
        # 计算结果概率分布
        x = self.dbw.transform(x)
        rows = x.shape[0]
        results = []
        for row in range(0, rows):
            results.append(self._search_node(self.root_node, x[row]))
        return np.asarray(results)

    def _prune_node(self, current_node: Node, alpha):
        # 如果有子结点,先对子结点部分剪枝
        if current_node.left_child_node is not None:
            self._prune_node(current_node.left_child_node, alpha)
        if current_node.right_child_node is not None:
            self._prune_node(current_node.right_child_node, alpha)
        # 再尝试对当前结点剪枝
        if current_node.left_child_node is not None or current_node.right_child_node is not None:
            # 避免跳层剪枝
            for child_node in [current_node.left_child_node, current_node.right_child_node]:
                # 当前剪枝的层必须是叶子结点的层
                if child_node.left_child_node is not None or child_node.right_child_node is not None:
                    return
            # 计算剪枝的前的损失值
            pre_prune_value = alpha * 2 + \
                              (
                                  0.0 if current_node.left_child_node.square_error is None else current_node.left_child_node.square_error) + \
                              (
                                  0.0 if current_node.right_child_node.square_error is None else current_node.right_child_node.square_error)
            # 计算剪枝后的损失值
            after_prune_value = alpha + current_node.square_error

            if after_prune_value <= pre_prune_value:
                # 剪枝操作
                current_node.left_child_node = None
                current_node.right_child_node = None
                current_node.feature_index = None
                current_node.feature_value = None
                current_node.square_error = None

    def prune(self, alpha=0.01):
        self._prune_node(self.root_node, alpha)
