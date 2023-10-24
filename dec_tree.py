import numpy as np
from preps import *
class Tree_Node(): #value для листа, остальное для десижн нод
    def __init__(self, value = None, children = None, feature_idx = None, info = None):
        self.value = value
        self.children = children
        self.feature_idx = feature_idx #indeks dlya priznaka
        self.info = info


class Decision_Tree_New():
    def __init__(self, max_depth = 6, min_samples_s = 3):
        self.max_depth = max_depth
        self.min_samples_s = min_samples_s
        self.root = None
    def fit(self, X, y):
        df = np.concatenate([X, y], axis=1)
        self.root = self.create_tree(df)

    def create_tree(self, df, current_level = 0):
        rows = df.shape[0]
        n_features = df.shape[1] - 1
        if rows >= self.min_samples_s and current_level <= self.max_depth:
            splitted_tree = self.get_best_split(df, n_features)
            limits = np.unique(df[:, splitted_tree['feature_idx']])  # Получите limits здесь
            childrens_dereva = []
            for lim in limits:
                child = df[df[:, splitted_tree['feature_idx']] == lim]
                childrens_dereva.append(self.create_tree(child, current_level + 1))
            return Tree_Node(None, childrens_dereva, splitted_tree['feature_idx'], splitted_tree['info'])
        leaf_val = leaf(df[:, -1])
        return Tree_Node(value=leaf_val)
    def get_best_split(self, df, n_features):
        splitted = {}
        info_gains = self.count_info_gain(df, n_features)
        sorted_gains = np.argsort(info_gains)[::-1]
        max_info_gain_id = sorted_gains[0]
        max_info_gain_feature = df[:, max_info_gain_id]
        limits = np.unique(max_info_gain_feature)
        children = []
        for lim in limits:
            child = df[df[:, max_info_gain_id] == lim]

            children.append(child)

        splitted['children'] = children
        splitted['feature_idx'] = max_info_gain_id
        splitted['info'] = max(info_gains)
        return splitted

    def entropy(self, feature, y, val):
        uniq_y = np.unique(y)
        val_cnt = np.sum(feature == val)
        tmp = 0
        for u_y in uniq_y:
            class_prob = np.sum(np.logical_and(feature == val, y == u_y)) / val_cnt
            tmp -= class_prob * np.log2(class_prob + 1e-15)

        entropy = (val_cnt / len(feature)) * tmp
        return entropy

    def count_info_gain(self, df, n_features):
        info_gains = []
        y = df[:, -1]  # Столбец меток классов
        for idx in range(n_features):
            gain = 0
            feature = df[:, idx]
            uniq_vals = np.unique(feature)
            for val in uniq_vals:
                gain += self.entropy(feature, y, val)
            info_gains.append(gain)
        return info_gains

    def predictor(self, x, tree):
        if tree.value is None:
            val = x[tree.feature_idx]
            uniqs = np.unique(x[tree.feature_idx])
            for i in range(len(uniqs)):
                result = self.predictor(x, tree.children[i])
                if result is not None:
                    return result
        return tree.value

    def predict(self, X):
        y_pred = [self.predictor(x, self.root) for x in X]
        return y_pred

