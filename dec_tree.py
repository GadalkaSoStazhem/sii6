import pandas as pd
import numpy as np

class Tree_Node(): #value is for leaf, other is for decision node
    def __init__(self, value = None, left = None, right = None, feature_idx = None, feature_limit = None, info = None):
        self.value = value
        self.left = left
        self.right = right
        self.feature_idx = feature_idx
        self.feature_limit = feature_limit
        self.info = info


class Decision_Tree():
    def __init__(self, max_depht = 2, min_samples = 1):
        self.max_depth = max_depht
        self.min_samples = min_samples
        self.root = None

    def tree_splitter(self, X, y, rows, feature_cnt):
        splitted = {}
        info_max = -1000000

        for idx in range(feature_cnt):
            vals = X[:, idx]
            lims = np.unique(vals)
            for lim in lims:
                left_x = np.array([ftr for ftr in X if float(ftr[idx]) <= float(lim)])
                right_x = np.array([ftr for ftr in X if float(ftr[idx]) > float(lim)])
                if len(left_x) != 0 and len(right_x) != 0:

                    left_y = np.array([ftr for ftr in y if float(ftr[idx]) <= float(lim)])
                    right_y = np.array([ftr for ftr in y if float(ftr[idx]) > float(lim)])
                    info_val = self.info_count(y, left_y, right_y)
                    if info_val > info_max:
                        splitted['left_x'] = left_x
                        splitted['left_y'] = left_y
                        splitted['right_x'] = right_x
                        splitted['right_y'] = right_y
                        splitted['feature_idx'] = idx
                        splitted['feature_limit'] = lim
                        splitted['info'] = info_val
                        info_max = info_val

            return splitted
    def info_count(self, parent, c_l, c_r):
        uniq_p = np.unique(parent)
        uniq_c_l = np.unique(c_l)
        uniq_c_r = np.unique(c_r)
        ent_p = 0
        ent_c_l = 0
        ent_c_r = 0
        for el in uniq_p:
            pres_class = len(parent[parent == el]) / len(parent)
            ent_p += -pres_class * np.log2(pres_class)
        for el in uniq_c_l:
            pres_class = len(c_l[c_l == el]) / len(c_l)
            ent_c_l += -pres_class * np.log2(pres_class)
        for el in uniq_c_r:
            pres_class = len(c_r[c_r == el]) / len(c_r)
            ent_c_r += -pres_class * np.log2(pres_class)
        return ent_p - (len(c_l) / len(parent)) * ent_c_l - (len(c_r) / len(parent)) * ent_c_r

    def create_tree(self, X, y, current_level = 0):
        rows = X.shape[0]
        features_cnt = X.shape[1]

        if rows >= self.min_samples and current_level <= self.max_depth:
            splitted = self.tree_splitter(X, y, rows, features_cnt)
            if splitted['info'] > 0:
                tree_left = self.create_tree(splitted['left_x'], splitted['left_y'], current_level + 1)
                tree_right = self.create_tree(splitted['right_x'], splitted['right_y'], current_level + 1)

                return Tree_Node(None, tree_left, tree_right, splitted['feature_idx'], splitted['feature_limit'], splitted['info'])

        y_n = list(y)
        val_leaf = max(y_n, key=y_n.count)
        return Tree_Node(value=val_leaf)

    def fit(self, X, y):
        self.root = self.create_tree(X, y)


    def predictor(self, x, trained_tree):
        if trained_tree.value == None:
            val = x[trained_tree.feature_idx]
            if val <= trained_tree.feature_limit:
                return self.predictor(x, trained_tree.left)
            else:
                return self.predictor(x, trained_tree.right)

        return trained_tree.value

    def predict(self, X):
        y_pred = [self.predictor(x, self.root) for x in X]
        return y_pred