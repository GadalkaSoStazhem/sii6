import numpy as np
import pandas as pd

class TreeNode:
    def __init__(self, label=None, feature=None):
        self.label = label
        self.feature = feature
        self.children = {}

class DecisionTreeC45:
    def __init__(self, max_depth=None, min_samples_split=2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.root = TreeNode()

    def entropy(self, y):
        unique_classes, class_counts = np.unique(y, return_counts=True)
        total_samples = len(y)
        entropy = 0

        for class_count in class_counts:
            class_probability = class_count / total_samples
            entropy -= class_probability * np.log2(class_probability)

        return entropy

    def info_gain(self, X, y, feature):
        feature_entropy = 0
        unique_values = X[feature].unique()

        for value in unique_values:
            subset = y[X[feature] == value]
            subset_entropy = self.entropy(subset)
            feature_entropy += (len(subset) / len(y)) * subset_entropy

        return self.entropy(y) - feature_entropy

    def choose_best_feature(self, X, y, features):
        best_feature = None
        best_info_gain = -1

        for feature in features:
            current_info_gain = self.info_gain(X, y, feature)
            if current_info_gain > best_info_gain:
                best_info_gain = current_info_gain
                best_feature = feature

        return best_feature

    def fit_recursive(self, X, y, node, depth):
        if len(np.unique(y)) <= 1 or depth == self.max_depth:
            # Реализация листового узла
            node.label = np.argmax(np.bincount(y))
            return

        if len(X) < self.min_samples_split:
            # Реализация листового узла
            node.label = np.argmax(np.bincount(y))
            return

        best_feature = self.choose_best_feature(X, y, X.columns)
        if best_feature is None:
            # Реализация листового узла
            node.label = np.argmax(np.bincount(y))
            return

        node.feature = best_feature
        unique_values = X[best_feature].unique()
        for value in unique_values:
            subset_indices = X[best_feature] == value
            subset_X = X[subset_indices]
            subset_y = y[subset_indices]
            child_node = TreeNode()
            node.children[value] = child_node
            self.fit_recursive(subset_X, subset_y, child_node, depth + 1)
    def fit(self, X, y):
        # Вызываем метод fit_recursive для создания дерева
        self.root = TreeNode()  # Инициализируем корень
        self.fit_recursive(X, y, self.root, depth=0)

    def predict(self, X):
        predictions = []

        for i in range(X.shape[0]):
            current_node = self.root
            while current_node.label is None:
                feature_value = X[current_node.feature].iloc[i]
                if feature_value in current_node.children:
                    current_node = current_node.children[feature_value]
                else:
                    break
            predictions.append(current_node.label)

        return predictions
