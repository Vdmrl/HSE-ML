import numpy as np
from collections import Counter
from typing import Callable


def entropy(k):
    # из ОМО
    k = np.array(k)
    total_amount = np.sum(k)
    probabilities = k[np.where(k != 0)] / total_amount
    entropy_values = -probabilities * np.log(
        probabilities)
    return round(np.sum(entropy_values), 2)


def gini(r):
    # из ОМО
    r = np.array(r)
    total_amount = np.sum(r)
    probabilities = r[np.where(r != 0)] / total_amount
    gini_values = probabilities * (1 - probabilities)
    return np.sum(gini_values)


def find_best_split(feature_vector, target_vector):
    """
    Оптимизированная версия функции поиска лучшего разбиения.

    :param feature_vector: вещественнозначный вектор значений признака
    :param target_vector: вектор классов объектов, len(feature_vector) == len(target_vector)

    :return thresholds: отсортированный по возрастанию вектор со всеми возможными порогами
    :return ginis: вектор со значениями критерия Джини для каждого из порогов в thresholds
    :return threshold_best: оптимальный порог (число)
    :return gini_best: оптимальное значение критерия Джини (число)
    """
    if np.unique(feature_vector).shape[0] <= 1:
        return -np.inf, -np.inf, -np.inf, -np.inf

    # Сортировка признаков и таргетов
    feature_vector = np.array(feature_vector)
    target_vector = np.array(target_vector)
    ind = feature_vector.argsort()
    sorted_feature = feature_vector[ind]
    sorted_target = target_vector[ind]

    # Вычисление уникальных порогов
    unique_features, index = np.unique(sorted_feature[::-1], return_index=True)
    index = sorted_feature.shape[0] - 1 - index[:-1]
    thresholds = (unique_features[:-1] + unique_features[1:]) / 2

    # Размеры выборок
    R = feature_vector.shape[0]
    R_L = index + 1
    R_R = R - R_L

    # Расчёт вероятностей классов
    cumsum_target = np.cumsum(sorted_target)
    p_1_L = cumsum_target[index] / R_L
    p_1_R = (cumsum_target[-1] - cumsum_target[index]) / R_R

    # Энтропии для левой и правой частей
    H_L = 1 - p_1_L ** 2 - (1 - p_1_L) ** 2
    H_R = 1 - p_1_R ** 2 - (1 - p_1_R) ** 2

    # Критерий Джини
    ginis = -R_L / R * H_L - R_R / R * H_R

    # Поиск оптимального порога
    max_gini_index = np.argmax(ginis)
    threshold_best = thresholds[max_gini_index]
    gini_best = ginis[max_gini_index]

    return thresholds, ginis, threshold_best, gini_best


class DecisionTree():
    def __init__(self, feature_types, max_depth=None, min_samples_split=None, min_samples_leaf=1):
        if np.any(list(map(lambda x: x != "real" and x != "categorical", feature_types))):
            raise ValueError("There is unknown feature type")

        self._tree = {}
        self._feature_types = feature_types
        self._max_depth = max_depth
        self._min_samples_split = min_samples_split
        self._min_samples_leaf = min_samples_leaf

    def _fit_node(self, sub_X, sub_y, node):

        if ((not (self._min_samples_split is None)) and len(sub_y) < self._min_samples_split) or (
                (not (self._max_depth is None)) and node["layer"] + 1 > self._max_depth):
            node["type"] = "terminal"
            node["class"] = Counter(sub_y).most_common(1)[0][0]
            return

        if np.all(sub_y == sub_y[0]):
            node["type"] = "terminal"
            node["class"] = sub_y[0]
            return

        if len(self._tree) == 0:
            node["layer"] = 1

        feature_best, threshold_best, gini_best, split = None, None, None, None
        for feature in range(sub_X.shape[1]):
            feature_type = self._feature_types[feature]
            categories_map = {}

            if feature_type == "real":
                feature_vector = sub_X[:, feature]
            elif feature_type == "categorical":
                counts = Counter(sub_X[:, feature])
                clicks = Counter(sub_X[sub_y == 1, feature])
                ratio = {}

                for key, current_count in counts.items():
                    if key in clicks:
                        current_click = clicks[key]
                    else:
                        current_click = 0

                    ratio[key] = current_click / current_count

                sorted_categories = list(map(lambda x: x[0], sorted(ratio.items(), key=lambda x: x[1])))
                categories_map = dict(zip(sorted_categories, list(range(len(sorted_categories)))))
                feature_vector = np.array(list(map(lambda x: categories_map[x], sub_X[:, feature])))
            else:
                raise ValueError

            if np.all(feature_vector == feature_vector[0]):
                continue

            _, _, threshold, gini = find_best_split(feature_vector, sub_y)
            if (gini_best is None or gini > gini_best) and gini is not None:
                feature_best = feature
                gini_best = gini
                split = feature_vector < threshold

                if feature_type == "real":
                    threshold_best = threshold
                elif feature_type == "categorical":
                    threshold_best = set((map(lambda x: x[0],
                                              filter(lambda x: x[1] < threshold, categories_map.items()))))
                else:
                    raise ValueError

        if feature_best is None:
            node["type"] = "terminal"
            node["class"] = Counter(sub_y).most_common(1)[0][0]
            return

        node["type"] = "nonterminal"

        node["feature_split"] = feature_best
        if self._feature_types[feature_best] == "real":
            node["threshold"] = threshold_best
        elif self._feature_types[feature_best] == "categorical":
            node["categories_split"] = threshold_best
        else:
            raise ValueError

        if len(sub_y[split]) < self._min_samples_leaf or len(sub_y[np.logical_not(split)]) < self._min_samples_leaf:
            node["type"] = "terminal"
            node["class"] = Counter(sub_y).most_common(1)[0][0]
            return

        node["left_child"], node["right_child"] = {"layer": node["layer"] + 1}, {"layer": node["layer"] + 1}

        self._fit_node(sub_X[split], sub_y[split], node["left_child"])
        self._fit_node(sub_X[np.logical_not(split)], sub_y[np.logical_not(split)], node["right_child"])

    def _predict_node(self, x, node):
        if node["type"] == "terminal":
            return node["class"]

        feature_split = node["feature_split"]

        if self._feature_types[feature_split] == "real":
            if x[feature_split] < node["threshold"]:
                return self._predict_node(x, node["left_child"])

            return self._predict_node(x, node["right_child"])
        elif self._feature_types[feature_split] == "categorical":
            if x[feature_split] in node["categories_split"]:
                return self._predict_node(x, node["left_child"])

            return self._predict_node(x, node["right_child"])

    def fit(self, X, y):
        X[np.isnan(X)] = 0  # Заполню пропуски нулями для их обработки
        self._fit_node(X, y, self._tree)

    def predict(self, X):
        X[np.isnan(X)] = 0  # Заполню пропуски нулями для их обработки
        predicted = []
        for x in X:
            predicted.append(self._predict_node(x, self._tree))
        return np.array(predicted)


def find_best_split_regression(feature_vector, target_vector):
    """
    Finds the best split for a regression task by minimizing the weighted variance of the split.

    :param feature_vector: Continuous-valued feature vector
    :param target_vector: Continuous target vector, len(feature_vector) == len(target_vector)

    :return thresholds: Sorted vector with all possible thresholds
    :return weighted_variances: Vector with weighted variances for each threshold
    :return threshold_best: Optimal threshold (number)
    :return variance_best: Optimal weighted variance (number)
    """
    if np.unique(feature_vector).shape[0] <= 1:
        return -np.inf, -np.inf, -np.inf, -np.inf

    # Преобразование входных данных в numpy массивы
    feature_vector = np.array(feature_vector)
    target_vector = np.array(target_vector)

    # Сортировка признаков и целевых значений по значениям признака
    ind = feature_vector.argsort()
    sorted_feature = feature_vector[ind]
    sorted_target = target_vector[ind]

    # Вычисление уникальных порогов
    unique_features, index = np.unique(sorted_feature[::-1], return_index=True)
    index = sorted_feature.shape[0] - 1 - index[:-1]
    thresholds = (unique_features[:-1] + unique_features[1:]) / 2

    # Размер выборки
    R = feature_vector.shape[0]
    R_L = index + 1
    R_R = R - R_L

    # Кумулятивные суммы целевых значений и их квадратов
    cumsum_target = np.cumsum(sorted_target)
    cumsum_target_squared = np.cumsum(sorted_target ** 2)

    # Средние и дисперсии для левой группы
    mean_L = cumsum_target[index] / R_L
    variance_L = cumsum_target_squared[index] / R_L - mean_L ** 2

    # Средние и дисперсии для правой группы
    mean_R = (cumsum_target[-1] - cumsum_target[index]) / R_R
    variance_R = (cumsum_target_squared[-1] - cumsum_target_squared[index]) / R_R - mean_R ** 2

    # Взвешенные дисперсии
    weighted_variances = R_L / R * variance_L + R_R / R * variance_R

    # Поиск оптимального порога (минимизация взвешенной дисперсии)
    min_variance_index = np.argmin(weighted_variances)
    threshold_best = thresholds[min_variance_index]
    variance_best = weighted_variances[min_variance_index]

    return thresholds, weighted_variances, threshold_best, variance_best


from sklearn.linear_model import LinearRegression


class LinearRegressionTree():
    def __init__(self, feature_types, max_depth=None, min_samples_split=None, min_samples_leaf=1):
        if np.any(list(map(lambda x: x != "real" and x != "categorical", feature_types))):
            raise ValueError("There is unknown feature type")

        self._tree = {}
        self._feature_types = feature_types
        self._max_depth = max_depth
        self._min_samples_split = min_samples_split
        self._min_samples_leaf = min_samples_leaf

    def _fit_node(self, sub_X, sub_y, node):
        if ((not (self._min_samples_split is None)) and len(sub_y) < self._min_samples_split) or (
                (not (self._max_depth is None)) and node["layer"] + 1 > self._max_depth):
            node["type"] = "terminal"
            node["class"] = Counter(sub_y).most_common(1)[0][0]
            return

        if np.all(sub_y == sub_y[0]):
            node["type"] = "terminal"
            node["class"] = sub_y[0]
            return

        if len(self._tree) == 0:
            node["layer"] = 1

        feature_best, threshold_best, gini_best, split = None, None, None, None
        for feature in range(sub_X.shape[1]):
            feature_type = self._feature_types[feature]
            categories_map = {}

            if feature_type == "real":
                feature_vector = sub_X[:, feature]
            elif feature_type == "categorical":
                counts = Counter(sub_X[:, feature])
                clicks = Counter(sub_X[sub_y == 1, feature])
                ratio = {}

                for key, current_count in counts.items():
                    if key in clicks:
                        current_click = clicks[key]
                    else:
                        current_click = 0

                    ratio[key] = current_click / current_count

                sorted_categories = list(map(lambda x: x[0], sorted(ratio.items(), key=lambda x: x[1])))
                categories_map = dict(zip(sorted_categories, list(range(len(sorted_categories)))))
                feature_vector = np.array(list(map(lambda x: categories_map[x], sub_X[:, feature])))
            else:
                raise ValueError

            if np.all(feature_vector == feature_vector[0]):
                continue

            _, _, threshold, gini = find_best_split_regression(feature_vector, sub_y)
            if (gini_best is None or gini > gini_best) and gini is not None:
                feature_best = feature
                gini_best = gini
                split = feature_vector < threshold

                if feature_type == "real":
                    threshold_best = threshold
                elif feature_type == "categorical":
                    threshold_best = set((map(lambda x: x[0],
                                              filter(lambda x: x[1] < threshold, categories_map.items()))))
                else:
                    raise ValueError

        if feature_best is None:
            node["type"] = "terminal"
            node["model"] = LinearRegression().fit(sub_X, sub_y)
            return

        node["type"] = "nonterminal"

        node["feature_split"] = feature_best
        if self._feature_types[feature_best] == "real":
            node["threshold"] = threshold_best
        elif self._feature_types[feature_best] == "categorical":
            node["categories_split"] = threshold_best
        else:
            raise ValueError

        if len(sub_y[split]) < self._min_samples_leaf or len(sub_y[np.logical_not(split)]) < self._min_samples_leaf:
            node["type"] = "terminal"
            node["model"] = LinearRegression().fit(sub_X, sub_y)
            return

        node["left_child"], node["right_child"] = {"layer": node["layer"] + 1}, {"layer": node["layer"] + 1}

        self._fit_node(sub_X[split], sub_y[split], node["left_child"])
        self._fit_node(sub_X[np.logical_not(split)], sub_y[np.logical_not(split)], node["right_child"])

    def _predict_node(self, x, node):
        if isinstance(x, (int, float)):  # If x is a scalar, wrap it in a list
            x = [x]

        if node["type"] == "terminal":
            return node["model"].predict([x])[0]

        feature_split = node["feature_split"]

        if self._feature_types[feature_split] == "real":
            if x[feature_split] < node["threshold"]:
                return self._predict_node(x, node["left_child"])

            return self._predict_node(x, node["right_child"])
        elif self._feature_types[feature_split] == "categorical":
            if x[feature_split] in node["categories_split"]:
                return self._predict_node(x, node["left_child"])

            return self._predict_node(x, node["right_child"])

    def fit(self, X, y):
        X[np.isnan(X)] = 0  # Заполню пропуски нулями для их обработки
        self._fit_node(X, y, self._tree)

    def predict(self, X):
        X[np.isnan(X)] = 0  # Заполню пропуски нулями для их обработки
        predicted = []
        for x in X:
            predicted.append(self._predict_node(x, self._tree))
        return np.array(predicted)
