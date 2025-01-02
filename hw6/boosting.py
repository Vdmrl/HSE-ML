from __future__ import annotations

from collections import defaultdict

import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.tree import DecisionTreeRegressor

from typing import Optional

import seaborn as sns
import matplotlib.pyplot as plt

import random

from sklearn.model_selection import train_test_split


def bootstrap(X, y, subsample=1.0, bagging_temperature=1.0, bootstrap_type: str = 'Bernoulli'):
    __bootstrap_types = {"Bernoulli", "Bayesian"}

    if subsample > 1:
        raise ValueError("Subsample param couldn't be higher then 1")

    if bootstrap_type not in __bootstrap_types:
        raise ValueError(f"Unsupported bootstrap type: {bootstrap_type}")

    if isinstance(subsample, float):
        sample_size = int(X.shape[0] * subsample)
    elif isinstance(subsample, int):
        sample_size = subsample
    else:
        raise TypeError("subsample must be a float or an int.")

    if bootstrap_type == 'Bernoulli':
        index, *_ = train_test_split(np.arange(X.shape[0]), np.arange(X.shape[0]), train_size=subsample)
        ind_cur = np.random.choice(index.shape[0], index.shape[0])
        return X[ind_cur], y[ind_cur], ind_cur

    elif bootstrap_type == 'Bayesian':
        # Генерация весов объектов
        weights = (-np.log(np.random.uniform(size=X.shape[0]))) ** bagging_temperature
        # Нормализация весов и отбор объектов
        ind_cur = np.random.choice(X.shape[0], size=sample_size, replace=True, p=weights / weights.sum())
        return X[ind_cur], y[ind_cur], ind_cur
    else:
        return X, y, None


def transform_target(y):
    return 2 * y - 1


def score(clf, x, y):
    return roc_auc_score(y == 1, clf.predict_proba(x)[:, 1])


class Boosting:
    def __init__(
            self,
            base_model_class=DecisionTreeRegressor,
            base_model_params: Optional[dict] = None,
            n_estimators: int = 10,
            learning_rate: float = 0.1,
            early_stopping_rounds: int | None = 0,
            subsample: float | int = 1.0,
            bagging_temperature: float | int = 1.0,
            bootstrap_type: str | None = None,
            dart: bool | None = False,
            dropout_rate: int | float = 0.05,
            seed: int = 0
    ):
        self.base_model_class = base_model_class
        self.base_model_params: dict = {} if base_model_params is None else base_model_params

        self.n_estimators: int = n_estimators

        self.models: list = []
        self.gammas: list = []  # вес каждой модели ансамбля

        self.learning_rate: float = learning_rate

        self.history = defaultdict(list)
        # {"train_roc_auc_score": [], "train_loss": [], "val_roc_auc_score": [], "val_loss": []}

        self.sigmoid = lambda x: 1 / (1 + np.exp(-x))
        self.loss_fn = lambda y, z: -np.log(self.sigmoid(y * z)).mean()
        self.loss_derivative = lambda y, z: y * self.sigmoid(-y * z)  # Исправьте формулу на правильную.

        self.current_train_prediction = None  # a(n-1). Сохраняю старые предскзания, чтобы не пересчитывать

        self.early_stopping_rounds = early_stopping_rounds
        self.early_stopping_current_rounds = 0

        self.subsample = subsample
        self.bagging_temperature = bagging_temperature
        self.bootstrap_type = bootstrap_type
        self.bootstrap_index = None

        self.dart = dart
        self.dropout_rate = dropout_rate

        self.seed = seed

    def partial_fit(self, X, y):
        # Новая модель
        self.models.append(self.base_model_class(**self.base_model_params).fit(X, y))
        if self.bootstrap_index is None:
            self.gammas.append(self.find_optimal_gamma(y, self.current_train_prediction, self.models[-1].predict(X)))
        else:
            self.gammas.append(self.find_optimal_gamma(y, self.current_train_prediction[self.bootstrap_index], self.models[-1].predict(X)))


        # Старое предсказание
        if not self.dart:
            if self.bootstrap_index is None:
                self.current_train_prediction += self.learning_rate * self.gammas[-1] * self.models[-1].predict(X)
            else:
                self.current_train_prediction[self.bootstrap_index] += self.learning_rate * self.gammas[-1] * self.models[-1].predict(X)

        return None

    def fit(self, X_train, y_train, X_val=None, y_val=None, plot=False):
        """
        :param X_train: features array (train set)
        :param y_train: targets array (train set)
        :param X_val: features array (eval set)
        :param y_val: targets array (eval set)
        :param plot: bool 
        """
        random.seed(self.seed)
        np.random.seed(self.seed)

        # Из 0/1 в -1/1
        if y_train.min() == 0:
            y_train = transform_target(y_train)
            y_val = None if y_val is None else transform_target(y_val)

        self.current_train_prediction = np.zeros(y_train.shape[0])
        self.models = []
        self.gammas = []
        self.history = defaultdict(list)
        self.early_stopping_current_rounds = 0

        self.bernoulli_ind = None

        val_predictions = None
        if X_val is not None and y_val is not None:
            val_predictions = np.zeros(y_val.shape[0])

        shift = y_train

        for _ in range(self.n_estimators):
            # bootstrap
            if self.bootstrap_type is not None:
                X_train, y_train, self.bootstrap_index = bootstrap(
                    X_train,
                    y_train,
                    subsample=self.subsample,
                    bagging_temperature=self.bagging_temperature,
                    bootstrap_type=self.bootstrap_type
                )

            # dart
            k = 0
            if self.dart and len(self.models) > 0:
                k = int(self.dropout_rate * len(self.models))
                drop_indices = random.sample(range(len(self.models)), k)
            else:
                drop_indices = []

            # train


            if self.bootstrap_index is None:
                self.partial_fit(X_train, shift)
                shift = self.loss_derivative(y_train, self.current_train_prediction)  # новое смещение
            else:
                self.partial_fit(X_train, shift[self.bootstrap_index])
                shift = self.loss_derivative(y_train, self.current_train_prediction[self.bootstrap_index])

            # dart
            if self.dart:
                prediction = np.zeros(y_train.shape[0])
                for i, model in enumerate(self.models):
                    if i not in drop_indices:
                        prediction += (self.learning_rate * self.gammas[i] / (len(self.models) - k)) * model.predict(
                            X_train)

                self.current_train_prediction = prediction
            else:
                if self.bootstrap_index is None:
                    self.current_train_prediction += self.learning_rate * self.gammas[-1] * self.models[-1].predict(X_train)
                else:
                    self.current_train_prediction[self.bootstrap_index] += self.learning_rate * self.gammas[-1] * self.models[-1].predict(X_train)


            # validation
            if self.bootstrap_index is None:
                self.history["train_loss"].append(self.loss_fn(y_train, self.current_train_prediction))
                self.history["train_roc_auc_score"].append(
                    roc_auc_score(y_train, self.sigmoid(self.current_train_prediction)))
            else:
                self.history["train_loss"].append(self.loss_fn(y_train, self.current_train_prediction[self.bootstrap_index]))
                self.history["train_roc_auc_score"].append(
                    roc_auc_score(y_train, self.sigmoid(self.current_train_prediction[self.bootstrap_index])))
            # Считать score каждый раз заново очень затратно, так как приходиться прогонять через все модели.
            # Поэтому вместо self.score буду использовать roc_auc_score(). Ну, чтобы модель каждый раз не считать.


            # Если есть валидация, буду сразу считать на ней метрики.
            # Буду искать лучшую модель по roc_auc_score
            if X_val is not None and y_val is not None:
                if self.dart:
                    val_prediction = np.zeros(y_val.shape[0])
                    for i, model in enumerate(self.models):
                        if i not in drop_indices:
                            val_prediction += (self.learning_rate * self.gammas[i] / (
                                    len(self.models) - k)) * model.predict(X_val)
                        else:
                            val_prediction += (self.learning_rate * self.gammas[i] * k / (
                                    len(self.models) - k + 1)) * model.predict(X_val)
                    val_predictions = val_prediction
                else:
                    val_predictions += self.learning_rate * self.gammas[-1] * self.models[-1].predict(X_val)

                # val_predictions += self.learning_rate * self.gammas[-1] * self.models[-1].predict(X_val)

                current_val_score = roc_auc_score(y_val, self.sigmoid(val_predictions))

                self.history["val_roc_auc_score"].append(current_val_score)
                self.history["val_loss"].append(self.loss_fn(y_val, val_predictions))

                # Ранняя остановка. Не очень понял, нужно ли сохранять лучший результат, после n rounds ухудшений
                if (self.early_stopping_rounds is not None
                        and self.early_stopping_rounds > 0
                        and len(self.history["val_roc_auc_score"]) >= 2):
                    if self.history["val_roc_auc_score"][-1] <= self.history["val_roc_auc_score"][-2]:
                        self.early_stopping_current_rounds += 1
                    else:
                        self.early_stopping_current_rounds = 0

                    if self.early_stopping_current_rounds >= self.early_stopping_rounds:
                        # Удаляю последние self.early_stopping_rounds моделей, чтобы оставить лучший результат.
                        # Вообще лучше сохранять лучший результат, но пока остановлюсь на таком варианте
                        self.models = self.models[:-self.early_stopping_rounds]
                        self.gammas = self.gammas[:-self.early_stopping_rounds]

                        break

        if plot:
            self.plot_history()

    def predict_proba(self, X):

        prediction = np.zeros(X.shape[0])

        for i in range(len(self.models)):
            prediction += self.learning_rate * self.gammas[i] * self.models[i].predict(X)

        sigmoid = self.sigmoid(prediction)  # Накидываю сигмоиду
        return np.stack([1 - sigmoid, sigmoid], axis=1)

    def find_optimal_gamma(self, y, old_predictions, new_predictions) -> float:
        # находит такую гамму, при которой ошибка на функции потерь минимальна
        gammas = np.linspace(start=0, stop=1, num=100)
        losses = [self.loss_fn(y, old_predictions + gamma * new_predictions) for gamma in gammas]
        return gammas[np.argmin(losses)]

    def score(self, X, y):
        # не буду использовать это функцию в обучении, так как она считает всё с нуля
        return score(self, X, y)

    def plot_history(self, X=None, y=None):
        """
        делаю сразу во время обучения, поэтому удалил параметры
        """

        # Если нет параметров, то оставляю метрики с обучения (если тогда были данные для валидации, то использую их)
        if X is not None and y is not None:
            self.history["val_roc_auc_score"] = []
            self.history["val_loss"] = []

            prediction = np.zeros_like(y, dtype=np.float64)

            for i in range(len(self.models)):
                prediction += self.learning_rate * self.gammas[i] * self.models[i].predict(X)

                # Опять же, если использовать self.score, то вычисления будут повторятся
                self.history["val_roc_auc_score"].append(roc_auc_score(y, self.sigmoid(prediction)))
                self.history["val_loss"].append(self.loss_fn(y, prediction))

        plt.figure(figsize=(14, 8))
        plt.subplot(2, 1, 1)
        sns.lineplot(
            x=list(range(1, len(self.history["train_roc_auc_score"]) + 1)),
            y=self.history["train_roc_auc_score"],
            label="Train ROC AUC score"

        )
        if "val_roc_auc_score" in self.history.keys():
            sns.lineplot(
                x=list(range(1, len(self.history["val_roc_auc_score"]) + 1)),
                y=self.history["val_roc_auc_score"],
                label="Validation ROC AUC score"
            )
        plt.title("ROC AUC Scores Over Models")
        plt.xlabel("Number of models")
        plt.ylabel("ROC AUC Score")
        plt.legend()
        plt.grid(True)
        plt.subplot(2, 1, 2)
        sns.lineplot(
            x=list(range(1, len(self.history["train_loss"]) + 1)),
            y=self.history["train_loss"],
            label="Train Loss"
        )
        if "val_loss" in self.history.keys():
            sns.lineplot(
                x=list(range(1, len(self.history["val_loss"]) + 1)),
                y=self.history["val_loss"],
                label="Validation Loss"
            )
        plt.title("Loss Over Models")
        plt.xlabel("Number of models")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    @property
    def feature_importances_(self):
        importances = np.zeros(self.models[0].feature_importances_.shape[0])
        for model in self.models:
            importances += model.feature_importances_

        importances /= self.models[0].feature_importances_.shape[0]
        return importances
