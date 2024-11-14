from dataclasses import dataclass
from enum import auto
from enum import Enum
from typing import Dict
from typing import Type

import numpy as np


@dataclass
class LearningRate:
    lambda_: float = 1e-3
    s0: float = 1
    p: float = 0.5

    iteration: int = 0

    def __call__(self):
        """
        Calculate learning rate according to lambda (s0/(s0 + t))^p formula
        """
        self.iteration += 1
        return self.lambda_ * (self.s0 / (self.s0 + self.iteration)) ** self.p


class LossFunction(Enum):
    MSE = auto()
    MAE = auto()
    LogCosh = auto()
    Huber = auto()


class BaseDescent:
    """
    A base class and templates for all functions
    """

    def __init__(self, dimension: int, lambda_: float = 1e-3, loss_function: LossFunction = LossFunction.MSE,
                 d: float = 1):
        """
        :param dimension: feature space dimension
        :param lambda_: learning rate parameter
        :param loss_function: optimized loss function
        """
        self.w: np.ndarray = np.random.rand(dimension)
        self.lr: LearningRate = LearningRate(lambda_=lambda_)
        self.loss_function: LossFunction = loss_function
        self.d = d

    def step(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        return self.update_weights(self.calc_gradient(x, y))

    def update_weights(self, gradient: np.ndarray) -> np.ndarray:
        """
        Template for update_weights function
        Update weights with respect to gradient
        :param gradient: gradient
        :return: weight difference (w_{k + 1} - w_k): np.ndarray
        """
        pass

    def calc_gradient(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Template for calc_gradient function
        Calculate gradient of loss function with respect to weights
        :param x: features array
        :param y: targets array
        :return: gradient: np.ndarray
        """
        pass

    def calc_loss(self, x: np.ndarray, y: np.ndarray) -> float:
        """
        Calculate loss for x and y with our weights
        :param x: features array
        :param y: targets array
        :return: loss: float
        """

        if self.loss_function == LossFunction.MSE:
            return 1 / y.shape[0] * (y - x.dot(self.w)).T.dot(y - x.dot(self.w))
        elif self.loss_function == LossFunction.MAE:
            return np.mean(np.abs(x.dot(self.w) - y))
        elif self.loss_function == LossFunction.LogCosh:
            return np.mean(np.log(np.cosh(x.dot(self.w) - y)))
        elif self.loss_function == LossFunction.Huber:
            if np.linalg.norm(y - x.dot(self.w)) <= self.d:
                return 1 / (2 * y.shape[0]) * (y - x.dot(self.w)).T.dot((y - x.dot(self.w)))
            else:
                return -np.sum(self.d * (np.abs(x.dot(self.w) - y) - 1 / 2 * self.d)) / y.shape[0]

    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        Calculate predictions for x
        :param x: features array
        :return: prediction: np.ndarray
        """
        return x.dot(self.w)


class VanillaGradientDescent(BaseDescent):
    """
    Full gradient descent class
    """

    def update_weights(self, gradient: np.ndarray) -> np.ndarray:
        """
        Update weights with respect to gradient
        :return: weight difference (w_{k + 1} - w_k): np.ndarray
        """
        difference = np.negative(self.lr() * gradient)
        self.w = self.w + difference
        return difference

    def calc_gradient(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        if self.loss_function == LossFunction.MSE:
            return -2 / x.shape[0] * x.T.dot((y - (x.dot(self.w))))
        elif self.loss_function == LossFunction.MAE:
            return x.T.dot(np.sign(x.dot(self.w) - y)) / x.shape[0]
        elif self.loss_function == LossFunction.LogCosh:
            return x.T.dot(np.tanh(x.dot(self.w) - y)) / x.shape[0]
        elif self.loss_function == LossFunction.Huber:
            if np.linalg.norm(y - x.dot(self.w)) <= self.d:
                residual = y - x.dot(self.w)
                return 1 / x.shape[0] * x.T.dot(residual)
            else:
                residual = np.sign(x.dot(self.w) - y)
                return x.T.dot(residual) * self.d / x.shape[0]


class StochasticDescent(VanillaGradientDescent):
    """
    Stochastic gradient descent class
    """

    def __init__(self, dimension: int, lambda_: float = 1e-3, batch_size: int = 50,
                 loss_function: LossFunction = LossFunction.MSE):
        """
        :param batch_size: batch size (int)
        """
        super().__init__(dimension, lambda_, loss_function)
        self.batch_size = batch_size

    def calc_gradient(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        batch_index = np.random.randint(low=0, high=len(y), size=self.batch_size)
        x_batch = x[batch_index]
        y_batch = y[batch_index]

        if self.loss_function == LossFunction.MSE:
            return 2 / self.batch_size * x_batch.T.dot((x_batch.dot(self.w) - y_batch))
        elif self.loss_function == LossFunction.MAE:
            return x_batch.T.dot(np.sign(x_batch.dot(self.w - y_batch))) / self.batch_size
        elif self.loss_function == LossFunction.LogCosh:
            return x_batch.T.dot(np.tanh(x_batch.dot(self.w) - y_batch)) / self.batch_size
        elif self.loss_function == LossFunction.Huber:
            if np.linalg.norm(y_batch - x_batch.dot(self.w)) <= self.d:
                return 1 / self.batch_size * x_batch.T.dot(y_batch - (x_batch.dot(self.w)))
            else:
                return self.d * x_batch.T.dot(np.sign(x_batch.dot(self.w) - y_batch)) / self.batch_size


class MomentumDescent(VanillaGradientDescent):
    """
    Momentum gradient descent class
    """

    def __init__(self, dimension: int, lambda_: float = 1e-3, loss_function: LossFunction = LossFunction.MSE):
        super().__init__(dimension, lambda_, loss_function)
        self.alpha: float = 0.9

        self.h: np.ndarray = np.zeros(dimension)

    def update_weights(self, gradient: np.ndarray) -> np.ndarray:
        """
        Update weights with respect to gradient
        :return: weight difference (w_{k + 1} - w_k): np.ndarray
        """
        self.h = self.alpha * self.h + self.lr() * gradient
        difference = np.negative(self.h)
        self.w = self.w + difference
        return difference


class Adam(VanillaGradientDescent):
    """
    Adaptive Moment Estimation gradient descent class
    """

    def __init__(self, dimension: int, lambda_: float = 1e-3, loss_function: LossFunction = LossFunction.MSE):
        super().__init__(dimension, lambda_, loss_function)
        self.eps: float = 1e-8

        self.m: np.ndarray = np.zeros(dimension)
        self.v: np.ndarray = np.zeros(dimension)

        self.beta_1: float = 0.9
        self.beta_2: float = 0.999

        self.iteration: int = 0

    def update_weights(self, gradient: np.ndarray) -> np.ndarray:
        """
        Update weights & params
        :return: weight difference (w_{k + 1} - w_k): np.ndarray
        """

        self.m = self.beta_1 * self.m + (1 - self.beta_1) * gradient
        self.v = self.beta_2 * self.v + (1 - self.beta_2) * np.power(gradient, 2)

        m_hat = self.m / (1 - np.power(self.beta_1, self.iteration + 1))
        v_hat = self.v / (1 - np.power(self.beta_2, self.iteration + 1))

        difference = np.negative(self.lr() * m_hat / (np.sqrt(v_hat) + self.eps))
        self.w = self.w + difference

        self.iteration += 1

        return difference


class Nadam(Adam):

    def update_weights(self, gradient: np.ndarray) -> np.ndarray:
        """
        Update weights & params
        :return: weight difference (w_{k + 1} - w_k): np.ndarray
        """
        self.m = self.beta_1 * self.m + (1 - self.beta_1) * gradient
        self.v = self.beta_2 * self.v + (1 - self.beta_2) * np.power(gradient, 2)

        self.iteration += 1

        bias_correction_m = (1 - np.power(self.beta_1, self.iteration))
        bias_correction_v = (1 - np.power(self.beta_2, self.iteration))

        denominator = (np.sqrt(self.v / bias_correction_v) + self.eps)

        weight_updated = (self.m * self.beta_1 / bias_correction_m +
                          (1 - self.beta_1) / bias_correction_m * gradient)

        difference = np.negative(self.lr() / denominator * weight_updated)
        self.w = self.w + difference

        return difference


class BaseDescentReg(BaseDescent):
    """
    A base class with regularization
    """

    def __init__(self, *args, mu: float = 0, **kwargs):
        """
        :param mu: regularization coefficient (float)
        """
        super().__init__(*args, **kwargs)

        self.mu = mu

    def calc_gradient(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Calculate gradient of loss function and L2 regularization with respect to weights
        """

        l2_gradient: np.ndarray = self.w  # градиент регуляризация разделить на 2 - это просто веса
        l2_gradient[-1] = 0  # bias
        return super().calc_gradient(x, y) + l2_gradient * self.mu


class VanillaGradientDescentReg(BaseDescentReg, VanillaGradientDescent):
    """
    Full gradient descent with regularization class
    """


class StochasticDescentReg(BaseDescentReg, StochasticDescent):
    """
    Stochastic gradient descent with regularization class
    """


class MomentumDescentReg(BaseDescentReg, MomentumDescent):
    """
    Momentum gradient descent with regularization class
    """


class AdamReg(BaseDescentReg, Adam):
    """
    Adaptive gradient algorithm with regularization class
    """


def get_descent(descent_config: dict) -> BaseDescent:
    descent_name = descent_config.get('descent_name', 'full')
    regularized = descent_config.get('regularized', False)

    descent_mapping: Dict[str, Type[BaseDescent]] = {
        'full': VanillaGradientDescent if not regularized else VanillaGradientDescentReg,
        'stochastic': StochasticDescent if not regularized else StochasticDescentReg,
        'momentum': MomentumDescent if not regularized else MomentumDescentReg,
        'adam': Adam if not regularized else AdamReg,
        'nadam': Nadam
    }

    if descent_name not in descent_mapping:
        raise ValueError(f'Incorrect descent name, use one of these: {descent_mapping.keys()}')

    descent_class = descent_mapping[descent_name]

    return descent_class(**descent_config.get('kwargs', {}))
