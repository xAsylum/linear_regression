from abc import abstractmethod
from array import array
from random import shuffle, sample

import numpy as np

filename = 'dane.data'


class BaseFunction:
    def __init__(self, theta):
        self.parameters = theta

    @abstractmethod
    def evaluate(self, x):
        pass
    @abstractmethod
    def diff(self, x):
        pass

    def theta(self, p):
        assert len(p) == len(self.parameters)
        self.parameters = p

class LossFunction:
    @abstractmethod
    def calculate_loss(self, base, x, y):
        pass
    @abstractmethod
    def calculate_gradient(self, base, x, y):
        pass

    def mse(self, base, planning_matrix, targets):
        assert len(planning_matrix) == len(targets)
        return np.mean([self.calculate_loss(base, planning_matrix[k], targets[k])
                        for k in range(len(targets))])

class Linear(BaseFunction):
    def evaluate(self, x):
        assert len(self.parameters) == len(x)
        return np.sum([x[i] * self.parameters[i] for i in range(len(x))])
    def diff(self, x):
        assert len(self.parameters) == len(x)
        return x

class QuadraticLossFunction(LossFunction):
    def calculate_loss(self, base, x, y):
        compute = base.evaluate(x) - y
        return compute ** 2

    def calculate_gradient(self, base, x, y):
        compute = base.evaluate(x)
        compute -= y
        vector = [x[i] * compute for i in range(len(x))]
        return vector


def normalize_features(train_data):
    mean = [np.mean(
        [train_data[i][j] for i in range(len(train_data))]
    ) for j in range(len(train_data[0]) - 1)]
    mean.append(np.float64(0))
    std = [np.mean(
        [train_data[i][j] for i in range(len(train_data))]
    ) for j in range(len(train_data[0]) - 1)]
    std.append(np.float64(1))
    data = [[(train_data[i][j] - mean[j]) / std[j]
             for j in range(len(train_data[0]))] for i in range(len(train_data))]

    return data


class LinearRegressionModel:

    def open_file(self, name):
        data = []
        with open(name, 'r') as file:
            while True:
                line = file.readline().strip()
                if not line:
                    break
                parts = line.split('\t')
                numbers = list(map(int, parts[:7]))
                float_number = float(parts[7].replace(',', '.'))
                data.append((*numbers, float_number))

        shuffle(data)
        self.raw_data = data

    def split_data(self, train_ratio, valid_ratio):
        assert train_ratio + valid_ratio < 1
        train = train_ratio
        valid = valid_ratio
        train *= len(self.raw_data)
        train = int(train)
        valid *= len(self.raw_data)
        valid = int(valid)
        valid += train
        self.train, self.validate, self.test = (
            self.raw_data[:train], self.raw_data[train:valid],
            self.raw_data[valid:])

    def __init__(self, file, normalise = True,
                 train_ratio = 0.6, valid_ratio = 0.2):
        self.raw_data = []
        self.train = []
        self.validate = []
        self.test = []
        self.open_file(file)
        self.split_data(train_ratio, valid_ratio)
        self.theta = None
        if normalise:
            self.test = normalize_features(self.test)
            self.train = normalize_features(self.train)
            self.validate = normalize_features(self.validate)

        self.eta = 0.0005
        self.baseFunction = Linear([0 for _ in range(len(self.train[0]))])
        self.lossFunction = QuadraticLossFunction()
        self.stop = 0.1
        self.mini_batch = False
        self.batch_size = 32
        pass

    def set_parameters(self, eta = None, loss_function = None,
                       base_function = None, stop = None):
        if eta is not None:
            self.eta = eta
        if loss_function is not None:
            self.lossFunction = loss_function
        if base_function is not None:
            self.baseFunction = base_function
        if stop is not None:
            self.stop = stop

        pass

    def gradient_regular(self, planning_matrix, targets):
        d = len(planning_matrix[0])
        ans = [0 for _ in range(d)]
        M = len(planning_matrix)
        for k in range(M):
            entry = self.lossFunction.calculate_gradient(self.baseFunction, planning_matrix[k], targets[k])
            for i in range(d):
                ans[i] += entry[i]

        for i in range(d):
            ans[i] /= M
        return ans


    def linear_regression(self):
        theta_size = len(self.train[0])
        theta = [[0 for _ in range(theta_size)]]
        k = 0
        planning_matrix = [
            [1, *[self.train[i][j] for j in range(len(self.train[i]) - 1)]]
            for i in range(len(self.train))
        ]
        targets = [
            self.train[i][-1]
            for i in range(len(self.train))
        ]
        while True:
            k = k + 1
            last_theta = theta[-1]
            self.baseFunction.theta(last_theta)
            if self.mini_batch:
                gradient = self.gradient_regular(planning_matrix, targets)
            else:
                mini_batch = sample(range(len(self.train)), self.batch_size)
                gradient = self.gradient_regular([planning_matrix[i] for i in mini_batch],
                                                 [targets[i] for i in mini_batch])
            theta.append([
                last_theta[i] - self.eta * gradient[i]
                for i in range(theta_size)
            ])
            #if (k % 100) == 0:
            #    print(f"{k}: Gradient: {np.linalg.norm(gradient)}, MSE: {
            #    self.lossFunction.mse(self.baseFunction, planning_matrix, targets)
            #    }")
            if np.linalg.norm(gradient) < self.stop:
                break
            if k > 2 * (10 ** 4):
                break
        self.theta = theta[-1]
        return theta[-1]

    def MSE(self):
        if self.theta is None:
            return
        self.baseFunction.theta(self.theta)

        planning_matrix = [
            [1, *[self.test[i][j] for j in range(len(self.test[i]) - 1)]]
            for i in range(len(self.test))
        ]
        targets = [
            self.test[i][-1]
            for i in range(len(self.test))
        ]

        return self.lossFunction.mse(self.baseFunction, planning_matrix, targets)

    def prediction(self, x):
        if self.theta is None:
            return
        self.baseFunction.theta(self.theta)
        return self.baseFunction.evaluate(x)







