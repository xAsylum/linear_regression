from abc import abstractmethod
from array import array
from os.path import split
import matplotlib.pyplot as plt
from random import shuffle, sample

import numpy as np
from enum import Enum

filename = 'dane.data'


class StopCondition(Enum):
    Gradient = 1,
    Iterations = 2,
    Both = 3

class BaseFunction:
    def __init__(self):
        self.parameters = None

    @abstractmethod
    def evaluate(self, x):
        pass
    @abstractmethod
    def diff(self, x):
        pass

    def theta(self, p):
        if self.parameters is not None:
            assert len(p) == len(self.parameters)
        self.parameters = p

class LossFunction:
    @abstractmethod
    def calculate_loss(self, base, x, y):
        pass
    @abstractmethod
    def calculate_gradient(self, base, x, y):
        pass

class Linear(BaseFunction):
    def evaluate(self, x):
        assert len(self.parameters) == len(x)
        return np.sum([x[i] * self.parameters[i] for i in range(len(x))])
    def diff(self, x):
        # assert len(self.parameters) == len(x)
        return x
def sgn(x):
    if x > 0:
        return 1
    return -1

# x -> [sgn(x_i) * |x_i|^p]
class Uninomial(BaseFunction):
    def __init__(self, p):
        self.pow = p
        super().__init__()

    def evaluate(self, x):
        assert len(self.parameters) == len(x)
        return np.sum([sgn(x[i]) * (abs(x[i])** self.pow) * self.parameters[i] for i in range(len(x))])
    def diff(self, x):
        # assert len(self.parameters) == len(x)
        return [sgn(x[i]) * (abs(x[i]) ** self.pow) for i in range(len(x))]

class Zeros(BaseFunction):
    def __init__(self, zeros):
        super().__init__()
        self.take = []
        for i in range(len(zeros)):
            if zeros[i] != 0:
                self.take.append(i)
        self.parameters = [0 for _ in range(len(self.take) + 1)]

    def evaluate(self, x):
        return self.parameters[0] + sum([self.parameters[i + 1] * x[self.take[i]] for i in range(len(self.take))])

    def diff(self, x):
        return [1, *[x[self.take[i]] for i in range(len(self.take))]]

class Gaussian(BaseFunction):
    def __init__(self, s):
        self.s =s
        super().__init__()

    def evaluate(self, x):
        return self.parameters[0] +  np.sum([self.parameters[i + 1] * np.exp(-((x[i+1] / self.s) ** 2)) for i in range(len(x) - 1)])

    def diff(self, x):
        return [1, *np.exp([(-((x[i+1] / self.s) ** 2) ) for i in range(len(x) - 1)])]


def gauss(x, s):
    return np.exp(-((x / s) ** 2))

class Custom(BaseFunction):
    def __init__(self, help_func = None):
        super().__init__()
        self.helpFunc = help_func
        if self.helpFunc is None:
            self.helpFunc = Linear()
        self.func = lambda x: [*[x[i] for i in range(8)],
                      *[x[i] * x[j] for i in range(1, 8) for j in range(i, 8)],
                      *[x[i] * x[j] * x[k] for i in range(1, 8) for j in range(i, 8) for k in range(j, 8)]]
        self.parameters = [0 for _ in range(len(self.func([0 for _ in range(8)])))]
        self.helpFunc.parameters = self.parameters


    def setdiff(self, func):
        self.func = func
        self.parameters = [0 for _ in range(len(self.func([0 for _ in range(8)])))]
        self.helpFunc.parameters = self.parameters
    def evaluate(self, x):
        self.helpFunc.theta(self.parameters)
        return self.helpFunc.evaluate(self.diff(x))


    def diff(self, x):
        if self.helpFunc is None:
            return self.func(x)
        return self.helpFunc.diff(self.func(x))

class QuadraticLossFunction(LossFunction):
    def calculate_loss(self, base, planning_matrix, targets):

        assert len(planning_matrix) == len(targets)
        m = len(planning_matrix)
        total = 0
        for k in range(m):
            x = planning_matrix[k]
            y = targets[k]
            compute = base.evaluate(x) - y
            compute **= 2
            total += compute
        total /= 2 * m
        return total

    def calculate_gradient(self, base, planning_matrix, targets):
        m = len(planning_matrix)
        t = len(base.parameters)
        ans = [0 for _ in range(t)]
        for k in range(m):
            diff = base.diff(planning_matrix[k])
            calc = (base.evaluate(planning_matrix[k]) - targets[k])
            for i in range(t):
                ans[i] += diff[i] * calc
        for i in range(t):
            ans[i] /= m
        return ans

class RidgeRegressionFunction(LossFunction):
    def __init__(self, alpha):
        self.alpha = alpha

    def calculate_loss(self, base, planning_matrix, targets):
        assert len(planning_matrix) == len(targets)
        quadratic = QuadraticLossFunction()
        t = len(base.parameters)
        total = quadratic.calculate_loss(base, planning_matrix, targets)
        total += self.alpha * np.sum([(i > 0) *
                base.parameters[i] ** 2 for i in range(t)])
        return total

    def calculate_gradient(self, base, planning_matrix, targets):
        quadratic = QuadraticLossFunction()
        t = len(base.parameters)
        ans = quadratic.calculate_gradient(base, planning_matrix, targets)
        for i in range(t):
            ans[i] += (i > 0) * 2 * self.alpha * base.parameters[i]
        return ans

class LassoRegressionFunction(LossFunction):
    def __init__(self, beta):
        self.beta = beta

    def calculate_loss(self, base, planning_matrix, targets):
        assert len(planning_matrix) == len(targets)
        quadratic = QuadraticLossFunction()
        t = len(base.parameters)
        total = quadratic.calculate_loss(base, planning_matrix, targets)
        total += self.beta * np.sum([(i > 0) *
                                      abs(base.parameters[i]) for i in range(t)])
        return total

    def calculate_gradient(self, base, planning_matrix, targets):
        assert len(planning_matrix) == len(targets)
        quadratic = QuadraticLossFunction()
        t = len(base.parameters)
        ans = quadratic.calculate_gradient(base, planning_matrix, targets)
        for i in range(t):
            ans[i] += (i > 0) * self.beta * sgn(base.parameters[i])
        return ans

class ElasticNetworkLossFunction(LossFunction):
    def __init__(self, alpha, beta):
        self.alpha = alpha
        self.beta = beta

    def calculate_loss(self, base, planning_matrix, targets):
        assert len(planning_matrix) == len(targets)
        quadratic = QuadraticLossFunction()
        t = len(base.parameters)
        total = quadratic.calculate_loss(base, planning_matrix, targets)
        total += self.alpha * np.sum([(i > 0) *
                                      base.parameters[i] ** 2 for i in range(t)])
        total += self.beta * np.sum([(i > 0) *
                                      abs(base.parameters[i]) for i in range(t)])


        return total

    def calculate_gradient(self, base, planning_matrix, targets):
        assert len(planning_matrix) == len(targets)
        quadratic = QuadraticLossFunction()
        t = len(base.parameters)
        ans = quadratic.calculate_gradient(base, planning_matrix, targets)
        for i in range(t):
            ans[i] += (i > 0) * 2 * self.alpha * base.parameters[i]
            ans[i] += (i > 0) * self.beta * sgn(base.parameters[i])
        return ans


class LinearRegressionModel:

    def solve_analytically(self, alpha = 0.0):
        if self.split:
            data = sample(self.train, int(self.fraction * len(self.train)))
        else:
            data = self.train

        planning_matrix = np.array([
            [1, *[data[i][j] for j in range(len(data[i]) - 1)]]
            for i in range(len(data))
        ])

        targets = np.array([data[i][-1] for i in range(len(data))])
        X = planning_matrix
        y = targets
        XtX = (X.T @ X) + (alpha * np.eye(len(planning_matrix[0])))
        Xty = X.T @ y
        self.theta = np.linalg.inv(XtX) @ Xty

    def estimate_coef(self):
        alfa = [0, 0.0001, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 1, 2, 5, 10]
        beta = [0, 0.0001, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 1, 2, 5, 10]
        self.lr = []
        self.rr = []
        pick_alfa = -1
        best_alpha = 9999999999
        pick_beta = -1
        best_beta = 9999999999
        self.set_parameters(stop=0.5, rep_count=1000, condition=StopCondition.Both)
        self.set_parameters(mini_batch=True, batch_size=64, print_c=-1)
        for i in range(len(alfa)):
            self.set_parameters(loss_function=RidgeRegressionFunction(alfa[i]))
            self.linear_regression()
            res = self.MSE(self.validate)
            self.rr.append([alfa[i], res])
            if res < best_alpha:
                best_alpha = res
                pick_alfa = alfa[i]
        for j in range(len(beta)):
            self.set_parameters(loss_function=LassoRegressionFunction(beta[j]))
            self.linear_regression()
            res = self.MSE(self.validate)
            self.lr.append([beta[j], res])
            if res < best_beta:
                best_beta = res
                pick_beta = beta[j]
        return pick_alfa, pick_beta
    def corr(self):
        corr_matrix = np.corrcoef([self.train[i][:8] for i in range(len(self.train))], rowvar=False)

        plt.figure(figsize=(8, 6))
        im = plt.imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1)
        plt.colorbar(im)

        labels = ['X1', 'X2', 'X3', 'X4', 'X5', 'X6', 'X7', 'Y']
        plt.xticks(ticks=np.arange(8), labels=labels, rotation=45, ha='right')
        plt.yticks(ticks=np.arange(8), labels=labels)

        plt.title("Correlation Matrix")
        plt.tight_layout()
        plt.show()
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

    def normalize_features(self):
        mean = [np.mean(
            [self.train[i][j] for i in range(len(self.train))]
        ) for j in range(len(self.train[0]) - 1)]
        self.mean = mean
        mean.append(np.float64(0))
        std = [np.std(
            [self.train[i][j] for i in range(len(self.train))]
        ) for j in range(len(self.train[0]) - 1)]
        self.std = std
        std.append(np.float64(1))
        self.train = [[(self.train[i][j] - mean[j]) / std[j]
                 for j in range(len(self.train[0]))] for i in range(len(self.train))]
        self.validate = [[(self.validate[i][j] - mean[j]) / std[j]
                 for j in range(len(self.validate[0]))] for i in range(len(self.validate))]
        self.test = [[(self.test[i][j] - mean[j]) / std[j]
                 for j in range(len(self.test[0]))] for i in range(len(self.test))]

    def split_data(self, train_ratio, valid_ratio, normalise = True):
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
        if normalise:
            self.normalize_features()

    def __init__(self, file, normalise = True,
                 train_ratio = 0.6, valid_ratio = 0.2):
        self.mean = []
        self.std = []
        self.raw_data = []
        self.train = []
        self.validate = []
        self.test = []
        self.open_file(file)
        self.split_data(train_ratio, valid_ratio, normalise)
        self.theta = None
        self.eta = 0.005
        self.baseFunction = Linear()
        self.lossFunction = QuadraticLossFunction()
        self.stop = 0.1
        self.mini_batch = False
        self.batch_size = 32
        self.condition = StopCondition.Iterations
        self.rep_count = 1000
        self.print = 50
        self.fraction = 1
        self.split = False
        self.lc = []
        self.lr = []
        self.rr = []
        pass

    def set_parameters(self, eta = None, loss_function = None,
                       base_function = None, stop = None,
                       condition = None, rep_count = None, print_c = None, mini_batch = None, batch_size = None,
                       split = False, fraction = 1):
        if eta is not None:
            self.eta = eta
        if loss_function is not None:
            self.lossFunction = loss_function
        if base_function is not None:
            self.baseFunction = base_function
        if stop is not None:
            self.stop = stop
        if condition is not None:
            self.condition = condition
        if rep_count is not None:
            self.rep_count = rep_count
        if print_c is not None:
            self.print = print_c
        if mini_batch is not None:
            self.mini_batch = mini_batch
        if batch_size is not None:
            self.batch_size = batch_size
        if split is not None:
            self.split = split
        if fraction is not None:
            self.fraction = fraction

        pass

    def gradient_regular(self, planning_matrix, targets, theta):
        self.baseFunction.theta(theta)
        return self.lossFunction.calculate_gradient(self.baseFunction, planning_matrix, targets)

    def linear_regression(self):
        self.lc = []
        if self.baseFunction.parameters is None:
            theta_size = len(self.train[0])
        else:
            theta_size = len(self.baseFunction.parameters)
        theta = [[0 for _ in range(theta_size)]]
        if self.split:
            data = sample(self.train, int(self.fraction * len(self.train)))
        else:
            data = self.train
        k = 0
        planning_matrix = [
            [1, *[data[i][j] for j in range(len(self.train[i]) - 1)]]
            for i in range(len(data))
        ]
        targets = [
            data[i][-1]
            for i in range(len(data))
        ]
        while True:
            k = k + 1
            last_theta = theta[-1]
            if not self.mini_batch:
                gradient = self.gradient_regular(planning_matrix, targets, last_theta)
            else:
                mini_batch = sample(range(len(data)), self.batch_size)
                gradient = self.gradient_regular([planning_matrix[i] for i in mini_batch],
                                                 [targets[i] for i in mini_batch], last_theta)
            theta.append([
                last_theta[i] - self.eta * gradient[i]
                for i in range(theta_size)
            ])

            if self.print > 0 and (k % self.print) == 0:
                grad = np.linalg.norm(gradient)
                mse = self.lossFunction.calculate_loss(self.baseFunction, planning_matrix, targets)
                print(f"{k}: Gradient: {grad}, MSE: {
                    mse
                }")
                self.lc.append([k, grad, mse])

            if (self.condition != StopCondition.Iterations
                    and np.linalg.norm(gradient) < self.stop):
                break

            if (self.condition != StopCondition.Gradient
                    and k > self.rep_count):
                break
        self.theta = theta[-1]
        return theta[-1]

    def MSEs(self):
        return [self.MSE(self.train), self.MSE(self.test)]

    def results(self):
        print(f"Train Set MSE: {self.MSE(self.train)}")
        print(f"Test Set MSE: {self.MSE(self.test)}")

    def MSE(self, data):
        if self.theta is None:
            return
        self.baseFunction.theta(self.theta)
        newfunc = QuadraticLossFunction()
        return newfunc.calculate_loss(self.baseFunction, [
            [1, *[data[i][j] for j in range(len(data[i]) - 1)]]
            for i in range(len(data))
        ], [ data[i][-1] for i in range(len(data))])

    def prediction(self, x):
        if self.theta is None:
            return
        self.baseFunction.theta(self.theta)
        return self.baseFunction.evaluate(x)

    def plot_learning_curves(self):
        # Unpack the data from self.lc
        iterations = [item[0] for item in self.lc]
        gradients = [item[1] for item in self.lc]
        mses = [item[2] for item in self.lc]

        # Create the plots
        fig, axs = plt.subplots(1, 2, figsize=(12, 5))

        # Plot Iteration vs Gradient
        axs[0].plot(iterations, gradients, marker='o', color='tab:blue')
        axs[0].set_title('Iteration vs Gradient')
        axs[0].set_xlabel('Iteration')
        axs[0].set_ylabel('Gradient')
        axs[0].grid(True)

        # Plot Iteration vs MSE
        axs[1].plot(iterations, mses, marker='s', color='tab:orange')
        axs[1].set_title('Iteration vs MSE')
        axs[1].set_xlabel('Iteration')
        axs[1].set_ylabel('Mean Squared Error (MSE)')
        axs[1].grid(True)

        plt.tight_layout()
        plt.show()

    def plot_reg_coef(self):
        # Unpack the data
        coef = [item[0] for item in self.lr]
        lasso = [item[1] for item in self.lr]
        ridge = [item[1] for item in self.rr]

        fig, axs = plt.subplots(1, 2, figsize=(12, 5))

        # Lasso plot
        axs[0].plot(coef, lasso, marker='o', color='tab:blue')
        axs[0].set_title('MSE for different lasso coefficients')
        axs[0].set_xlabel('Lasso Coefficient')
        axs[0].set_ylabel('MSE on Validation Set')
        axs[0].set_xscale('log')
        axs[0].grid(True, which="both", linestyle='--', linewidth=0.5)

        # Ridge plot
        axs[1].plot(coef, ridge, marker='o', color='tab:blue')
        axs[1].set_title('MSE for different ridge coefficients')
        axs[1].set_xlabel('Ridge Coefficient')
        axs[1].set_ylabel('MSE on Validation Set')
        axs[1].set_xscale('log')
        axs[1].grid(True, which="both", linestyle='--', linewidth=0.5)

        plt.tight_layout()
        plt.show()

    def print_results(self):
        # Prepare train data
        train_data = [(self.prediction([1, *[self.train[k][j] for j in range(len(self.train[0]) - 1)]]),
                       self.train[k][-1])
                      for k in range(len(self.train))]

        # Prepare test data
        test_data = [(self.prediction([1, *[self.test[k][j] for j in range(len(self.test[0]) - 1)]]),
                      self.test[k][-1])
                     for k in range(len(self.test))]

        # Extract predictions and actuals
        train_preds = [x[0] for x in train_data]
        train_actuals = [x[1] for x in train_data]
        test_preds = [x[0] for x in test_data]
        test_actuals = [x[1] for x in test_data]

        # Plot
        plt.figure(figsize=(10, 6))
        plt.scatter(train_actuals, train_preds, alpha=0.6, label='Train Data', color='blue')
        plt.scatter(test_actuals, test_preds, alpha=0.6, label='Test Data', color='orange')
        plt.xlabel('Actual Values')
        plt.ylabel('Predicted Values')
        plt.title('Actual vs Predicted Values')

        # Line of perfect prediction
        max_val = max(max(train_actuals + test_actuals), max(train_preds + test_preds))
        min_val = min(min(train_actuals + test_actuals), min(train_preds + test_preds))
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect Prediction')

        plt.legend()
        plt.grid(True)
        plt.show()
    def plot_x2_x7(self):
        X = [self.raw_data[i][1] for i in range(len(self.raw_data))]
        Y = [self.raw_data[i][6] for i in range(len(self.raw_data))]
        plt.plot(X, Y, alpha=0.6, label='x_7(x_2)', color='orange')
        plt.xlabel('Cecha nr 2')
        plt.ylabel('Cecha nr 7')
        plt.title('Zależność cech 2 i 7')
        plt.plot()

        plt.legend()
        plt.grid(True)
        plt.show()





