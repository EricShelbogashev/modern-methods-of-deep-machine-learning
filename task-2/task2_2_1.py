import sys

from matplotlib import pyplot as plt

sys.path.append('sample')
from sample.plot_generator import PlotGenerator
from sample.sample_factory import SampleFactory
import sample.sample_converter as sample_converter

import numpy as np


class Perceptron:
    def __init__(self, learning_rate=0.1, dimension=1, activation_function='sigmoid', learn_type='gradient', to='x1'):
        if activation_function not in ['step', 'sigmoid']:
            raise ValueError("активационная функция должна быть 'step' или 'sigmoid'")
        if learn_type not in ['gradient', 'other']:
            raise ValueError("метод обучения должен быть 'gradient' или 'theorem'")
        self.learning_rate = learning_rate
        self.activation_function = activation_function
        self.weights = np.random.rand(dimension + 1) * 0.01
        self.to = to

    @staticmethod
    def _sigmoid(x):
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def _step_function(x):
        return 1 if x >= 0 else 0

    def _activate(self, x):
        if self.activation_function == 'sigmoid':
            return self._sigmoid(x)
        elif self.activation_function == 'step':
            return self._step_function(x)
        else:
            raise ValueError("Unsupported activation function. Choose 'sigmoid' or 'step'.")

    def predict(self, x, y):
        point = sample_converter.convert_point(x, y, self.to)
        summation = np.dot(point, self.weights[:-1]) + self.weights[-1]
        return self._activate(summation)

    def fit(self, train, epochs=10):
        raw_sample = train[:, :2]
        labels = train[:, 2]
        for _ in range(epochs):
            for dot, label in zip(raw_sample, labels):
                input_feature = sample_converter.convert_point(dot[0], dot[1], self.to)
                prediction = self.predict(dot[0], dot[1])
                error = label - prediction
                print(self.weights, input_feature, prediction, error)
                self.weights[:-1] = self.weights[:-1] + self.learning_rate * error * input_feature
                self.weights[-1] = self.learning_rate * error


def plot_perceptron_predictions_optimized(X, Y, perceptron):
    vectorized_predict = np.vectorize(perceptron.predict)
    Z = vectorized_predict(X, Y)
    plt.contourf(X, Y, Z, cmap='Spectral')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Perceptron Predictions Optimized')
    plt.show()


class Ensemble:
    def __init__(self, models):
        self.models = models

    def predict(self, x, y):
        predictions = [model.predict(x, y) for model in self.models]
        return np.average(predictions)


if __name__ == "__main__":
    sample_type = 'gauss'
    zipped_sample = SampleFactory().generate_samples(sample_type=sample_type, num_samples=800, error=0.0)
    PlotGenerator.plot_samples(zipped_sample, "")
    perceptron_1 = Perceptron(learning_rate=0.01, activation_function='step', to='sin_x2')
    perceptron_2 = Perceptron(learning_rate=0.01, activation_function='step', to='sin_x1')
    perceptron_1.fit(zipped_sample, epochs=50)
    perceptron_2.fit(zipped_sample, epochs=50)
    perceptron = Ensemble([perceptron_1, perceptron_2])

    x_values = np.linspace(-4, 4, 100)
    y_values = np.linspace(-4, 4, 100)
    X, Y = np.meshgrid(x_values, y_values)
    plot_perceptron_predictions_optimized(X, Y, perceptron)
    plot_perceptron_predictions_optimized(X, Y, perceptron_1)
    plot_perceptron_predictions_optimized(X, Y, perceptron_2)
