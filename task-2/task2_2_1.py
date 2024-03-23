import sys

from matplotlib import pyplot as plt

sys.path.append('sample')
from sample.plot_generator import PlotGenerator
from sample.sample_factory import SampleFactory
import sample.sample_converter as sample_converter

import numpy as np


class Perceptron:
    def __init__(self, learning_rate=0.1, dimension=1, activation_function='sigmoid', learn_type='gradient'):
        if activation_function not in ['step', 'sigmoid']:
            raise ValueError("активационная функция должна быть 'step' или 'sigmoid'")
        if learn_type not in ['gradient', 'other']:
            raise ValueError("метод обучения должен быть 'gradient' или 'theorem'")
        self.learning_rate = learning_rate
        self.activation_function = activation_function
        self.weights = np.random.rand(dimension + 1) * 0.01

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

    def predict(self, input_feature):
        summation = np.dot(input_feature, self.weights[:-1]) + self.weights[-1]
        return self._activate(summation)

    def fit(self, x, y, epochs=10):
        for _ in range(epochs):
            for input_feature, label in zip(x, y):
                prediction = self.predict(input_feature)
                error = label - prediction
                print(input_feature, label, error)
                self.weights[:-1] += self.learning_rate * error * input_feature
                self.weights[-1] += self.learning_rate * error


def prepare_sample(data, to):
    raw_sample = data[:, :2]
    labels = data[:, 2]
    converted_sample = sample_converter.convert(raw_sample, to)
    return converted_sample, labels


def plot_perceptron_predictions_optimized(x_values, y_values, perceptron, sample_converter, to):
    X, Y = np.meshgrid(x_values, y_values)
    samples = np.array([sample_converter.convert_point(x, y, to) for x, y in zip(np.ravel(X), np.ravel(Y))])
    predictions = np.array([perceptron.predict(sample) for sample in samples])

    # Reshape predictions to match the shape of X and Y for plotting
    Z = predictions.reshape(X.shape)

    # Plotting
    plt.contourf(X, Y, Z, levels=[-0.5, 0.5, 1.5], colors=['red', 'blue'], alpha=0.5)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Perceptron Predictions Optimized')
    plt.show()


class Ensemble:
    def __init__(self, models: [Perceptron]):
        self.models = models

    def fit(self, x, y, epochs=10):
        x_parts = np.array_split(x, len(self.models))
        y_parts = np.array_split(y, len(self.models))
        for i in range(len(self.models)):
            x_train = np.concatenate([part for j, part in enumerate(x_parts) if j != i])
            y_train = np.concatenate([part for j, part in enumerate(y_parts) if j != i])
            self.models[i].fit(x_train, y_train, epochs)

    def predict(self, x):
        predictions = [model.predict(x) for model in self.models]
        return np.average(predictions)


if __name__ == "__main__":
    sample_type = 'circle'
    to = 'sin_x1'
    zipped_sample = SampleFactory().generate_samples(sample_type=sample_type, num_samples=800, error=0.5)
    train_data_1, labels = prepare_sample(zipped_sample, 'x1')
    train_data_2, labels = prepare_sample(zipped_sample, 'x2')
    perceptron_1 = Perceptron(learning_rate=0.1, activation_function='step')
    perceptron_2 = Perceptron(learning_rate=0.1, activation_function='step')
    perceptron_1.fit(train_data_1, labels, epochs=80)
    perceptron_2.fit(train_data_2, labels, epochs=80)
    perceptron = Ensemble([perceptron_1, perceptron_2])

    x_values = np.linspace(-1, 1, 100)
    y_values = np.linspace(-1, 1, 100)
    plot_perceptron_predictions_optimized(x_values, y_values, perceptron, sample_converter, to)
