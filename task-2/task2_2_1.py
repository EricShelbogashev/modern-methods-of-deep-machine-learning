import sys

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
        print(input_feature, self.weights, self.weights[:-1], self.weights[-1])
        summation = np.dot(input_feature, self.weights[:-1]) + self.weights[-1]
        return self._activate(summation)

    def fit(self, x, y, epochs=10):
        for _ in range(epochs):
            for input_feature, label in zip(x, y):
                prediction = self.predict(input_feature)
                error = label - prediction
                self.weights[:-1] += self.learning_rate * error * input_feature
                self.weights[-1] += self.learning_rate * error


def prepare_sample(data):
    raw_sample = zipped_sample[:, :2]
    labels = zipped_sample[:, 2]
    converted_sample = sample_converter.convert(raw_sample)
    return converted_sample, labels


if __name__ == "__main__":
    sample_type = 'circle'
    zipped_sample = SampleFactory().generate_samples(sample_type=sample_type, num_samples=1000, error=0.5)
    train_data, labels = prepare_sample(zipped_sample)
    perceptron = Perceptron(learning_rate=0.1, activation_function='sigmoid')
    perceptron.fit(train_data, labels, epochs=1000)

