import numpy as np
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from torch import Tensor


class ElementaryPerceptron(nn.Module):
    def __init__(self, activation_fn, input_size=1):
        super(ElementaryPerceptron, self).__init__()
        self.fc = nn.Linear(input_size, 1)  # Линейный слой
        self.activation_fn = activation_fn  # Функция активации

    def forward(self, x):
        return self.activation_fn(self.fc(x))  # Прямой проход


def step_function(x):
    return torch.heaviside(x, torch.tensor([0.5]))


def sigmoid_function(x):
    return torch.sigmoid(x)


def train_elementary_perceptron(model: ElementaryPerceptron,
                                data: Tensor,
                                labels: Tensor,
                                count_of_epochs: int,
                                learning_rate: float = 0.1):
    for epoch in range(count_of_epochs):
        for i in range(len(data)):
            x = data[i]
            y = labels[i]
            output = model.forward(x)
            predicted = 1.0 if output > 0 else 0.0
            # Расчет ошибки
            error = y - predicted
            # Обновление весов с использованием градиентного спуска
            if error != 0:
                model.fc.weight.data[0] += learning_rate * error * x


def generate_samples_xor(num_samples: int = 1000, error: float = 0.1):
    width = 1.0
    height = 1.0
    x = np.random.uniform(-width / 2, width / 2, num_samples)
    y = np.random.uniform(-height / 2, height / 2, num_samples)

    x_error = np.random.uniform(-error, error, num_samples)
    y_error = np.random.uniform(-error, error, num_samples)

    labels = np.logical_xor(x > 0.0, y > 0.0).astype(float)
    x += x_error * (width / 2)
    y += y_error * (height / 2)

    return np.column_stack((x, y, labels))


def generate_samples_spiral(num_samples: int = 1000, error: float = 0.1):
    n = num_samples // 2

    def generate_samples_impl(n: int, noise: float, delta: float, label: int):
        r = np.arange(n) / n * 5
        t = 1.75 * np.arange(n) / n * 2 * np.pi + delta
        x = r * np.sin(t) + noise * np.random.uniform(-1, 1, n)
        y = r * np.cos(t) + noise * np.random.uniform(-1, 1, n)
        labels = np.full(n, label)
        return np.column_stack((x, y, labels))

    samples1 = generate_samples_impl(n, error, 0, 0)
    samples2 = generate_samples_impl(n, error, np.pi, 1)
    return np.vstack((samples1, samples2))


def generate_samples_circle(num_samples=1000, error=0.1):
    inner_radius: float = 0.25
    outer_radius: float = 0.5
    num_inner_heap = num_samples // 2
    num_outer_heap = num_samples - num_inner_heap

    inner_radius_for_center = np.random.uniform(low=inner_radius,
                                                high=inner_radius + error * inner_radius)
    inner_radius_for_bound = np.random.uniform(low=inner_radius - error * inner_radius,
                                               high=inner_radius)

    def _generate_samples(num_samples: int, inner_radius: float, outer_radius: float):
        angles = np.random.uniform(0, 2 * np.pi, num_samples)
        radii = np.random.uniform(inner_radius, outer_radius, num_samples)
        x = radii * np.cos(angles)
        y = radii * np.sin(angles)
        return np.column_stack((x, y))

    generated_center = _generate_samples(num_inner_heap, 0, inner_radius_for_center)
    generated_bound = _generate_samples(num_outer_heap, inner_radius_for_bound, outer_radius)

    generated_center_res = np.hstack((generated_center, np.zeros((generated_center.shape[0], 1))))
    generated_bound_res = np.hstack((generated_bound, np.ones((generated_bound.shape[0], 1))))

    return np.vstack((generated_center_res, generated_bound_res))


def generate_samples_gauss(num_samples=1000, error=0.1):
    variance_scale = np.interp(error, [0, 0.5], [0.5, 4])
    n = num_samples // 2

    positive_samples = np.random.normal(2, variance_scale, size=(n, 2))
    positive_labels = np.ones((n, 1))
    positive_data = np.hstack((positive_samples, positive_labels))

    negative_samples = np.random.normal(-2, variance_scale, size=(n, 2))
    negative_labels = np.zeros((n, 1))
    negative_data = np.hstack((negative_samples, negative_labels))

    return np.vstack((positive_data, negative_data))


def input_translator(x: torch.Tensor, modification_type: str) -> torch.Tensor:
    if modification_type == "x1":
        return x[:, 0:1]  # Возвращает x1
    elif modification_type == "x2":
        return x[:, 1:2]  # Возвращает x2
    elif modification_type == "x1_squared":
        return x[:, 0:1] ** 2  # Возвращает x1^2
    elif modification_type == "x2_squared":
        return x[:, 1:2] ** 2  # Возвращает x2^2
    elif modification_type == "sin_x1":
        return torch.sin(x[:, 0:1])  # Возвращает sin(x1)
    elif modification_type == "sin_x2":
        return torch.sin(x[:, 1:2])  # Возвращает sin(x2)
    else:
        raise ValueError("Unknown modification type")


samples = generate_samples_xor(1000)
data_example = np.column_stack((samples[:, 0], samples[:, 1]))
labels_example = samples[:, 2]

# Конвертация данных в тензоры PyTorch
data_tensor = torch.tensor(data_example, dtype=torch.float32)
# data_example = input_translator(data_tensor, "x2_squared")
labels_tensor = torch.tensor(labels_example, dtype=torch.float32)

# Создание экземпляра элементарного перцептрона с шаговой функцией активации
activation_function = sigmoid_function
perceptron_step = ElementaryPerceptron(activation_fn=activation_function)
