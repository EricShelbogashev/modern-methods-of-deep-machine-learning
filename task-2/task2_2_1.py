import numpy as np
import torch
import torch.nn as nn


# Определение элементарного перцептрона с использованием PyTorch
class ElementaryPerceptron(nn.Module):
    def __init__(self, input_size, activation_fn):
        super(ElementaryPerceptron, self).__init__()
        self.fc = nn.Linear(input_size, 1)  # Линейный слой
        self.activation_fn = activation_fn  # Функция активации

    def forward(self, x):
        return self.activation_fn(self.fc(x))  # Прямой проход


# Функция активации "шаговая функция"
def step_function(x):
    return torch.heaviside(x, torch.tensor([0.5]))


# Функция активации "сигмоид"
def sigmoid_function(x):
    return torch.sigmoid(x)


# Обучение элементарного перцептрона
def train_elementary_perceptron(model, data, labels, epochs):
    for epoch in range(epochs):
        for i in range(len(data)):
            x = data[i]
            y = labels[i]
            output = model(x)
            predicted = 1.0 if output > 0 else 0.0
            # Обновление весов, если прогноз неверен
            if y == 1 and predicted == 0:
                for index, value in enumerate(x):
                    if value != 0:
                        model.fc.weight.data[0][index] += 1
            elif y == 0 and predicted == 1:
                for index, value in enumerate(x):
                    if value != 0:
                        model.fc.weight.data[0][index] -= 1


# Пример данных (замените этими вашими реальными данными)
np.random.seed(0)  # Для воспроизводимости
data = np.random.rand(100, 10)  # Случайные данные
labels = np.random.choice([0, 1], size=(100,))  # Случайные метки

# Конвертация данных в тензоры PyTorch
data_tensor = torch.tensor(data, dtype=torch.float32)
labels_tensor = torch.tensor(labels, dtype=torch.float32)

# Создание экземпляра элементарного перцептрона с шаговой функцией активации
perceptron_step = ElementaryPerceptron(input_size=10, activation_fn=step_function)

# Обучение элементарного перцептрона
train_elementary_perceptron(perceptron_step, data_tensor, labels_tensor, epochs=10)

# Вывод весов после обучения
print("Weights:", perceptron_step.fc.weight.data)
