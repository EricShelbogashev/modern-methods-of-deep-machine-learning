import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from task2_2_1 import ElementaryPerceptron, sigmoid_function


class PerceptronEnsemble(nn.Module):
    def __init__(self, input_size, num_perceptrons, activation_fn):
        super(PerceptronEnsemble, self).__init__()
        # Создаем список перцептронов, используя ModuleList
        self.perceptrons = nn.ModuleList(
            [ElementaryPerceptron(input_size, activation_fn) for _ in range(num_perceptrons)])

    def forward(self, x):
        # Применяем все перцептроны к входным данным и конкатенируем результаты
        outputs = torch.cat([p(x).unsqueeze(1) for p in self.perceptrons], dim=1)
        # Возвращаем среднее значение выходов всех перцептронов
        return torch.mean(outputs, dim=1)


# Пример использования
ensemble = PerceptronEnsemble(input_size=784, num_perceptrons=5, activation_fn=sigmoid_function)
criterion = nn.BCEWithLogitsLoss()  # Функция потерь для бинарной классификации
optimizer = optim.SGD(ensemble.parameters(), lr=0.01)  # Оптимизатор


def train_ensemble(model, data, labels, epochs):
    for epoch in range(epochs):
        for i in range(len(data)):
            optimizer.zero_grad()  # Обнуляем градиенты
            x = data[i]
            y = labels[i].unsqueeze(0)  # Добавляем размерность для метки
            output = model(x)
            loss = criterion(output, y)  # Вычисляем функцию потерь
            loss.backward()  # Вычисляем градиенты
            optimizer.step()  # Обновляем веса


np.random.seed(0)  # Для воспроизводимости результатов
data = np.random.rand(100, 10)  # Случайные данные
labels = np.random.choice([0, 1], size=(100,))  # Случайные метки

# Преобразуем данные в тензоры PyTorch
data_tensor = torch.tensor(data, dtype=torch.float32)
labels_tensor = torch.tensor(labels, dtype=torch.float32)

# Создаем ансамбль перцептронов с активационной функцией сигмоид
ensemble = PerceptronEnsemble(input_size=10, num_perceptrons=5, activation_fn=sigmoid_function)

# Обучаем ансамбль перцептронов
train_ensemble(ensemble, data_tensor, labels_tensor, epochs=10)
