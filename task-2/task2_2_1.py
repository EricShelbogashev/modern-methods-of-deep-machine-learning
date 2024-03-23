import numpy as np
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from SampleFactory import SampleFactory


class ElementaryPerceptron(nn.Module):
    def __init__(self, activation_fn=torch.sigmoid):
        super().__init__()
        # Линейный слой: принимает входные данные и производит линейное преобразование
        self.fc = nn.Linear(1, 1)
        # Функция активации: нелинейность, применяемая к выходу линейного слоя
        self.activation_fn = activation_fn

    def forward(self, x):
        # Прямой проход: вычисление выходных данных модели на основе входных данных
        return self.activation_fn(self.fc(x))


def train_elementary_perceptron(model, data, labels, epochs, learning_rate=0.1):
    criterion = torch.nn.BCELoss()  # Binary Cross Entropy Loss for binary classification
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        for i in range(data.size(0)):
            optimizer.zero_grad()
            output = model(data[i])  # Model prediction for a single sample
            loss = criterion(output, labels[i].unsqueeze(0))
            loss.backward()  # Backpropagation
            optimizer.step()  # Update model parameters


def generate_samples(num_samples=1000, pattern='xor', noise=0.1):
    """
    Генерация образцов на основе указанного шаблона.
    """
    return SampleFactory().generate_samples(pattern, num_samples, noise)


def input_translator(x, modification_type):
    """
    Преобразование входных признаков на основе указанного типа модификации.
    """
    if modification_type == "x1":
        return x[:, 0:1]
    elif modification_type == "x2":
        return x[:, 1:2]
    elif modification_type == "x1_squared":
        return x[:, 0:1] ** 2
    elif modification_type == "x2_squared":
        return x[:, 1:2] ** 2
    elif modification_type == "sin_x1":
        return torch.sin(x[:, 0:1])
    elif modification_type == "sin_x2":
        return torch.sin(x[:, 1:2])
    elif modification_type == "x1_x2":
        return x[:, 0:1] * x[:, 1:2]
    else:
        raise ValueError(f"Неизвестный тип модификации: {modification_type}")


def plot_classification_areas(model, samples, modification_type):
    data = torch.tensor(samples[:, :2], dtype=torch.float32)
    labels = torch.tensor(samples[:, 2], dtype=torch.float32).unsqueeze(1)

    # Создание сетки точек для визуализации
    x_min, x_max = data[:, 0].min() - 1, data[:, 0].max() + 1
    y_min, y_max = data[:, 1].min() - 1, data[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                         np.arange(y_min, y_max, 0.01))

    # Преобразование сетки в тензор и последующая модификация входных данных
    meshgrid_input = torch.Tensor(np.c_[xx.ravel(), yy.ravel()])
    translated_input = input_translator(meshgrid_input, modification_type)

    print(translated_input)
    # Предсказание классов для каждой точки сетки с использованием модифицированных входных данных
    Z = model(translated_input).detach().numpy()

    # Преобразование выходных данных для получения бинарных классовых меток
    Z = Z.reshape(xx.shape)

    # Визуализация результатов классификации
    plt.contourf(xx, yy, Z, alpha=0.7, cmap='coolwarm')
    plt.scatter(data[:, 0], data[:, 1], c=labels, s=50, edgecolor='k', cmap='coolwarm')
    plt.xlabel('Признак 1')
    plt.ylabel('Признак 2')
    plt.title('Области классификации')
    plt.show()


def get_perceptron(sample, modification_type, activation_fn, epochs, learning_rate):
    # Подготовка данных
    data_tensor = torch.tensor(sample[:, :2], dtype=torch.float32)
    labels_tensor = torch.tensor(sample[:, 2], dtype=torch.float32)

    # Модификация входных данных
    modified_data = input_translator(data_tensor, modification_type)

    # Создание и обучение модели
    model = ElementaryPerceptron(activation_fn=activation_fn)
    train_elementary_perceptron(model, modified_data, labels_tensor, epochs, learning_rate)
    return model


sample_factory = SampleFactory()
samples = sample_factory.generate_samples('spiral', 100, 0.5)  # Предполагается, что у вас есть реализация SampleFactory
modification_type = 'sin_x1'
epochs = 400
learning_rate = 0.1
activation_fn = torch.sigmoid

# perceptron_model = get_perceptron(samples, modification_type, activation_fn, epochs, learning_rate)
# plot_classification_areas(perceptron_model, samples, modification_type)
#
# weight = perceptron_model.fc.weight.data.numpy()  # Для получения веса в виде numpy массива
# bias = perceptron_model.fc.bias.data.numpy()  # Для получения смещения в виде numpy массива
# print(weight, bias)

def plot_classification_areas_ensemble(perceptrons, samples, modification_type):
    data = torch.tensor(samples[:, :2], dtype=torch.float32)
    labels = torch.tensor(samples[:, 2], dtype=torch.float32).unsqueeze(1)

    # Создание сетки точек для визуализации
    x_min, x_max = data[:, 0].min() - 1, data[:, 0].max() + 1
    y_min, y_max = data[:, 1].min() - 1, data[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                         np.arange(y_min, y_max, 0.01))

    # Преобразование сетки в тензор и последующая модификация входных данных
    meshgrid_input = torch.Tensor(np.c_[xx.ravel(), yy.ravel()])
    translated_input = input_translator(meshgrid_input, modification_type)

    print(translated_input)
    # Предсказание классов для каждой точки сетки с использованием модифицированных входных данных
    Z_tmp = []
    for model in perceptrons:
        Z_tmp.append(model(translated_input).detach().numpy())
    Z_tmp = np.array(Z_tmp)  # Преобразование списка массивов в один NumPy массив
    Z = np.mean(Z_tmp, axis=0)

    # Визуализация результатов классификации
    plt.contourf(xx, yy, Z, alpha=0.7, cmap='coolwarm')
    plt.scatter(data[:, 0], data[:, 1], c=labels, s=50, edgecolor='k', cmap='coolwarm')
    plt.xlabel('Признак 1')
    plt.ylabel('Признак 2')
    plt.title('Области классификации')
    plt.show()

perceptrons = []
for type in ['x1', 'x2']:
    perceptrons.append(get_perceptron(samples, type, activation_fn, epochs, learning_rate))
plot_classification_areas_ensemble(perceptrons, samples, modification_type)