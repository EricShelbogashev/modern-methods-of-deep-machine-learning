import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import solve

N = 10
epsilon_0 = 0.5  # абсолютное

x_values = np.random.uniform(-1, 1, N)

errors_uniform = np.random.uniform(-epsilon_0, epsilon_0, N)
errors_distribution = np.random.normal(0, epsilon_0, N)

a, b, c, d = np.random.uniform(-3, 3, 4)
y_values_a_uniform = a * x_values ** 3 + b * x_values ** 2 + c * x_values + d + errors_uniform
y_values_b_uniform = x_values * np.sin(2 * np.pi * x_values) + errors_uniform
y_values_a_distribution = a * x_values ** 3 + b * x_values ** 2 + c * x_values + d + errors_distribution
y_values_b_distribution = x_values * np.sin(2 * np.pi * x_values) + errors_distribution

x_dense = np.arange(-1, 1, 0.001)
y_a_original = a * x_dense ** 3 + b * x_dense ** 2 + c * x_dense + d
y_b_original = x_dense * np.sin(2 * np.pi * x_dense)


def regression(M, x_dense, y_values, y_dense, plot_string):
    A = np.zeros(shape=(M, M))
    b = np.zeros(shape=M)

    def element_a(i, j):
        result = 0
        for k in range(N):
            result += x_values[k] ** (j + i)
        return result

    def element_b(i):
        result = 0
        for k in range(N):
            result += y_values[k] * (x_values[k] ** i)
        return result

    for i in range(M):
        for j in range(M):
            A[i][j] = element_a(i, j)

    for i in range(M):
        b[i] = element_b(i)
    w = solve(A, b)

    def compute_polynomial():
        result = 0
        for index in range(M):
            result += w[index] * x_dense ** index
        return result

    plt.figure(figsize=(10, 6))
    plt.plot(x_dense, y_dense, label=f'Оригинальная функция {plot_string}', color='black')
    plt.scatter(x_values, y_values, label='Выборка с равномерной ошибкой', color='red', marker='x')
    plt.plot(x_dense, compute_polynomial(), color='blue', label='Полиномиальная регрессия')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.show()


def plot_string_a(a: float, b: float, c: float, d: float):
    return f"({a.__round__(2)})x^3 + {b.__round__(2)}(x^2) + {c.__round__(2)}(x) + {d.__round__(2)}"


M = 7
regression(M, x_dense, y_values_a_uniform, y_a_original, plot_string_a(a, b, c, d))
regression(M, x_dense, y_values_b_uniform, y_b_original, "b, равномерно")
regression(M, x_dense, y_values_a_distribution, y_a_original, plot_string_a(a, b, c, d))
regression(M, x_dense, y_values_b_distribution, y_b_original, "b, нормально")

