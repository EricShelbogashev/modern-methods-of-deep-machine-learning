import numpy
import numpy as np


# sin_x1, sin_x2,, x1, x2, x1_squared, x2_squared, x1_x2_product
def convert(points: numpy.array, to: str = 'sin_x1') -> np.array:
    x_values = points[:, 0]
    y_values = points[:, 1]
    if to == 'sin_x1':
        return np.sin(x_values)
    elif to == 'sin_x2':
        return np.sin(y_values)
    elif to == 'x1':
        return x_values
    elif to == 'x2':
        return y_values
    elif to == 'x1_squared':
        return x_values ** 2
    elif to == 'x2_squared':
        return y_values ** 2
    elif to == 'x1_x2_product':
        return x_values * y_values


def convert_point(x: int, y: int, to: str = 'sin_x1') -> np.array:
    if to == 'sin_x1':
        return np.sin(x)
    elif to == 'sin_x2':
        return np.sin(y)
    elif to == 'x1':
        return x
    elif to == 'x2':
        return y
    elif to == 'x1_squared':
        return x ** 2
    elif to == 'x2_squared':
        return y ** 2
    elif to == 'x1_x2_product':
        return x * y
