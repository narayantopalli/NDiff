import cupy as cp


def sin(arr):
    xp = cp.get_array_module(arr)
    eix = xp.exp(1j * arr)

    def derivative(scale):
        return scale * xp.array([eix.real])

    return eix.imag, derivative


def cos(arr):
    xp = cp.get_array_module(arr)
    eix = xp.exp(1j * arr)

    def derivative(scale):
        return -scale * xp.array([eix.imag])

    return eix.real, derivative


def exp(arr):
    xp = cp.get_array_module(arr)
    res = xp.exp(arr)

    def derivative(scale):
        return scale * xp.array([res])

    return res, derivative


def add(a, b):
    xp = cp.get_array_module(a)
    res = xp.add(a, b)

    def derivative(scale):
        return scale * xp.array([xp.ones_like(res), xp.ones_like(res)])

    return res, derivative


def subtract(a, b):
    xp = cp.get_array_module(a)
    res = xp.subtract(a, b)

    def derivative(scale):
        return scale * xp.array([xp.ones_like(res), -1 * xp.ones_like(res)])

    return res, derivative


def multiply(a, b):
    xp = cp.get_array_module(a)
    res = xp.multiply(a, b)

    def derivative(scale):
        return scale * xp.array([b * xp.ones_like(res), a * xp.ones_like(res)])

    return res, derivative


def divide(a, b):
    xp = cp.get_array_module(a)
    res = xp.divide(a, b)
    a = a * xp.ones_like(res)
    b = b * xp.ones_like(res)

    def derivative(scale):
        return scale * xp.array([1. / b, -a / (b ** 2)])

    return res, derivative


def power(a, b):
    xp = cp.get_array_module(a)
    res = xp.power(a, b)

    def derivative(scale):
        return scale * (b / xp.where(a != 0, a, xp.inf)) * xp.array([res])

    return res, derivative


def maximum(a, b):
    xp = cp.get_array_module(a)
    res = xp.maximum(a, b)

    def derivative(scale):
        mask = (a >= b)
        return scale * xp.ones_like(res) * xp.array([mask, ~mask])

    return res, derivative


def minimum(a, b):
    xp = cp.get_array_module(a)
    res = xp.minimum(a, b)

    def derivative(scale):
        mask = (a <= b)
        return scale * xp.ones_like(res) * xp.array([mask, ~mask])

    return res, derivative

__all__ = ['sin', 'cos', 'exp', 'add', 'subtract', 'multiply', 'divide', 'power', 'maximum', 'minimum']
