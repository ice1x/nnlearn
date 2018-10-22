# -*- coding: utf-8 -*-
import numpy as np


def nonlin(x, derivative=False):
    if derivative:
        return x * (1 - x)

    return 1 / (1 + np.exp(-x))


# входные данные
X = np.array([[0, 0, 1],
              [0, 1, 1],
              [1, 0, 1],
              [1, 1, 1]])

# выходные данные
Y = np.array([[0],
              [1],
              [1],
              [0]])

# случайно инициализируем веса, в среднем - 0
np.random.seed(1)
syn0 = 2 * np.random.random((3, 4)) - 1  # первый слой весов, Synapse 0, объединяет l0 с l1.
syn1 = 2 * np.random.random((4, 1)) - 1  # второй слой весов, Synapse 1, объединяет l1 с l2.

for j in range(60000):

    # проходим вперёд по слоям 0, 1 и 2
    l0 = X
    l1 = nonlin(np.dot(l0, syn0))
    l2 = nonlin(np.dot(l1, syn1))

    # как сильно мы ошиблись относительно нужной величины?
    l2_error = Y - l2

    if (j % 10000) == 0:
        print("Error:" + str(np.mean(np.abs(l2_error))))

    # в какую сторону нужно двигаться?
    # если мы были уверены в предсказании, то сильно менять его не надо
    l2_delta = l2_error * nonlin(l2, derivative=True)

    # как сильно значения l1 влияют на ошибки в l2?
    l1_error = l2_delta.dot(syn1.T)

    # в каком направлении нужно двигаться, чтобы прийти к l1?
    # если мы были уверены в предсказании, то сильно менять его не надо
    l1_delta = l1_error * nonlin(l1, derivative=True)

    syn1 += l1.T.dot(l2_delta)
    syn0 += l0.T.dot(l1_delta)
