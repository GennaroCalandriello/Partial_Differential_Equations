import logging
import numpy as np
from numba import njit, float64, int32

"""Explicit"""
L = 2 * np.pi
D = 0.5
cos = False


@njit()
def parameters(N, dt):
    x = np.linspace(-L, L, N)
    y = x
    p = np.zeros((N, N))
    dx = x[1] - x[0]
    dy = dx
    betax = 0.5 * dt / dx
    betay = 0.5 * dt / dy
    alphax = dt * D / dx ** 2
    alphay = dt * D / dy ** 2
    p = initial_state_(N, p, x, y, L)
    return (x, y, dx, p, betax, betay, alphax, alphay)


@njit()
def initial_state_(N, p, x, y, L):
    """Construct the initial state"""
    a = 1
    shift = 0.3
    for i in range(N):
        for j in range(N):
            p[i, j] = np.exp(
                -((x[i] - shift * L) ** 2 + (y[j] - shift * L) ** 2) / (2 * a)
            )
    return p


@njit()
def g(x, y):
    """Drift function on x derivative"""
    return y


@njit()
def f(x, y, t):
    """Drift function on y derivative"""
    w = 0.10
    a0 = 0.1
    a = a0 * np.cos(w * t)
    # a=0.1
    if cos:
        return x - a * y * (1 - x ** 2)
    else:
        return x - a0 * y * (1 - x ** 2)


@njit()
def boundary_conditions(kind, N, matrix):
    """Impose boundary conditions; kind should be: absorbing, periodic or reflecting"""
    if kind == "absorbing":
        matrix[0, :] = 0
        matrix[:, 0] = 0
        matrix[N - 1, :] = 0
        matrix[:, N - 1] = 0
    if kind == "periodic":
        matrix[0, :] = matrix[N - 2, :]
        matrix[N - 1, :] = matrix[1, :]
        matrix[:, 0] = matrix[:, N - 2]
        matrix[:, N - 1] = matrix[:, 1]
    if kind == "reflecting":
        matrix[0, :] = matrix[1, :]
        matrix[:, 0] = matrix[:, 1]
        matrix[N - 2, :] = matrix[N - 1, :]
        matrix[:, N - 2] = matrix[:, N - 1]
    return matrix
