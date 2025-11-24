
# utils/finite_difference.py
import numpy as np

def finite_difference_derivative(f, x, h=1e-6, order=1):
    """
    Compute the finite difference approximation of the derivative of function f at point x.
    f should accept a 1D numpy array and return a scalar.
    """
    x = np.array(x, dtype=float)
    grad = np.zeros_like(x)
    for i in range(len(x)):
        dx = np.zeros_like(x)
        dx[i] = h
        if order == 1:
            grad[i] = (f(x + dx) - f(x - dx)) / (2 * h)
        elif order == 2:
            grad[i] = (f(x + dx) - 2 * f(x) + f(x - dx)) / (h ** 2)
        else:
            raise NotImplementedError("Only order 1 and 2 are implemented.")
    return grad
