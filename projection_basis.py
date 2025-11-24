
# utils/projection_basis.py
import numpy as np

def random_direction(n):
    v = np.random.randn(n)
    return v / np.linalg.norm(v)

def generate_directions(n, count=256):
    return [random_direction(n) for _ in range(count)]
