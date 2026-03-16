import numpy as np
from numpy import dot
from numpy.linalg import norm


def cosin_sim(A: np.ndarray, B: np.ndarray) -> float:
    """행렬곱 /"""
    return dot(A, B) / (norm(A) * norm(B))


def value_per(value: float, sub: float) -> float:
    if sub == 0:
        return 0
    if value == 0:
        return 0
    return value / sub
