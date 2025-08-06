import numpy as np

import numba


@numba.njit()
def assign(x):
    mask = x.ravel() > 0.5
    x.ravel()[mask] = 0.0
    return x


a = np.random.uniform(size=(10, 3, 3))
z = assign(a)
z
