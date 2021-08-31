import numpy as np


def atleast_kd(array, k, append_dims=True):
    array = np.asarray(array)

    if append_dims:
        new_shape = array.shape + (1,) * (k-array.ndim)
    else:
        new_shape = (1,) * (k-array.ndim) + array.shape

    return array.reshape(new_shape)
