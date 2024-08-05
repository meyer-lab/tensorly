import numpy as np
from .core import (
    Backend,
    backend_types,
    backend_basic_math,
    backend_array,
)
import scipy.special


class NumpyBackend(Backend, backend_name="numpy"):
    @staticmethod
    def context(tensor):
        return {"dtype": tensor.dtype}

    @staticmethod
    def tensor(data, dtype=None, **kwargs):
        return np.array(data, dtype=dtype)

    @staticmethod
    def is_tensor(tensor):
        return isinstance(tensor, np.ndarray)

    @staticmethod
    def to_numpy(tensor):
        return np.copy(tensor)

    @staticmethod
    def clip(tensor, a_min=None, a_max=None):
        return np.clip(tensor, a_min, a_max)

    @staticmethod
    def logsumexp(tensor, axis=0):
        return scipy.special.logsumexp(tensor, axis=axis)


for name in (
    backend_types
    + backend_basic_math
    + backend_array
    + [
        "arange",
        "argmax",
        "argmin",
        "argsort",
        "concatenate",
        "conj",
        "copy",
        "diag",
        "dot",
        "flip",
        "kron",
        "log",
        "log2",
        "max",
        "mean",
        "moveaxis",
        "nan",
        "shape",
        "sign",
        "sort",
        "stack",
        "sum",
        "tensordot",
        "trace",
        "transpose",
    ]
):
    NumpyBackend.register_method(name, getattr(np, name))

for name in ["solve", "qr", "svd", "eigh", "lstsq"]:
    NumpyBackend.register_method(name, getattr(np.linalg, name))

for name in ["digamma"]:
    NumpyBackend.register_method(name, getattr(scipy.special, name))
