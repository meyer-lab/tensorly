import numpy as np
from .core import (
    Backend,
    backend_types,
    backend_basic_math,
    backend_array,
    backend_all_but_pytorch,
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


for name in (
    backend_types
    + backend_basic_math
    + backend_array
    + backend_all_but_pytorch
    + [
        "nan",
        "copy",
        "arange",
        "flip",
        "kron",
        "concatenate",
        "max",
        "mean",
        "sum",
        "sign",
        "conj",
        "diag",
        "log",
        "log2",
        "sort",
        "dot",
        "shape",
    ]
):
    NumpyBackend.register_method(name, getattr(np, name))

for name in ["solve", "qr", "svd", "eigh", "lstsq"]:
    NumpyBackend.register_method(name, getattr(np.linalg, name))

for name in ["digamma"]:
    NumpyBackend.register_method(name, getattr(scipy.special, name))

NumpyBackend.register_method("logsumexp", scipy.special.logsumexp)
