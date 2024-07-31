try:
    import cupy as cp
    import cupyx.scipy.special

except ImportError as error:
    message = (
        "Impossible to import cupy.\n"
        "To use TensorLy with the cupy backend, "
        "you must first install cupy!"
    )
    raise ImportError(message) from error

import warnings

from .core import (
    Backend,
    backend_types,
    backend_basic_math,
    backend_array,
    backend_all_but_pytorch,
)


class CupyBackend(Backend, backend_name="cupy"):
    @staticmethod
    def context(tensor):
        return {"dtype": tensor.dtype}

    @staticmethod
    def tensor(data, dtype=cp.float32, **kwargs):
        return cp.array(data, dtype=dtype)

    @staticmethod
    def is_tensor(tensor):
        return isinstance(tensor, cp.ndarray)

    @staticmethod
    def to_numpy(tensor):
        if isinstance(tensor, cp.ndarray):
            return cp.asnumpy(tensor)
        return tensor


for name in (
    backend_types
    + backend_basic_math
    + backend_array
    + backend_all_but_pytorch
    + [
        "nan",
        "copy",
        "arange",
        "dot",
        "kron",
        "concatenate",
        "max",
        "flip",
        "mean",
        "sum",
        "sign",
        "conj",
        "diag",
        "log2",
        "sort",
        "shape",
    ]
):
    CupyBackend.register_method(name, getattr(cp, name))

for name in ["svd", "qr", "eigh", "solve", "lstsq"]:
    CupyBackend.register_method(name, getattr(cp.linalg, name))

CupyBackend.register_method("gamma", cp.random.gamma)
CupyBackend.register_method("logsumexp", cupyx.scipy.special.logsumexp)
