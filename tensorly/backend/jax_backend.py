from packaging.version import Version

try:
    import jax

    jax.config.update("jax_enable_x64", True)
    import jax.numpy as np
    import jax.scipy.special
except ImportError as error:
    message = (
        "Impossible to import JAX.\n"
        "To use TensorLy with the JAX backend, "
        "you must first install JAX!"
    )
    raise ImportError(message) from error

import numpy

from .core import (
    Backend,
    backend_types,
    backend_basic_math,
    backend_array,
)


if Version(jax.__version__) < Version("0.3.5"):
    raise RuntimeError("TensorLy only supports JAX v0.3.5 and above.")


class JaxBackend(Backend, backend_name="jax"):
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
        return numpy.asarray(tensor)

    @staticmethod
    def lstsq(a, b, rcond=None):
        return np.linalg.lstsq(a, b, rcond=rcond, numpy_resid=True)

    @staticmethod
    def logsumexp(tensor, axis=0):
        return jax.scipy.special.logsumexp(tensor, axis=axis)

    @staticmethod
    def index_update(tensor, indices, values):
        return tensor.at[indices].set(values)


for name in (
    backend_types
    + backend_basic_math
    + backend_array
    + [
        "arange",
        "argmax",
        "argmin",
        "argsort",
        "clip",
        "concatenate",
        "conj",
        "copy",
        "diag",
        "dot",
        "flip",
        "kron",
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
    JaxBackend.register_method(name, getattr(np, name))

for name in ["solve", "qr", "svd", "eigh"]:
    JaxBackend.register_method(name, getattr(np.linalg, name))

for name in ["gamma"]:
    JaxBackend.register_method(name, getattr(jax.random, name))
