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
    backend_all_but_pytorch,
)


if Version(jax.__version__) < Version("0.3.0"):
    raise RuntimeError("TensorLy only supports JAX v0.3.0 and above.")


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

    def copy(self, tensor):
        # See https://github.com/tensorly/tensorly/pull/397
        # and https://github.com/google/jax/issues/3473
        return self.tensor(tensor.copy(), **self.context(tensor))
        # return copy.copy(tensor)

    @staticmethod
    def lstsq(a, b, rcond=None):
        return np.linalg.lstsq(a, b, rcond=rcond, numpy_resid=True)

    @staticmethod
    def index_update(tensor, indices, values):
        return tensor.at[indices].set(values)


for name in (
    backend_types
    + backend_basic_math
    + backend_array
    + backend_all_but_pytorch
    + [
        "nan",
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
        "clip",
        "log2",
        "sort",
        "dot",
        "shape",
    ]
):
    JaxBackend.register_method(name, getattr(np, name))

for name in ["solve", "qr", "svd", "eigh"]:
    JaxBackend.register_method(name, getattr(np.linalg, name))

for name in ["gamma"]:
    JaxBackend.register_method(name, getattr(jax.random, name))

JaxBackend.register_method("logsumexp", jax.scipy.special.logsumexp)
