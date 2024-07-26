from __future__ import annotations
from typing import Sequence

from packaging.version import Version

try:
    import paddle
except ImportError as error:
    message = (
        "Impossible to import Paddle.\n"
        "To use TensorLy with the Paddle backend, "
        "you must first install Paddle!"
    )
    raise ImportError(message) from error

import numpy as np

from .core import (
    Backend,
    backend_types,
    backend_basic_math,
    backend_array,
)


if Version(paddle.__version__) < Version("2.6.0"):
    raise RuntimeError("TensorLy only supports paddle v2.6.0 and above.")


class PaddleBackend(Backend, backend_name="paddle"):
    # set default device to cpu
    place = paddle.device.set_device("cpu")

    @staticmethod
    def context(tensor: paddle.Tensor):
        return {
            "dtype": tensor.dtype,
            # "place": tensor.place,
            # "stop_gradient": tensor.stop_gradient,
        }

    @staticmethod
    def tensor(
        data: paddle.Tensor,
        dtype: paddle.dtype = None,
        place: str | None = None,
        stop_gradient: bool = None,
    ):
        """
        Tensor constructor for the Paddle backend.

        Parameters
        ----------
        data : array-like
            Data for the tensor.
        dtype : paddle.dtype, optional
            Data type of the tensor. If None, the dtype is inferred from the data.
        place : str, optional
            Device on which the tensor is allocated. If None, the place is inferred from the data in case of a paddle
            Tensor.
        stop_gradient : bool, optional.
            Whether to block the gradient propagation of Autograd. If None, stop_gradient is inferred from the
            data.

        """
        if paddle.is_tensor(data):
            # If source is a tensor, use clone-detach as suggested by Paddle
            tensor = data.clone().detach()
        else:
            # Else, use Paddle's tensor constructor
            if place is None:
                # set default device to cpu when place is not specified
                # and  gpu is avaiable
                current_device = paddle.device.get_device()
                if current_device.startswith("gpu"):
                    place = "cpu"

            tensor = paddle.to_tensor(data, place=place)

        # Set dtype/place/stop_gradient if specified
        if dtype is not None:
            tensor = tensor.astype(dtype)
        if place is not None:
            tensor = tensor.to(device=place)
        if stop_gradient is not None:
            tensor.stop_gradient = stop_gradient
        return tensor

    @staticmethod
    def index_update(tensor: paddle.Tensor, indices, values):
        """Updates the value of tensors in the specified indices
            Should be used as::

                index_update(tensor, tensorly.index[:, 3:5], values)

            Equivalent of::

                tensor[:, 3:5] = values

        Parameters
        ----------
        tensor : tensorly.tensor
            intput tensor which values to update
        indices : tensorly.index
            indices to update
        values : tensorly.tensor
            values to use to fill tensor[indices]

        Returns
        -------
        tensor
            updated tensor

        Example
        -------

        >>> import tensorly as tl
        >>> import numpy as np
        >>> tensor = tl.tensor([[1, 2, 3], [4, 5, 6]])
        >>> cpy = tensor.copy()
        >>> tensor[:, 1] = 0
        >>> tensor
        array([[1, 0, 3],
                [4, 0, 6]])
        >>> tl.index_update(tensor, tl.index[:, 1], 0)
        array([[1, 0, 3],
               [4, 0, 6]])

        See also
        --------
        index
        """
        if paddle.is_tensor(values):
            if paddle.is_tensor(indices):
                # indices is bool mask
                mask_coord = paddle.concat(
                    paddle.nonzero(indices, as_tuple=True),
                    axis=1,
                )
                t = paddle.scatter_nd_add(
                    tensor * (~indices).astype(tensor.dtype),
                    mask_coord,
                    values,
                )
                paddle.assign(t, tensor) # inplace update
            else:
                # NOTE: Assure source Tensor is contiguous
                tensor[indices] = (
                    values if values.is_contiguous()
                    else values.contiguous()
                )
        else:
            tensor[indices] = values
        return tensor

    @staticmethod
    def to_numpy(tensor: paddle.Tensor):
        if paddle.is_tensor(tensor):
            return tensor.numpy()
        elif isinstance(tensor, np.ndarray):
            return tensor
        else:
            return np.asarray(tensor)

    @staticmethod
    def shape(tensor: paddle.Tensor):
        return tuple(tensor.shape)

    @staticmethod
    def ndim(tensor: paddle.Tensor):
        return tensor.ndim

    @staticmethod
    def arange(start, stop=None, step=1.0, *args, **kwargs):
        if stop is None:
            return paddle.arange(
                start=0.0, end=float(start), step=float(step), *args, **kwargs
            )
        else:
            return paddle.arange(
                float(start), float(stop), float(step), *args, **kwargs
            )

    @staticmethod
    def clip(tensor: paddle.Tensor, a_min=None, a_max=None, inplace=False):
        if inplace:
            paddle.clip_(tensor, a_min, a_max)
            return tensor
        else:
            return paddle.clip(tensor, a_min, a_max)

    @staticmethod
    def all(tensor: paddle.Tensor):
        return paddle.sum(tensor != 0)

    def transpose(self, tensor: paddle.Tensor, axes: int | Sequence[int] | None = None):
        axes = axes or list(range(self.ndim(tensor)))[::-1]
        if not isinstance(axes, (tuple, list)):
            axes = list(axes)
        return tensor.transpose(axes)

    @staticmethod
    def copy(tensor: paddle.Tensor):
        return tensor.clone()

    @staticmethod
    def norm(tensor: paddle.Tensor, order=None, axis=None):
        # paddle does not accept `None` for any keyword arguments. additionally,
        # paddle doesn't seems to support keyword arguments in the first place
        kwds = {}
        if axis is not None:
            kwds["axis"] = axis
        if order and order != "inf":
            kwds["p"] = order

        if order == "inf":
            res = paddle.max(paddle.abs(tensor), **kwds)
            return res
        return paddle.linalg.norm(tensor, **kwds)

    @staticmethod
    def dot(a: paddle.Tensor, b: paddle.Tensor):
        if a.ndim > 2 and b.ndim > 2:
            return paddle.tensordot(a, b, axes=([-1], [-2]))
        if not a.ndim or not b.ndim:
            return a * b
        return paddle.matmul(a, b)

    @staticmethod
    def tensordot(a: paddle.Tensor, b: paddle.Tensor, axes=2, **kwargs):
        return paddle.tensordot(a, b, axes=axes, **kwargs)

    @staticmethod
    def mean(tensor: paddle.Tensor, axis=None):
        if axis is None:
            return paddle.mean(tensor)
        else:
            return paddle.mean(tensor, axis=axis)

    @staticmethod
    def sum(tensor: paddle.Tensor, axis=None, keepdims=False):
        if axis is None:
            axis = tuple(range(tensor.ndim))
        return paddle.sum(tensor, axis=axis, keepdim=keepdims)

    @staticmethod
    def max(tensor: paddle.Tensor, axis=None):
        if axis is None:
            return paddle.max(tensor)
        else:
            return paddle.max(tensor, axis=axis)[0]

    @staticmethod
    def flip(tensor: paddle.Tensor, axis=None):
        if isinstance(axis, int):
            axis = [axis]

        if axis is None:
            return paddle.flip(tensor, axis=[i for i in range(tensor.ndim)])
        else:
            return paddle.flip(tensor, axis=axis)

    @staticmethod
    def concatenate(tensors: paddle.Tensor, axis=0):
        return paddle.concat(tensors, axis=axis)

    @staticmethod
    def argmin(input: paddle.Tensor, axis=None):
        return paddle.argmin(input, axis=axis)

    @staticmethod
    def argsort(input: paddle.Tensor, axis=None):
        return paddle.argsort(input, axis=axis)

    @staticmethod
    def argmax(input: paddle.Tensor, axis=None):
        return paddle.argmax(input, axis=axis)

    @staticmethod
    def stack(arrays: paddle.Tensor, axis=0):
        return paddle.stack(arrays, axis=axis)

    @staticmethod
    def diag(tensor: paddle.Tensor, k=0):
        return paddle.diag(tensor, offset=k)

    @staticmethod
    def sort(tensor: paddle.Tensor, axis):
        if axis is None:
            tensor = tensor.flatten()
            axis = -1

        return paddle.sort(tensor, axis=axis)

    @staticmethod
    def update_index(tensor: paddle.Tensor, index, values):
        tensor.index_put_(index, values)

    @staticmethod
    def lstsq(a: paddle.Tensor, b: paddle.Tensor, rcond=None, driver="gelsd"):
        return paddle.linalg.lstsq(a, b, rcond=rcond, driver=driver)

    @staticmethod
    def eigh(tensor: paddle.Tensor):
        return paddle.linalg.eigh(tensor)

    @staticmethod
    def sign(tensor: paddle.Tensor):
        """paddle.sign does not support complex numbers."""
        return paddle.sgn(tensor)

    @staticmethod
    def logsumexp(tensor: paddle.Tensor, axis=0):
        return paddle.logsumexp(tensor, axis=axis)


# Register the other functions
for name in (
    backend_types
    + backend_basic_math
    + backend_array
    + [
        "nan",
        "is_tensor",
        "trace",
        "conj",
        "finfo",
        "log2",
        "digamma",
    ]
):
    if name in ["pi", "e", "inf", "nan"]:
        PaddleBackend.register_method(name, getattr(np, name))
    elif name in ["arctanh", "arccosh", "arcsinh", "arctan", "arccos", "arcsin"]:
        name = name.replace("arc", "a")
        PaddleBackend.register_method(name, getattr(paddle, name))
    else:
        PaddleBackend.register_method(name, getattr(paddle, name))


for name in ["kron", "moveaxis"]:
    PaddleBackend.register_method(name, getattr(paddle, name))

for name in ["solve", "qr", "svd", "eigh"]:
    PaddleBackend.register_method(name, getattr(paddle.linalg, name))
