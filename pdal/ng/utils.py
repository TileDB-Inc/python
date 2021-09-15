import itertools as it
from typing import Iterable, Iterator, List, TypeVar

import numpy as np
from numpy.typing import NDArray

T = TypeVar("T")
Scalar = TypeVar("Scalar", bound=np.generic)


def chunked(iterable: Iterable[T], n: int) -> Iterator[List[T]]:
    """Break *iterable* into lists of length *n*

    >>> list(chunked([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 3))
    [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10]]

    The last yielded list will have fewer than *n* elements if the length of *iterable*
    is not divisible by *n*.
    """
    iterator = iter(iterable)
    return iter(lambda: list(it.islice(iterator, n)), [])


def rechunk_arrays(
    arrays: Iterable[NDArray[Scalar]], n: int
) -> Iterator[NDArray[Scalar]]:
    """Rechunk an iterable of *arrays* into arrays of length *n*

    >>> list(rechunk_arrays(chunk_array(np.arange(1, 11), 3), 4))
    [array([1, 2, 3, 4]), array([5, 6, 7, 8]), array([ 9, 10])]

    The last yielded array will have fewer than *n* elements if the sum of lengths
    of *arrays* is not divisible by *n*.
    """
    # TODO: optimize this
    buffer: List[Scalar] = []
    for array in arrays:
        buffer.extend(array)
        if len(buffer) >= n:
            yield np.array(buffer[:n])
            del buffer[:n]
    while buffer:
        yield np.array(buffer[:n])
        del buffer[:n]
