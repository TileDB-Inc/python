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

    >>> list(rechunk_arrays(chunked(range(1, 11), 3), 4))
    [array([1, 2, 3, 4]), array([5, 6, 7, 8]), array([ 9, 10])]

    The last yielded array will have fewer than *n* elements if the sum of lengths
    of *arrays* is not divisible by *n*.
    """
    pending: List[NDArray[Scalar]] = []
    remaining_len = n
    iter_arrays = iter(arrays)
    array = next(iter_arrays, None)
    while array is not None:
        assert remaining_len > 0
        array_len = len(array)
        if array_len < remaining_len:
            if array_len > 0:
                pending.append(array)
                remaining_len -= array_len
            array = next(iter_arrays, None)
        else:
            # concatenate all pending arrays and a slice of the current array
            pending.append(array[:remaining_len])
            yield np.concatenate(pending) if len(pending) > 1 else pending[0]
            # set the remaining slice of the current array as the next array
            # to be processed and reset the pending and remaining_len
            array = array[remaining_len:]
            pending.clear()
            remaining_len = n

    if pending:
        yield np.concatenate(pending) if len(pending) > 1 else pending[0]
