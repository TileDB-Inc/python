import itertools as it
from typing import Iterable, Iterator, List, TypeVar

T = TypeVar("T")


def chunked(iterable: Iterable[T], n: int) -> Iterator[List[T]]:
    """Break *iterable* into lists of length *n*

    >>> list(chunked([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 3))
    [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10]]

    The last yielded list will have fewer than *n* elements if the length of *iterable*
    is not divisible by *n*.
    """
    iterator = iter(iterable)
    return iter(lambda: list(it.islice(iterator, n)), [])


def rechunked(iterable: Iterable[Iterable[T]], n: int) -> Iterator[List[T]]:
    """Rechunk an *iterable* of chunks into lists of length *n*

    >>> list(rechunked([[1, 2, 3], [4, 5], [6, 7, 8, 9], [10]], 4))
    [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10]]

    The last yielded list will have fewer than *n* elements if the sum of lengths
    of the *iterable* chunks is not divisible by *n*.
    """
    return chunked(it.chain.from_iterable(iterable), n)


if __name__ == "__main__":
    s = range(11)
    chunks = list(chunked(s, 3))
    assert chunks == [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10]]
    assert list(chunked(iter(s), 3)) == chunks

    new_chunks = list(rechunked(chunks, 4))
    assert new_chunks == [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10]]
    assert list(rechunked(iter(chunks), 4)) == new_chunks

    new_chunks = list(rechunked(chunks, 2))
    assert new_chunks == [[0, 1], [2, 3], [4, 5], [6, 7], [8, 9], [10]]
    assert list(rechunked(iter(chunks), 2)) == new_chunks
