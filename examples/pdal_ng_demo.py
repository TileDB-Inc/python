from typing import Any, Optional, cast

import numpy as np
import pdal.ng

Point = pdal.ng.Point


class EvenFilter(pdal.ng.Filter):

    dim_idx: int

    def __init__(self, dim_idx: int, **kwargs: Any):
        super().__init__(dim_idx=dim_idx, **kwargs)

    def _filter_point(self, point: Point) -> Optional[Point]:
        return point if point[self.dim_idx] % 2 == 0 else None


class NegateFilter(pdal.ng.Filter):
    def _filter_point(self, point: Point) -> Optional[Point]:
        # If point was a regular ndarray or scalar we'd just return `-point`.
        # Since point is a structured array scalar (np.void), we have to operate on every
        # field explicitly. Additionally, there doesn't seem to be a way to create a new
        # structured array scalar directly; instead we create a 0d array and get its item
        dtype = point.dtype
        assert dtype.names is not None
        arr = np.array(tuple(-point[name] for name in dtype.names), dtype)
        return cast(Point, arr[()])


class StdoutWriter(pdal.ng.Writer):
    def _write_point(self, point: Point) -> None:
        print(point)


if __name__ == "__main__":
    dtype = [(dim, np.float32) for dim in ("X", "Y", "Z")]
    points = np.array([(i, 2 * i, 3 * i) for i in range(20)], dtype=dtype)

    print("========================== Input points ==========================")
    for i, point in enumerate(points):
        print(i, point)
    print()

    pipeline = NegateFilter() | EvenFilter(dim_idx=2)

    print("========================== Output points ==========================")
    for i, point in enumerate(pipeline.process_points(points)):
        print(i, point)
    print()

    print("========================== Output chunks ==========================")
    for i, chunk in enumerate(pipeline.process_chunks(3, [points])):
        print(f"chunk {i}: {chunk.__class__.__name__}, {len(chunk)} points")
        for point in chunk:
            print("\t", point)
    print()

    print("====================== Write output points to stdout ======================")
    for point in (pipeline | StdoutWriter()).process_points(points):
        pass
    print()

    print("====================== Write output chunks to stdout ======================")
    for chunk in (pipeline | StdoutWriter()).process_chunks(3, [points]):
        pass
    print()
