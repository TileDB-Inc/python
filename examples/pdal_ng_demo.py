from typing import Any, Optional

import numpy as np
import pdal.ng


class EvenFilter(pdal.ng.Filter):

    dim_idx: int

    def __init__(self, dim_idx: int, **kwargs: Any):
        super().__init__(dim_idx=dim_idx, **kwargs)

    def _filter_point(self, point: pdal.ng.Point) -> Optional[pdal.ng.Point]:
        return point if point[self.dim_idx] % 2 == 0 else None


class NegateFilter(pdal.ng.Filter):
    def _filter_point(self, point: pdal.ng.Point) -> Optional[pdal.ng.Point]:
        # If each point was a regular ndarray or scalar we'd just return `-point`.
        # Since point is a structured array scalar (np.void), we have to operate on every
        # field explicitly. Additionally, there doesn't seem to be a way to create a new
        # structured array scalar directly; instead we create a 0d array and get its item
        return np.array((-point["X"], -point["Y"], -point["Z"]), point.dtype)[()]


class StdoutWriter(pdal.ng.Writer):
    def _write_chunk(self, chunk: pdal.ng.Chunk) -> None:
        for point in chunk:
            print(point)


def main() -> None:
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
    for i, chunk in enumerate(pipeline.process_chunks(points, chunk_size=3)):
        print(f"chunk {i}: {chunk.__class__.__name__}, {len(chunk)} points")
        for point in chunk:
            print("\t", point)
    print()

    print("====================== Write output points to stdout ======================")
    for point in (pipeline | StdoutWriter()).process_points(points):
        pass
    print()

    print("====================== Write output chunks to stdout ======================")
    for chunk in (pipeline | StdoutWriter()).process_chunks(points, chunk_size=3):
        pass
    print()


if __name__ == "__main__":
    main()
