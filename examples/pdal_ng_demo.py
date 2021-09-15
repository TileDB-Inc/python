from pprint import pprint
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
        return -point


class StdoutWriter(pdal.ng.Writer):
    def __init__(self, type: str = "writers.stdout", **kwargs: Any):
        super().__init__(type=type, **kwargs)

    def _write_point(self, point: pdal.ng.Point) -> None:
        print(point)


if __name__ == "__main__":
    points = np.array([(i, 2 * i, 3 * i) for i in range(20)], dtype=np.float32)

    print("========================== Input points ==========================")
    pprint(points)
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
    for point in (pipeline | StdoutWriter()).process_chunks(3, [points]):
        pass
    print()
