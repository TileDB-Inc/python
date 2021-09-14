from pprint import pprint
from typing import Any, Iterable, Optional

import numpy as np
import pdal.ng


class NumpyReader(pdal.ng.Reader):
    def __init__(
        self,
        points: Iterable[pdal.ng.Point],
        type: str = "readers.numpy",
        **kwargs: Any,
    ):
        super().__init__(points=points, type=type, **kwargs)

    def process_points(
        self, *point_streams: pdal.ng.PointStream
    ) -> pdal.ng.PointStream:
        return iter(self.points)


class EvenFilter(pdal.ng.Filter):
    def __init__(self, dim_idx: int, type: str = "filters.even", **kwargs: Any):
        super().__init__(dim_idx=dim_idx, type=type, **kwargs)

    def _filter_point(self, point: pdal.ng.Point) -> Optional[pdal.ng.Point]:
        return point if point[self.dim_idx] % 2 == 0 else None


class NegateFilter(pdal.ng.Filter):
    def __init__(self, type: str = "filters.negate", **kwargs: Any):
        super().__init__(type=type, **kwargs)

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

    pipeline = NumpyReader(points) | NegateFilter() | EvenFilter(dim_idx=2)
    print("========================== Pipeline spec ==========================")
    pprint(pipeline.spec)
    print()

    print("========================== Output points ==========================")
    for i, point in enumerate(pipeline):
        print(i, point)
    print()

    print("========================== Output point chunks ==========================")
    for i, chunk in enumerate(pipeline.process_chunks(3)):
        print(f"chunk {i}: {chunk.__class__.__name__}, {len(chunk)} points")
        for point in chunk:
            print("\t", point)
    print()

    print("========================== Pipe output to stdout ==========================")
    for point in pipeline | StdoutWriter():
        pass
