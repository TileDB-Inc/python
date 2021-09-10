from __future__ import annotations

import itertools as it
import json
import subprocess
from abc import ABC, abstractmethod
from typing import Any, Dict, Iterator, List, Mapping, Optional, Sequence, Tuple, Union

from .utils import chunked, rechunked

PDAL_DRIVERS = json.loads(
    subprocess.run(["pdal", "--drivers", "--showjson"], capture_output=True).stdout
)
PDAL_DRIVER_TYPES = [d["name"] for d in PDAL_DRIVERS]

# TODO: set actual PDAL point type
Point = Tuple[float, float]
PointStream = Iterator[Point]
Chunk = Sequence[Point]
ChunkStream = Iterator[Chunk]


class Pipeline:
    def __init__(self, *stages: Stage):
        self._stages = list(stages)

    @property
    def spec(self) -> Sequence[Mapping[str, Any]]:
        return [stage.spec for stage in self._stages]

    def __or__(self, other: Union[Stage, Pipeline]) -> Pipeline:
        new = self.__class__(*self._stages)
        new |= other
        return new

    def __ior__(self, other: Union[Stage, Pipeline]) -> Pipeline:
        if isinstance(other, Stage):
            self._stages.append(other)
        elif isinstance(other, Pipeline):
            self._stages.extend(other._stages)
        else:
            raise TypeError(f"Expected Stage or Pipeline instance, not {other}")
        return self

    def validate(self) -> None:
        stages = self._stages

        if not stages:
            raise ValueError("Empty pipeline is not allowed")

        tags = tuple(stage.tag for stage in stages)
        if len(tags) != len(set(tags)):
            raise ValueError(f"Duplicate tag in {tags}")

        default_inputs: List[Stage] = []
        for i, stage in enumerate(stages):
            if not isinstance(stage, Reader):
                if stage.inputs:
                    for input in stage.inputs:
                        if input not in tags[:i]:
                            raise ValueError(f"Undefined stage tag '{input}'")
                else:
                    stages[i] = stage = stage.replace(inputs=default_inputs)
                default_inputs.clear()
            elif stage.inputs:
                raise ValueError(f"Inputs not permitted for reader '{stage}'")
            default_inputs.append(stage)

        # ensure the only sink stage is the last one
        all_inputs = set(input for stage in stages for input in stage.inputs)
        sinks = tuple(tag for tag in tags if tag not in all_inputs)
        if len(sinks) == 1:
            assert sinks[0] == tags[-1]
        else:
            raise ValueError(
                f"Exactly one sink stage is allowed, {len(sinks)} found: {sinks}"
            )

    def __iter__(self) -> PointStream:
        return self.process_points()

    def process_points(self, *point_streams: PointStream) -> PointStream:
        # TODO: create new Reader stages for each point_stream
        self.validate()
        # For each stage, determine the input(s) based on the input tags and call its
        # process_points() to get its (lazy) output. This output in turn may be used as
        # input to subsequent stage(s). Once all iterators have been determined, return
        # the iterator of the last stage that subsumes all the previous ones.
        tagged_point_streams: Dict[str, PointStream] = {}
        for stage in self._stages:
            input_streams = tuple(tagged_point_streams[tag] for tag in stage.inputs)
            tagged_point_streams[stage.tag] = stage.process_points(*input_streams)
        return tagged_point_streams[self._stages[-1].tag]

    def process_chunks(self, n: int, *chunk_streams: ChunkStream) -> ChunkStream:
        # TODO: create new Reader stages for each chunk_stream
        self.validate()
        # For each stage, determine the input(s) based on the input tags and call its
        # process_chunks() to get its (lazy) output. This output in turn may be used as
        # input to subsequent stage(s). Once all iterators have been determined, return
        # the iterator of the last stage that subsumes all the previous ones.
        tagged_chunk_streams: Dict[str, ChunkStream] = {}
        for stage in self._stages:
            input_streams = tuple(tagged_chunk_streams[tag] for tag in stage.inputs)
            tagged_chunk_streams[stage.tag] = stage.process_chunks(n, *input_streams)
        return tagged_chunk_streams[self._stages[-1].tag]


class Stage(ABC):
    def __init__(self, **kwargs: Any):
        stage_kind = self.__class__.__name__.lower()
        stage_type = kwargs.get("type")
        if stage_type is not None:
            if not stage_type.startswith(f"{stage_kind}s."):
                raise ValueError(f"Invalid {stage_kind} type {stage_type!r}")
            if stage_type not in PDAL_DRIVER_TYPES:
                raise ValueError(f"Unknown stage type {stage_type!r}")

        if "tag" not in kwargs:
            kwargs["tag"] = str(id(self))

        if "inputs" in kwargs:
            kwargs["inputs"] = tuple(
                input.tag if isinstance(input, Stage) else input
                for input in kwargs["inputs"]
            )

        self._kwargs = kwargs

    @property
    def inputs(self) -> Tuple[str, ...]:
        return self._kwargs.get("inputs", ())

    @property
    def tag(self) -> str:
        return str(self._kwargs["tag"])

    @property
    def spec(self) -> Mapping[str, Any]:
        return dict(self._kwargs)

    def __or__(self, other: Union[Stage, Pipeline]) -> Pipeline:
        return Pipeline(self) | other

    def replace(self, **kwargs: Any) -> Stage:
        return self.__class__(**dict(self._kwargs, **kwargs))

    @abstractmethod
    def process_points(self, *point_streams: PointStream) -> PointStream:
        ...

    def process_chunks(self, n: int, *chunk_streams: ChunkStream) -> ChunkStream:
        if chunk_streams:
            return map(self._process_chunk, rechunked(it.chain(*chunk_streams), n))
        else:
            return chunked(self.process_points(), n)

    def _process_chunk(self, chunk: Chunk) -> Chunk:
        return list(self.process_points(iter(chunk)))


class Reader(Stage):
    def __init__(self, filename: str, **kwargs: Any):
        super().__init__(filename=filename, **kwargs)

    def __str__(self) -> str:
        return f"Reader(filename={self._kwargs['filename']!r})"

    def process_points(self, *point_streams: PointStream) -> PointStream:
        # TODO
        yield from self._kwargs["filename"]


class Filter(Stage):
    def __init__(self, type: str, **kwargs: Any):
        super().__init__(type=type, **kwargs)

    def __str__(self) -> str:
        return f"Filter(type={self._kwargs['type']!r})"

    def process_points(self, *point_streams: PointStream) -> PointStream:
        for point_stream in point_streams:
            for point in map(self._filter_point, point_stream):
                if point is not None:
                    yield point

    def _filter_point(self, point: Point) -> Optional[Point]:
        # TODO
        return f"{self}({point})"


class Writer(Stage):
    def __init__(self, filename: str, **kwargs: Any):
        super().__init__(filename=filename, **kwargs)

    def __str__(self) -> str:
        return f"Writer(filename={self._kwargs['filename']!r})"

    def process_points(self, *point_streams: PointStream) -> PointStream:
        for point_stream in point_streams:
            for point in point_stream:
                self._write_point(point)
                yield point

    def _process_chunk(self, chunk: Chunk) -> Chunk:
        self._write_chunk(chunk)
        return chunk

    def _write_chunk(self, chunk: Chunk) -> None:
        for point in chunk:
            self._write_point(point)

    def _write_point(self, point: Point) -> None:
        # TODO
        print(f"\t{self}({point})")


def run_pipeline(name: str, pipeline: Pipeline) -> None:
    print(f"* {name}.spec:", json.dumps(pipeline.spec, indent=4))
    print(f"* {name}.process_points():")
    print("Output:")
    points = list(pipeline.process_points())
    print("Points:")
    for point in points:
        print(f"\t{point}")
    print(f"* {name}.process_chunks(2):")
    print("Output:")
    chunks = list(pipeline.process_chunks(2))
    print("Chunks:")
    for chunk in chunks:
        print(f"\t{chunk}")
    print()


if __name__ == "__main__":
    readA = Reader("A.las", spatialreference="EPSG:26916")
    reproj = Filter("filters.reprojection", in_srs="EPSG:26916", out_srs="EPSG:4326")
    readB = Reader("B.las")
    merge = Filter("filters.merge", inputs=[reproj, readB])
    write = Writer("output.tif", type="writers.gdal")
    p = readA | reproj | readB | merge | write
    run_pipeline("p", p)

    p1 = Pipeline()
    p1 |= readA
    p1 |= reproj
    p1 |= readB
    p1 |= merge
    p1 |= write
    p1.validate()
    assert p1.spec == p.spec

    p2 = (
        Reader("A.las", spatialreference="EPSG:26916")
        | Reader("B.las", tag="readB")
        | Filter(
            "filters.reprojection",
            in_srs="EPSG:26916",
            out_srs="EPSG:4326",
            tag="reproj",
        )
        | Reader("C.las", tag="readC")
        | Filter("filters.merge", inputs=["reproj", "readC"])
        | Writer("output.tif", type="writers.gdal")
    )
    run_pipeline("p2", p2)

    p3 = readA | Writer("out1.las") | readB | Writer("out2.las")
    run_pipeline("p3", p3)
