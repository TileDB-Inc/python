from __future__ import annotations

import itertools as it
import json
import subprocess
from abc import ABC, abstractmethod
from typing import Any, Dict, Iterator, List, Mapping, Optional, Sequence, Tuple, Union

from .utils import chunked, rechunked

_PDAL_DRIVERS = json.loads(
    subprocess.run(["pdal", "--drivers", "--showjson"], capture_output=True).stdout
)

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

    def finalize(self) -> None:
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
        self.finalize()
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
        self.finalize()
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
        if "tag" not in kwargs:
            kwargs["tag"] = str(id(self))
        if "inputs" in kwargs:
            kwargs["inputs"] = tuple(
                input.tag if isinstance(input, Stage) else input
                for input in kwargs["inputs"]
            )
        self._kwargs = kwargs

    def __init_subclass__(cls) -> None:
        selected_prefix = cls.__name__.lower() + "s"
        for driver in _PDAL_DRIVERS:
            full_name = driver["name"]
            prefix, _, suffix = full_name.partition(".")
            if prefix == selected_prefix:
                cls.__set_constructor(full_name, suffix, driver["description"])

    @classmethod
    def __set_constructor(cls, type: str, name: str, description: str) -> None:
        constructor = lambda *args, **kwargs: cls(*args, **kwargs, type=type)
        constructor.__name__ = name
        constructor.__qualname__ = f"{cls.__name__}.{name}"
        constructor.__doc__ = description
        setattr(cls, name, staticmethod(constructor))

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
