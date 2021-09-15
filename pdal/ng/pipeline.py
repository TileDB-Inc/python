from __future__ import annotations

import itertools as it
import json
import subprocess
from abc import ABC, abstractmethod
from typing import (
    Any,
    Dict,
    Iterator,
    List,
    Mapping,
    Optional,
    Sequence,
    Set,
    Union,
    cast,
)

import numpy as np
from numpy.typing import NDArray

from .utils import chunked, rechunk_arrays

_PDAL_DRIVERS = json.loads(
    subprocess.run(["pdal", "--drivers", "--showjson"], capture_output=True).stdout
)

# Point is a numpy structured scalar (numpy.void)
# https://numpy.org/doc/stable/user/basics.rec.html#indexing-with-an-integer-to-get-a-structured-scalar
Point = np.void
# Chunk is a 1-dimensional numpy (structured) array of points
Chunk = NDArray[Point]
PointStream = Iterator[Point]
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

    def __iter__(self) -> PointStream:
        return self.process_points()

    def process_points(self, *chunks: Chunk) -> PointStream:
        return cast(PointStream, self._process(*chunks))

    def process_chunks(self, *chunks: Chunk, chunk_size: int) -> ChunkStream:
        return cast(ChunkStream, self._process(*chunks, chunk_size=chunk_size))

    def _process(
        self, *chunks: Chunk, chunk_size: Optional[int] = None
    ) -> Union[PointStream, ChunkStream]:
        if chunks:
            pipeline = Pipeline(ChunkReader(*chunks))
            pipeline |= self
        else:
            pipeline = self
        # For each stage, determine the input streams based on the input tags and call
        # its {read,process}_{points,chunks} method to compute (lazily) the output stream.
        # This output stream can in turn be used as input to subsequent stage(s).
        tagged_streams: Dict[str, Union[PointStream, ChunkStream]] = {}
        for stage in pipeline._iter_final_stages():
            istreams = tuple(tagged_streams[tag] for tag in stage.inputs)
            ostream: Union[PointStream, ChunkStream]
            if isinstance(stage, Reader):
                assert not istreams
                if chunk_size is None:
                    ostream = stage.read_points()
                else:
                    ostream = stage.read_chunks(chunk_size)
            elif isinstance(stage, (Filter, Writer)):
                assert istreams
                istream: Iterator[np._ArrayOrScalarCommon]
                if len(istreams) == 1:
                    istream = istreams[0]
                else:
                    istream = it.chain.from_iterable(istreams)
                    if chunk_size is not None:
                        # Chaining multiple streams chunked by chunk_size is not
                        # necessarily chunked by chunk_size so we need to rechunk it
                        istream = rechunk_arrays(cast(ChunkStream, istream), chunk_size)
                if chunk_size is None:
                    ostream = stage.process_points(cast(PointStream, istream))
                else:
                    ostream = stage.process_chunks(cast(ChunkStream, istream))
                    # The output stream of a Writer is the same as its input so since the
                    # input is chunked by chunk_size we don't need to rechunk it. For a
                    # Filter this is not the case so we need to rechunk it.
                    if isinstance(stage, Filter):
                        ostream = rechunk_arrays(ostream, chunk_size)
            else:
                assert False
            tagged_streams[stage.tag] = ostream

        # Once the stream of every stage has been determined, return the stream of the
        # last stage that effectively subsumes all the previous ones
        return tagged_streams[stage.tag]

    def _iter_final_stages(self) -> Iterator[Stage]:
        if not self._stages:
            raise ValueError("Empty pipeline is not allowed")

        tags = tuple(stage.tag for stage in self._stages)
        if len(tags) != len(set(tags)):
            raise ValueError(f"Duplicate tag in {tags}")

        input_tags: Set[str] = set()
        default_inputs: List[Stage] = []
        for i, stage in enumerate(self._stages):
            if not isinstance(stage, Reader):
                if stage.inputs:
                    prev_tags = tags[:i]
                    if any(tag not in prev_tags for tag in stage.inputs):
                        raise ValueError(f"{stage} has a previously undefined input")
                else:
                    if not default_inputs:
                        raise ValueError(f"Inputs are required for {stage}")
                    stage = stage.replace(inputs=default_inputs)
                default_inputs.clear()
                input_tags.update(stage.inputs)
            elif stage.inputs:
                raise ValueError(f"Inputs are not allowed for {stage}")
            default_inputs.append(stage)
            yield stage

        # ensure that there's only one sink stage, the last one
        sink_tags = tuple(tag for tag in tags if tag not in input_tags)
        if len(sink_tags) != 1:
            raise ValueError(
                f"Exactly one sink stage is allowed, {len(sink_tags)} found: {sink_tags}"
            )
        if sink_tags[0] != tags[-1]:
            raise ValueError(
                f"The sink stage is {sink_tags[0]}, not the last one {tags[-1]}"
            )


class Stage(ABC):
    def __init__(
        self,
        *,
        type: Optional[str] = None,
        tag: Optional[str] = None,
        inputs: Sequence[Union[Stage, str]] = (),
        **kwargs: Any,
    ):
        self.type = type
        self.tag = tag if tag is not None else str(id(self))
        self.inputs = tuple(i.tag if isinstance(i, Stage) else i for i in inputs)
        self.__dict__.update(kwargs)

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
    def spec(self) -> Mapping[str, Any]:
        return {
            k: v for k, v in self.__dict__.items() if k not in ("type", "inputs") or v
        }

    def __str__(self) -> str:
        return f"{self.__class__.__name__}(type={self.type!r}, tag={self.tag!r})"

    def __or__(self, other: Union[Stage, Pipeline]) -> Pipeline:
        return Pipeline(self) | other

    def replace(self, **kwargs: Any) -> Stage:
        return self.__class__(**dict(self.__dict__, **kwargs))


class Reader(Stage):
    @abstractmethod
    def read_points(self) -> PointStream:
        """Return an iterator of points from the underlying source"""

    def read_chunks(self, chunk_size: int) -> ChunkStream:
        return map(np.array, chunked(self.read_points(), chunk_size))


class ChunkReader(Reader):

    chunks: Sequence[Chunk]

    def __init__(self, *chunks: Chunk):
        super().__init__(chunks=chunks)

    def read_points(self) -> PointStream:
        return it.chain.from_iterable(self.chunks)


class Filter(Stage):
    @abstractmethod
    def process_points(self, point_stream: PointStream) -> PointStream:
        """Filter out or transform points from the given stream"""

    def process_chunks(self, chunk_stream: ChunkStream) -> ChunkStream:
        return (c for c in map(self._filter_chunk, chunk_stream) if len(c) > 0)

    def _filter_chunk(self, chunk: Chunk) -> Chunk:
        return np.fromiter(self.process_points(iter(chunk)), dtype=chunk.dtype)


class Writer(Stage):
    def process_points(self, point_stream: PointStream) -> PointStream:
        for point in point_stream:
            self._write_point(point)
            yield point

    def process_chunks(self, chunk_stream: ChunkStream) -> ChunkStream:
        for chunk in chunk_stream:
            self._write_chunk(chunk)
            yield chunk

    def _write_chunk(self, chunk: Chunk) -> None:
        for point in chunk:
            self._write_point(point)

    @abstractmethod
    def _write_point(self, point: Point) -> None:
        """Write the given point to the underlying sink"""
