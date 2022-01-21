"""
Translate a point cloud input file to a multi-resolution point cloud pyramid TileDB array
"""

import argparse
import contextlib
import logging
import os.path
import tempfile
from concurrent.futures import ThreadPoolExecutor, wait
from typing import Mapping, Optional, Sequence

import numpy as np
import pdal
import tiledb


logger = logging.getLogger(__name__)


def create_pyramid(
    input_path: str,
    uri: str,
    splits: int,
    full_res_uri: Optional[str] = None,
    max_workers: Optional[int] = None,
) -> None:
    tmp_dir = None
    if full_res_uri is None:
        tmp_dir = tempfile.TemporaryDirectory(
            prefix=os.path.splitext(os.path.basename(input_path))[0], suffix=".tdb"
        )
        full_res_uri = tmp_dir.name

    if not tiledb.object_type(full_res_uri):
        _pdal_to_tiledb(input_path, full_res_uri)

    with ThreadPoolExecutor(max_workers) as executor, tiledb.open(full_res_uri) as a:
        tiledb.Array.create(uri, a.schema)
        logger.info("Created empty array at %s", uri)
        with contextlib.ExitStack() as stack:
            timestamp_arrays = tuple(
                stack.enter_context(tiledb.open(uri, mode="w", timestamp=t))
                for t in range(1, splits + 1)
            )
            _partition_array(a, timestamp_arrays, splits, executor)
        _consolidate_vacuum_array_timestamps(uri, splits, executor)

    if tmp_dir is not None:
        tmp_dir.cleanup()


def _pdal_to_tiledb(input_path: str, output_uri: str) -> None:
    pipeline = pdal.Reader(input_path) | pdal.Writer.tiledb(array_name=output_uri)
    logger.info(
        "Start running PDAL pipeline to translate %s to temp TileDB array at %s",
        input_path,
        output_uri,
    )
    num_points = pipeline.execute_streaming()
    logger.info("Written %d points to %s", num_points, output_uri)


def _partition_array(
    input_array: tiledb.Array,
    output_arrays: Sequence[tiledb.Array],
    splits: int,
    executor: ThreadPoolExecutor,
) -> None:
    futures = []
    multirange_indexer = input_array.query(return_incomplete=True).multi_index[:]
    for i, name_values in enumerate(multirange_indexer, start=1):
        for start, output_array in enumerate(output_arrays):
            futures.append(
                executor.submit(_write_array, output_array, start, splits, name_values)
            )
    logger.info("Waiting for %d _write_array tasks to complete", len(futures))
    wait(futures)
    logger.info("%d _write_array tasks completed", len(futures))


def _write_array(
    array: tiledb.Array,
    start: int,
    step: int,
    name_values: Mapping[str, np.ndarray],
) -> None:
    domain = array.schema.domain
    dim_names = tuple(domain.dim(i).name for i in range(domain.ndim))
    dim_values = tuple(name_values[name][start::step] for name in dim_names)
    attr_values = {
        name: values[start::step]
        for name, values in name_values.items()
        if name not in dim_names
    }
    array[dim_values] = attr_values
    logger.debug("Written %d points at timestamp %d", len(dim_values[0]), start + 1)


def _consolidate_vacuum_array_timestamps(
    uri: str, splits: int, executor: ThreadPoolExecutor
) -> None:
    futures = [
        executor.submit(_consolidate_vacuum_array, uri, t) for t in range(1, splits + 1)
    ]
    logger.info("Waiting for %d _consolidate_vacuum_array tasks to complete", splits)
    wait(futures)
    logger.info("%d _consolidate_vacuum_array tasks completed", splits)


def _consolidate_vacuum_array(uri: str, timestamp: Optional[int] = None) -> None:
    timestamp_range = (timestamp, timestamp) if timestamp is not None else None
    tiledb.consolidate(uri, timestamp=timestamp_range)
    logger.debug("Consolidated %s at timestamp %s", uri, timestamp)
    tiledb.vacuum(uri, timestamp=timestamp_range)
    logger.debug("Vacuumed %s at timestamp %s", uri, timestamp)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-i", "--input", required=True, help="Input file path")
    parser.add_argument("-o", "--output", required=True, help="Output TileDB URI")
    parser.add_argument(
        "-f",
        "--full-res",
        help="Full resolution TileDB URI. If not given, the full-res array "
        "is created in a temporary directory and deleted at the end",
    )
    parser.add_argument(
        "-r",
        "--resolution",
        type=int,
        metavar="R",
        help="Resolution levels. The lowest resolution contains 1/R points",
    )
    parser.add_argument(
        "-w",
        "--workers",
        type=int,
        help="Max number of worker threads to work concurrently",
    )
    parser.add_argument(
        "-l",
        "--log-level",
        default="INFO",
        choices=("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"),
        help="Logging verbosity",
    )

    args = parser.parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] (%(threadName)s): %(message)s",
    )
    create_pyramid(
        input_path=args.input,
        uri=args.output,
        splits=args.resolution,
        full_res_uri=args.full_res,
        max_workers=args.workers,
    )


if __name__ == "__main__":
    main()
