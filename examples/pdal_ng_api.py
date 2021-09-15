import json

from pdal.ng import Pipeline, Reader, Filter, Writer


def create_pipeline() -> Pipeline:
    readA = Reader(filename="A.las", spatialreference="EPSG:26916")
    reproj = Filter.reprojection(in_srs="EPSG:26916", out_srs="EPSG:4326", tag="A2")
    readB = Reader(filename="B.las", tag="B")
    merge = Filter.merge(inputs=[reproj, readB])
    write = Writer.gdal(filename="output.tif")
    return readA | reproj | readB | merge | write


def create_pipeline_one_line() -> Pipeline:
    return (
        Reader(filename="A.las", spatialreference="EPSG:26916")
        | Filter.reprojection(in_srs="EPSG:26916", out_srs="EPSG:4326", tag="A2")
        | Reader(filename="B.las", tag="B")
        | Filter.merge(inputs=["A2", "B"])
        | Writer.gdal(filename="output.tif")
    )


def create_pipeline_in_place() -> Pipeline:
    p = Pipeline()
    p |= Reader(filename="A.las", spatialreference="EPSG:26916")
    p |= Filter.reprojection(in_srs="EPSG:26916", out_srs="EPSG:4326", tag="A2")
    p |= Reader(filename="B.las", tag="B")
    p |= Filter.merge(inputs=["A2", "B"])
    p |= Writer.gdal(filename="output.tif")
    return p


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
    run_pipeline("p1", create_pipeline())
    run_pipeline("p2", create_pipeline_one_line())
    run_pipeline("p3", create_pipeline_in_place())

    # p4 = Reader("A.las") | Writer("out1.las") | Reader("B.las") | Writer("out2.las")
    # run_pipeline("p4", p4)
