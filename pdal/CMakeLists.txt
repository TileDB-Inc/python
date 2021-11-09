set(EXTENSION_SRC PyArray.cpp PyPipeline.cpp)

set(extension "libpdalpython")
add_cython_target(${extension} "libpdalpython.pyx" CXX PY3)
add_library(${extension} MODULE ${EXTENSION_SRC} libpdalpython)
target_include_directories(${extension} PRIVATE . ${Python3_NumPy_INCLUDE_DIRS})
target_link_libraries(${extension} ${PDAL_LIBRARIES})
python_extension_module(${extension})
set(pb11ext "libpybind11")
message(WARNING "something" arrow_shared ${Arrow_INCLUDE_DIR})
pybind11_add_module(${pb11ext} MODULE ${EXTENSION_SRC} libpybind11.cpp)

execute_process(
        COMMAND "${PYTHON_EXECUTABLE}" -c
        "import pyarrow as pa; print(pa.get_include())"
        OUTPUT_VARIABLE _tmp_dir_a
        OUTPUT_STRIP_TRAILING_WHITESPACE COMMAND_ECHO STDOUT)
set(Arrow_INCLUDE_DIR ${_tmp_dir_a})

message(WARNING "SOMETHING" ${_tmp_dir_b})
target_include_directories(${pb11ext} PRIVATE ${Python3_NumPy_INCLUDE_DIRS} ${Arrow_INCLUDE_DIR})
target_link_libraries(${pb11ext} PRIVATE ${PDAL_LIBRARIES} "arrow" "arrow_python")

install(TARGETS ${extension} ${pb11ext} LIBRARY DESTINATION "pdal")
