set(EXTENSION_SRC PyArray.cpp PyPipeline.cpp)

set(extension "libpdalpython")
add_cython_target(${extension} "libpdalpython.pyx" CXX PY3)
add_library(${extension} MODULE ${EXTENSION_SRC} libpdalpython)
target_include_directories(${extension} PRIVATE . ${Python3_NumPy_INCLUDE_DIRS})
target_link_libraries(${extension} ${PDAL_LIBRARIES})
python_extension_module(${extension})

set(pb11ext "libpybind11")
pybind11_add_module(${pb11ext} MODULE libpybind11.cpp)

target_include_directories(${pb11ext}
        PRIVATE
        .
        ${PDAL_INCLUDE_DIRS}
        ${Python3_INCLUDE_DIRS}
        ${Python3_NumPy_INCLUDE_DIRS})

install(TARGETS ${extension} ${pb11ext} LIBRARY DESTINATION "pdal")
