#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <pdal/pdal_config.hpp>
#include "PyDimension.hpp"
#include <pdal/StageFactory.hpp>
#include "Python/Python.h"


#include <variant>

namespace py = pybind11;

namespace pdal {
    using namespace pybind11::literals;

    std::vector<py::dict> getDimensionsPB11() {
        py::object np = py::module_::import("numpy");
        py::object dtype = np.attr("dtype");
        std::vector<py::dict> dims;
        for (const auto& dim: getValidDimensions())
        {
            py::dict d("name"_a=dim.name, "description"_a=dim.description, "dtype"_a=dtype(dim.type + std::to_string(dim.size)));
            dims.push_back(std::move(d));
        }
        return dims;
    };

    py::object getInfoPB11() {
        using namespace Config;
        return py::make_simple_namespace(
                    "version"_a = versionString(),
                    "major"_a = versionMajor(),
                    "minor"_a = versionMinor(),
                    "patch"_a = versionPatch(),
                    "debug"_a = debugInformation(),
                    "sha1"_a = sha1(),
                    "plugin"_a = pluginInstallPath()
        );
    }

    PYBIND11_MODULE(libpybind11, m)
    {
    m.doc() = "blank funcs";

    m.def("getInfoPB11", &getInfoPB11, "getInfo");
    m.def("getDimensionsPB11", &getDimensionsPB11, "getDimensions");
    m.def("infer_reader_driver", &StageFactory::inferReaderDriver,
          "driver"_a);
    m.def("infer_writer_driver", &StageFactory::inferWriterDriver,
          "driver"_a);
};

}; // namespace pdal