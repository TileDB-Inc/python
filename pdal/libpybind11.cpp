#include <pybind11/pybind11.h>
#include <iostream>
#include <pdal/StageFactory.hpp>


namespace pdal {
    std::string getInfoPb() {
        return "some info";
    }

    std::string getDimensionsPb() {
        return "dims details";
    }

    namespace py = pybind11;
    using namespace pybind11::literals;

    PYBIND11_MODULE(libpybind11, m)
    {
    m.doc() = "blank funcs";

    m.def("getInfoPb", &getInfoPb, "getInfo");
    m.def("getDimensionsPb", &getDimensionsPb, "getDimensions");
    m.def("infer_reader_driver", &StageFactory::inferReaderDriver,
          "driver"_a);
    m.def("infer_writer_driver", &StageFactory::inferWriterDriver,
          "driver"_a);
};

}; // namespace pdal