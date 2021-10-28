#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include <pdal/pdal_config.hpp>
#include <pdal/StageFactory.hpp>
#include <pdal/PipelineExecutor.hpp>

#include "PyArray.hpp"
#include "PyDimension.hpp"
#include "PyPipeline.hpp"

namespace py = pybind11;

namespace pdal {

    using namespace pybind11::literals;
    using namespace pybind11::detail;
    using namespace pdal::python;

    py::object getInfo() {
        using namespace Config;
        return py::module_::import("types").attr("SimpleNamespace")(
                "version"_a = versionString(),
                "major"_a = versionMajor(),
                "minor"_a = versionMinor(),
                "patch"_a = versionPatch(),
                "debug"_a = debugInformation(),
                "sha1"_a = sha1(),
                "plugin"_a = pluginInstallPath()
        );
    };

    std::vector<py::dict> getDimensions() {
        py::object np = py::module_::import("numpy");
        py::object dtype = np.attr("dtype");
        std::vector<py::dict> dims;
        for (const auto& dim: getValidDimensions())
        {
            py::dict d(
                "name"_a=dim.name,
                "description"_a=dim.description,
                "dtype"_a=dtype(dim.type + std::to_string(dim.size))
            );
            dims.push_back(std::move(d));
        }
        return dims;
    };

    class Pipeline {
    public:
        Pipeline() {}

        virtual ~Pipeline() { _inputs.clear(); }

        int64_t execute() { return _get_executor()->execute(); }

        // writable props

        void setInputs(std::vector<py::array> ndarrays) {
            _inputs.clear();
            for (const auto& ndarray: ndarrays) {
                PyArrayObject* ndarray_ptr = (PyArrayObject*)ndarray.ptr();
                _inputs.push_back(std::make_shared<Array>(ndarray_ptr));
            }
            _del_executor();
        }

        int getLoglevel() { return _loglevel; }

        void setLogLevel(int level) {
            _loglevel = level;
            _del_executor();
        }

        // readable props

        std::string log() { return _get_executor()->getLog(); }

        std::string schema() { return _get_executor()->getSchema(); }

        std::string pipeline() { return _get_executor()->getPipeline(); }

        std::string metadata() { return _get_executor()->getMetadata(); }

        std::vector<py::array> arrays() {
            PipelineExecutor* executor = _get_executor();
            if (!executor->executed())
                throw std::runtime_error("call execute() before fetching arrays");
            std::vector<py::array> output;
            for (const auto &view: executor->getManagerConst().views()) {
                PyArrayObject* arr(viewToNumpyArray(view));
                output.push_back(py::reinterpret_steal<py::array>((PyObject*)arr));
            }
            return output;
        }

        std::vector<py::array> meshes() {
            PipelineExecutor* executor = _get_executor();
            if (!executor->executed())
                throw std::runtime_error("call execute() before fetching arrays");
            std::vector<py::array> output;
            for (const auto &view: executor->getManagerConst().views()) {
                PyArrayObject* arr(meshToNumpyArray(view->mesh()));
                output.push_back(py::reinterpret_steal<py::array>((PyObject*)arr));
            }
            return output;
        }

        std::string _get_json() const {
            PYBIND11_OVERRIDE_PURE(std::string, Pipeline, _get_json);
        }

        bool _has_inputs() { return !_inputs.empty(); }

        void _copy_inputs(const Pipeline& other) { _inputs = other._inputs; }

        void _del_executor() { _executor.reset(); }

    private:
        std::shared_ptr<PipelineExecutor> _executor;
        std::vector <std::shared_ptr<Array>> _inputs;
        int _loglevel;

        PipelineExecutor* _get_executor() {
            if (!_executor)
            {
                std::string json = _get_json();
                PipelineExecutor* executor = new PipelineExecutor(json);
                executor->setLogLevel(_loglevel);
                readPipeline(executor, json);
                addArrayReaders(executor, _inputs);
                _executor.reset(executor);
            }
            return _executor.get();
        }
    };

    PYBIND11_MODULE(libpybind11, m)
    {
    py::class_<Pipeline>(m, "Pipeline")
        .def(py::init<>())
        .def("execute", &Pipeline::execute)
        .def_property("inputs", nullptr, &Pipeline::setInputs)
        .def_property("loglevel", &Pipeline::getLoglevel, &Pipeline::setLogLevel)
        .def_property_readonly("log", &Pipeline::log)
        .def_property_readonly("schema", &Pipeline::schema)
        .def_property_readonly("pipeline", &Pipeline::pipeline)
        .def_property_readonly("metadata", &Pipeline::metadata)
        .def_property_readonly("arrays", &Pipeline::arrays)
        .def_property_readonly("meshes", &Pipeline::meshes)
        .def_property_readonly("_has_inputs", &Pipeline::_has_inputs)
        .def("_copy_inputs", &Pipeline::_copy_inputs)
        .def("_get_json", &Pipeline::_get_json)
        .def("_del_executor", &Pipeline::_del_executor);
    m.def("getInfo", &getInfo, "getInfo");
    m.def("getDimensions", &getDimensions, "getDimensions");
    m.def("infer_reader_driver", &StageFactory::inferReaderDriver,
    "driver"_a);
    m.def("infer_writer_driver", &StageFactory::inferWriterDriver,
    "driver"_a);
    };

}; // namespace pdal
