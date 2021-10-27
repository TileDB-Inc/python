#include <memory>
#include <pybind11/pybind11.h>
#include <pybind11/attr.h>
#include <pybind11/buffer_info.h>
#include <pybind11/complex.h>
#include <pybind11/embed.h>
#include <pybind11/functional.h>
#include <pybind11/operators.h>
#include <pybind11/options.h>

#include <pybind11/stl.h>
#include <pybind11/pytypes.h>
#include <pybind11/iostream.h>
#include <pybind11/eval.h>
#include <pybind11/cast.h>

#include <pybind11/stl_bind.h>
#include <pybind11/numpy.h>
#include "nlohmann/json.hpp"
#include "pybind11_json/pybind11_json.hpp"

#include <pdal/pdal_config.hpp>
#include <pdal/StageFactory.hpp>
#include <pdal/Mesh.hpp>
#include <pdal/PointView.hpp>
#include <pdal/PipelineManager.hpp>
#include <pdal/PipelineExecutor.hpp>
#include "PyDimension.hpp"
#include "PyArray.hpp"
#include "PyPipeline.hpp"

namespace py = pybind11;

namespace pdal {

    class PipelineExecShareable : public PipelineExecutor, public std::enable_shared_from_this<PipelineExecShareable>
    {
    public:
        PipelineExecShareable(std::string const& json) : PipelineExecutor(json) {}
    };


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
            py::dict d("name"_a=dim.name, "description"_a=dim.description, "dtype"_a=dtype(dim.type + std::to_string(dim.size)));
            dims.push_back(std::move(d));
        }
        return dims;
    };


    class Pipeline
    {
    public:
        std::shared_ptr<PipelineExecShareable> _executor;
        std::vector <std::shared_ptr<Array>> _inputs;
        int _loglevel;

        Pipeline() {}

        Pipeline(const Pipeline&) {}

        virtual ~Pipeline() {
            _inputs.clear();
        }

        virtual std::shared_ptr<Pipeline> clone() const = 0;

        int64_t execute() {
            return _get_executor()->execute();
        }

        // writable props

        void setInputs(py::list ndarrays) {
            _inputs.clear();
            for (py::handle arr: ndarrays)
            {
                    py::array py_array = py::cast<py::array>(arr);
                    PyObject* pyobj_array = py_array.ptr();
                    std::shared_ptr<Array> array = std::make_shared<Array>((PyArrayObject*) pyobj_array);
                    _inputs.push_back(std::move(array));
            }
            _del_executor();
        }

        int getLoglevel() {
            return _loglevel;
        }

        void setLogLevel(int level) {
            _loglevel = level;
            _del_executor();
        }

        // readable props

        std::string log() {
            return _get_executor()->getLog();
        }

        std::string schema() {
            return _get_executor()->getSchema();
        }

        std::string pipeline() {
            return _get_executor()->getPipeline();
        }

        std::string metadata() {
            return _get_executor()->getMetadata();
        }

        std::vector<std::shared_ptr<Array>> arrays() {
            PipelineExecShareable* executor = _get_executor();
            if (!executor->executed())
                throw std::runtime_error("call execute() before fetching arrays");
            std::vector <std::shared_ptr<Array>> output;
            for (const auto &view: executor->getManagerConst().views()) {
                PyArrayObject* arr(python::viewToNumpyArray(view));
                output.push_back(std::make_shared<Array>((PyArrayObject*) arr));
            }
            return output;
        }

        std::vector<std::shared_ptr<Array>> meshes() {
            PipelineExecShareable* executor = _get_executor();
            if (!executor->executed())
                throw std::runtime_error("call execute() before fetching arrays");
            std::vector<std::shared_ptr<Array>> output;
            for (const auto &view: executor->getManagerConst().views()) {
                PyArrayObject *arr(python::meshToNumpyArray(view->mesh()));
                output.push_back(std::make_shared<Array>((PyArrayObject*) arr));
            }
            return output;
        }

    protected:

        virtual std::string _json() = 0;

        bool _has_inputs() {
            return !_inputs.empty();
        }

        void _del_executor() {
            _executor.reset();
        }

        PipelineExecShareable* _get_executor(bool set_new = true) {
            if (!_executor && set_new)
            {
                _executor = std::make_shared<PipelineExecShareable>(_json());
                _executor.get()->setLogLevel(_loglevel);
                readPipeline(_executor.get(), _json());
                addArrayReaders(_executor.get(), _inputs);
            }
            return _executor.get();
        }

    };

    class PyPipeline : public Pipeline {
    public:
        using Pipeline::Pipeline;

        PyPipeline(const Pipeline& pipeline) : Pipeline(pipeline) {}

        std::shared_ptr<Pipeline> clone() const override {
            auto self = py::cast(this);
            auto cloned = self.attr("clone")();

            auto keep_python = std::make_shared<py::object>(cloned);
            auto ptr = cloned.cast<PyPipeline*>();

            return std::shared_ptr<Pipeline>(keep_python, ptr);
        }

        std::string _json() override {
            PYBIND11_OVERRIDE_PURE(
                    std::string,
                    Pipeline,
                    _json,
            );
        }

    };

    class PipelinePublic : public Pipeline
    {
    public:
        using Pipeline::_json;
        using Pipeline::_has_inputs;
        using Pipeline::_del_executor;
        using Pipeline::_get_executor;
    };



    PYBIND11_MODULE(libpybind11, m)
    {
    m.doc() = "Pipeline class";
    py::class_<PipelineExecShareable, std::shared_ptr<PipelineExecShareable>>(m, "PipelineExecShareable");
    py::class_<Pipeline, PyPipeline, std::shared_ptr<Pipeline>>(m, "Pipeline", py::dynamic_attr())
        .def(py::init<>())
        .def(py::init<const Pipeline&>())
        .def("execute", &Pipeline::execute)
        .def_property("inputs", nullptr, &Pipeline::setInputs)
        .def_property("loglevel", &Pipeline::getLoglevel, &Pipeline::setLogLevel)
        .def_property_readonly("log", &Pipeline::log)
        .def_property_readonly("schema", &Pipeline::schema)
        .def_property_readonly("pipeline", &Pipeline::pipeline)
        .def_property_readonly("metadata", &Pipeline::metadata)
        .def_property_readonly("arrays", &Pipeline::arrays)
        .def_property_readonly("meshes", &Pipeline::meshes)
        .def("_json", &PipelinePublic::_json)
        .def("_has_inputs", &PipelinePublic::_has_inputs)
        .def("_del_executor", &PipelinePublic::_del_executor)
        .def("_get_executor", &PipelinePublic::_get_executor);
    m.def("getInfo", &getInfo, "getInfo");
    m.def("getDimensions", &getDimensions, "getDimensions");
    m.def("infer_reader_driver", &StageFactory::inferReaderDriver,
    "driver"_a);
    m.def("infer_writer_driver", &StageFactory::inferWriterDriver,
    "driver"_a);
    };

}; // namespace pdal