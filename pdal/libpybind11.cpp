#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>

#include <nlohmann/json.hpp>

#include <pdal/pdal_config.hpp>
#include <pdal/StageFactory.hpp>
#include <pdal/Mesh.hpp>
#include <pdal/PointView.hpp>
#include <pdal/PipelineManager.hpp>
#include <pdal/PipelineExecutor.hpp>
#include "PyDimension.hpp"
#include "PyArray.hpp"
#include "PyPipeline.hpp"

#include "Python/Python.h"
// only here for intellisense; already included from PyArray
#include <numpy/ndarraytypes.h>



namespace py = pybind11;
namespace nl = nlohmann;


namespace pybind11 {
    namespace detail {
        using namespace pdal::python;

        template<>
        struct type_caster<PyArrayObject> {
        public:
            PYBIND11_TYPE_CASTER(PyArrayObject, _("PyArrayObject"));

            bool load(handle src, bool) {
                PyObject* source = src.ptr();
                PyArrayObject* tmp = (PyArrayObject*) source;
                if (!tmp)
                    return false;
                value = *tmp;
                Py_DECREF(&tmp);
                return !(PyErr_Occurred());
            }

//            static handle cast(Array src, return_value_policy, handle) {
//                return PyNullImporter_Type;
//            }
        };
    }
}

namespace pdal {
    using namespace pybind11::literals;
    using namespace pdal::python;

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
    };

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

    // pipeline classes
    class Pipeline {
    public:
        PipelineExecutor* _executor;
        std::vector <std::shared_ptr<Array>> _inputs;

        Pipeline() : _executor(nullptr)
        {};

        virtual ~Pipeline() {
            _inputs.clear();
        };

        Pipeline(const Pipeline &pipeline) : _inputs(pipeline._inputs) {}

        // props
        void setInputs(std::vector<Array*> ndarrays) {
            _inputs.clear();
            for (auto &arr: ndarrays)
            {
                _inputs.push_back(std::make_shared<Array>(arr->));
            }
            _delete_executor();
        }

        std::string pipeline() {
            return _get_executor()->getPipeline();
        }

        std::string metadata() {
            return _get_executor()->getMetadata();
        }

        int getLoglevel() {
            return _get_executor()->getLogLevel();
        }

        void l_setLogLevel(int level) {
            _get_executor()->setLogLevel(level);
        }

        std::string log() {
            return _get_executor()->getLog();
        }

        nl::json schema() {
            return nl::json(_get_executor()->getSchema());
        }

        std::vector<std::shared_ptr<Array>> arrays() {
            PipelineExecutor *executor = _get_executor();
            if (!executor->executed())
                throw std::runtime_error("call execute() before fetching arrays");
            std::vector <std::shared_ptr<Array>> output;
            for (const auto &view: executor->getManagerConst().views()) {
                PyArrayObject* arr(python::viewToNumpyArray(view));
                Array ab(arr);
                output.push_back(std::shared_ptr<Array>(&ab));
            }
            return output;
        }

        std::vector <PyArrayObject> meshes() {
            PipelineExecutor *executor = _get_executor();
            if (!executor->executed())
                throw std::runtime_error("call execute() before fetching arrays");
            std::vector <PyArrayObject> output;
            for (const auto &view: executor->getManagerConst().views()) {
                PyArrayObject *arr(python::meshToNumpyArray(view->mesh()));
                output.push_back(*arr);
            }
            return output;
        }

        int64_t execute() {
            return _get_executor()->execute();
        }

        bool validate() {
            return _get_executor()->validate();
        }

//        Mesh get_meshio(size_t idx)
//        {
//            try
//            {
//                py::object Mesh = py::module_::import("Mesh");
//            }
//            catch (py::error_already_set &e)
//            {
//                if (e.matches(PyExc_ModuleNotFoundError))
//                {
//                    // next line may not be necessary
//                    e.discard_as_unraisable(__func__);
//                    throw std::runtime_error("The get_meshio function can only be used if you have installed meshio. Try pip install meshio");
//                }
//                else
//                {
//                    throw;
//                }
//            };
//            python::Array array = arrays()[idx];
//            Mesh mesh = meshes()[idx];
//            if (meshes.size() == 0)
//            {
//                return NULL;
//            }
//            py::object np = py::module_::import("numpy");
//            py::object stack = np.attr("stack");
//            return Mesh(
//                    stack(py::make_tuple(array["X"], array["Y"], array["Z"]), 1),
//                    py::make_tuple(py::make_tuple("triangle", stack(py::make_tuple(mesh["A"], mesh["B"], mesh["C"]), 1)))
//            );
//        }

        virtual nl::json _json() {
            return nl::json("a");
        }

        size_t _num_inputs() {
            return _inputs.size();
        }

        void _delete_executor() {
            if (_executor != nullptr) {
                _executor = nullptr;
            }
        }

        PipelineExecutor *_get_executor() {
            return _executor;
        }

    };

        class PyPipeline : public Pipeline {
        public:
            using Pipeline::Pipeline;

            using Pipeline::_executor;
            using Pipeline::_inputs;

            nl::json _json() override {
                PYBIND11_OVERRIDE(
                        nl::json,
                        Pipeline,
                        _json,
                );
            }

        };



    PYBIND11_MODULE(libpybind11, m)
    {
    m.doc() = "blank funcs";
    py::bind_vector<std::vector<PyObject*>>(m, "ArrayList");
    py::class_<Pipeline, PyPipeline>(m, "Pipeline")
        .def(py::init<>())
//        .def("__copy__()", [](const Pipeline &self){
//            return Pipeline(self);
//        })
        .def_property("inputs", nullptr, &Pipeline::setInputs)
        .def_property_readonly("pipeline", &Pipeline::pipeline)
        .def_property_readonly("metadata", &Pipeline::metadata)
        .def_property("loglevel", &Pipeline::getLoglevel, &Pipeline::l_setLogLevel)
        .def_property_readonly("log", &Pipeline::log)
        .def_property_readonly("schema", &Pipeline::schema);
//        .def_property_readonly("arrays", &Pipeline::arrays)
//        .def_property_readonly("meshes", &Pipeline::meshes)
//        .def("_json", &Pipeline::_json);
    m.def("getInfoPB11", &getInfoPB11, "getInfo");
    m.def("getDimensionsPB11", &getDimensionsPB11, "getDimensions");
    m.def("infer_reader_driver", &StageFactory::inferReaderDriver,
    "driver"_a);
    m.def("infer_writer_driver", &StageFactory::inferWriterDriver,
    "driver"_a);
    };

}; // namespace pdal