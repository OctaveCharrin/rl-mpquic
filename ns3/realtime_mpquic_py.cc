/*
 * realtime_mpquic_py.cc — pybind11 binding for the realtime-mpquic shared-memory
 * structs. Built into the module `ns3ai_realtime_mpquic_py`, imported by the
 * Python Ns3DataPlane (src/ns3env/dataplane.py).
 *
 * EnvStruct (C++ -> Python) is read-only on the Python side; its per-path arrays
 * are exposed through index getters (pybind11 cannot def_readwrite a raw C
 * array). ActStruct (Python -> C++) exposes read/write scalars plus a split
 * setter so Python can write the per-path traffic-split ratios.
 */

#include "realtime_mpquic.h"

#include <ns3/ai-module.h>

#include <pybind11/pybind11.h>

namespace py = pybind11;

PYBIND11_MODULE(ns3ai_realtime_mpquic_py, m)
{
    py::class_<EnvStruct>(m, "PyEnvStruct")
        .def(py::init<>())
        .def_readwrite("numPaths", &EnvStruct::numPaths)
        .def_readwrite("clockS", &EnvStruct::clockS)
        .def_readwrite("done", &EnvStruct::done)
        .def_readwrite("appDecisionDue", &EnvStruct::appDecisionDue)
        .def_readwrite("currentBitrateKbps", &EnvStruct::currentBitrateKbps)
        .def_readwrite("rttMs", &EnvStruct::rttMs)
        .def_readwrite("jitterMs", &EnvStruct::jitterMs)
        .def_readwrite("loss", &EnvStruct::loss)
        .def_readwrite("throughputMbps", &EnvStruct::throughputMbps)
        .def_readwrite("lastLatencyMs", &EnvStruct::lastLatencyMs)
        .def_readwrite("lastJitterMs", &EnvStruct::lastJitterMs)
        .def_readwrite("lastLoss", &EnvStruct::lastLoss)
        .def_readwrite("lastBytes", &EnvStruct::lastBytes)
        .def("cwnd", [](EnvStruct& e, uint32_t i) { return e.cwnd[i]; })
        .def("srtt", [](EnvStruct& e, uint32_t i) { return e.srttMs[i]; })
        .def("bufferOcc", [](EnvStruct& e, uint32_t i) { return e.bufferOcc[i]; })
        .def("pathThroughput", [](EnvStruct& e, uint32_t i) { return e.pathThroughputMbps[i]; })
        .def("pathLoss", [](EnvStruct& e, uint32_t i) { return e.pathLoss[i]; })
        .def("pathActive", [](EnvStruct& e, uint32_t i) { return e.pathActive[i]; });

    py::class_<ActStruct>(m, "PyActStruct")
        .def(py::init<>())
        .def_readwrite("command", &ActStruct::command)
        .def_readwrite("targetBitrateKbps", &ActStruct::targetBitrateKbps)
        .def("setSplit", [](ActStruct& a, uint32_t i, double v) { a.splitRatio[i] = v; })
        .def("getSplit", [](ActStruct& a, uint32_t i) { return a.splitRatio[i]; });

    py::class_<ns3::Ns3AiMsgInterfaceImpl<EnvStruct, ActStruct>>(m, "Ns3AiMsgInterfaceImpl")
        .def(py::init<bool,
                      bool,
                      bool,
                      uint32_t,
                      const char*,
                      const char*,
                      const char*,
                      const char*>())
        .def("PyRecvBegin", &ns3::Ns3AiMsgInterfaceImpl<EnvStruct, ActStruct>::PyRecvBegin)
        .def("PyRecvEnd", &ns3::Ns3AiMsgInterfaceImpl<EnvStruct, ActStruct>::PyRecvEnd)
        .def("PySendBegin", &ns3::Ns3AiMsgInterfaceImpl<EnvStruct, ActStruct>::PySendBegin)
        .def("PySendEnd", &ns3::Ns3AiMsgInterfaceImpl<EnvStruct, ActStruct>::PySendEnd)
        .def("PyGetFinished", &ns3::Ns3AiMsgInterfaceImpl<EnvStruct, ActStruct>::PyGetFinished)
        .def("GetCpp2PyStruct",
             &ns3::Ns3AiMsgInterfaceImpl<EnvStruct, ActStruct>::GetCpp2PyStruct,
             py::return_value_policy::reference)
        .def("GetPy2CppStruct",
             &ns3::Ns3AiMsgInterfaceImpl<EnvStruct, ActStruct>::GetPy2CppStruct,
             py::return_value_policy::reference);
}
