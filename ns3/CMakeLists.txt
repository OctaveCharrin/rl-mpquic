# realtime-mpquic: RL-controlled real-time video conferencing over (abstracted)
# multipath QUIC. Struct-based ns3-ai message interface (Python is the brain,
# C++ the body). Symlinked into ns-3-dev/contrib/ai/examples/rl-mpquic by
# scripts/install_ns3_example.sh.

build_lib_example(
        NAME ns3ai_realtime_mpquic
        SOURCE_FILES realtime_mpquic.cc
        LIBRARIES_TO_LINK
            ${libai}
            ${libcore}
            ${libnetwork}
            ${libinternet}
            ${libpoint-to-point}
            ${libapplications}
            ${libtraffic-control}
            ${libflow-monitor}
)

pybind11_add_module(ns3ai_realtime_mpquic_py realtime_mpquic_py.cc)
set_target_properties(ns3ai_realtime_mpquic_py PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(ns3ai_realtime_mpquic_py PRIVATE ${libai})

# Build the Python binding module along with the C++ scenario.
add_dependencies(ns3ai_realtime_mpquic ns3ai_realtime_mpquic_py)
