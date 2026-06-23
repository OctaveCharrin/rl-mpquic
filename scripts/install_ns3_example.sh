#!/usr/bin/env bash
#
# install_ns3_example.sh — wire the vendored C++ scenario into the NS-3 tree and
# build it.
#
# The ns3-ai build system discovers examples under
# contrib/ai/examples/<name>/CMakeLists.txt, registered by an explicit
# add_subdirectory() in contrib/ai/examples/CMakeLists.txt. This script:
#   1. symlinks rl-mpquic/ns3  ->  $NS3_DIR/contrib/ai/examples/rl-mpquic
#   2. idempotently adds `add_subdirectory(rl-mpquic)` to examples/CMakeLists.txt
#   3. builds the ns3ai_realtime_mpquic target (and its pybind module)
#
# The repo stays the single source of truth; nothing is copied into ns-3-dev.
#
# Usage:
#   scripts/install_ns3_example.sh            # build too (default)
#   scripts/install_ns3_example.sh --no-build # just symlink + register
#   NS3_DIR=/path/to/ns-3-dev scripts/install_ns3_example.sh

set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SRC_DIR="${REPO_DIR}/ns3"
NS3_DIR="${NS3_DIR:-${HOME}/ns-3-dev}"
EXAMPLES_DIR="${NS3_DIR}/contrib/ai/examples"
LINK_NAME="rl-mpquic"
LINK_PATH="${EXAMPLES_DIR}/${LINK_NAME}"
EXAMPLES_CMAKE="${EXAMPLES_DIR}/CMakeLists.txt"
TARGET="ns3ai_realtime_mpquic"

DO_BUILD=1
[[ "${1:-}" == "--no-build" ]] && DO_BUILD=0

echo "==> repo:    ${REPO_DIR}"
echo "==> ns-3:    ${NS3_DIR}"

if [[ ! -d "${EXAMPLES_DIR}" ]]; then
    echo "ERROR: ${EXAMPLES_DIR} not found. Is NS3_DIR correct and ns3-ai installed?" >&2
    exit 1
fi
if [[ ! -f "${SRC_DIR}/realtime_mpquic.cc" ]]; then
    echo "ERROR: vendored sources not found in ${SRC_DIR}" >&2
    exit 1
fi

# 1. Symlink the source directory into the examples tree (idempotent).
if [[ -L "${LINK_PATH}" ]]; then
    current="$(readlink "${LINK_PATH}")"
    if [[ "${current}" != "${SRC_DIR}" ]]; then
        echo "==> updating existing symlink (${current} -> ${SRC_DIR})"
        ln -sfn "${SRC_DIR}" "${LINK_PATH}"
    else
        echo "==> symlink already in place: ${LINK_PATH}"
    fi
elif [[ -e "${LINK_PATH}" ]]; then
    echo "ERROR: ${LINK_PATH} exists and is not a symlink; refusing to overwrite." >&2
    exit 1
else
    echo "==> creating symlink: ${LINK_PATH} -> ${SRC_DIR}"
    ln -s "${SRC_DIR}" "${LINK_PATH}"
fi

# 2. Register the subdirectory in examples/CMakeLists.txt (idempotent).
if grep -qE "add_subdirectory\(${LINK_NAME}\)" "${EXAMPLES_CMAKE}"; then
    echo "==> add_subdirectory(${LINK_NAME}) already present"
else
    echo "==> appending add_subdirectory(${LINK_NAME}) to ${EXAMPLES_CMAKE}"
    printf '\nadd_subdirectory(%s)\n' "${LINK_NAME}" >> "${EXAMPLES_CMAKE}"
fi

# 3. Build.
if [[ "${DO_BUILD}" -eq 1 ]]; then
    echo "==> building ${TARGET} (this reconfigures CMake first)"
    cd "${NS3_DIR}"
    ./ns3 build "${TARGET}"
    echo ""
    echo "==> done. Built module:"
    ls -1 "${SRC_DIR}"/ns3ai_realtime_mpquic_py*.so 2>/dev/null \
        || echo "    (warning: no .so found in ${SRC_DIR}; check build output above)"
else
    echo "==> skipping build (--no-build). Build later with:"
    echo "      cd ${NS3_DIR} && ./ns3 build ${TARGET}"
fi
