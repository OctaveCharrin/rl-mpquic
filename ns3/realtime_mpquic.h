/*
 * realtime_mpquic.h — shared-memory message structs for the RL-controlled
 * real-time video-conferencing-over-(abstracted)-multipath-QUIC scenario.
 *
 * These two structs are the wire format of the ns3-ai shared-memory bridge.
 * They are included by BOTH the C++ scenario (realtime_mpquic.cc) and the
 * pybind11 binding (realtime_mpquic_py.cc) so the layout is identical on both
 * sides, and they mirror the Python snapshot in src/ns3env/dataplane.py.
 *
 * Decision epoch = ONE video FRAME (real-time push at `fps` frames/sec). The
 * bridge therefore exchanges one Env/Act pair per frame (~33 ms of sim time at
 * 30 fps). Two RL agents run at two timescales over this single bridge:
 *
 *   - Transport agent: acts every frame; sets splitRatio[] (how to split the
 *     next frame's bytes across the subflows).
 *   - App agent: acts only on a 1 s cadence (gated by EnvStruct.appDecisionDue);
 *     sets targetBitrateKbps, which PERSISTS in ActStruct between app decisions.
 */

#ifndef REALTIME_MPQUIC_H
#define REALTIME_MPQUIC_H

#include <cstdint>

// Upper bound on candidate paths (subflows). Fixed-size arrays keep the shared
// memory layout POD/trivially-copyable, which the struct-based msg interface
// requires. Must match kMaxPaths usage in Python (we only read numPaths slots).
static constexpr uint32_t kMaxPaths = 8;

// C++ -> Python: observable state at a decision point (one per frame), plus the
// realized result of the most-recently-completed frame.
struct EnvStruct
{
    uint32_t numPaths;
    double clockS;            // current sim time (s)
    uint8_t done;             // 1 once the episode horizon is reached
    uint8_t appDecisionDue;   // 1 on a 1 s boundary -> Python also runs App agent

    // --- App-level aggregate state (the WebRTC sender's view) -------------- //
    double currentBitrateKbps; // target bitrate currently in effect
    double rttMs;              // aggregate (split-weighted) smoothed RTT
    double jitterMs;           // EWMA inter-frame latency variation
    double loss;               // app loss estimate: EWMA late/lost-frame fraction
    double throughputMbps;     // aggregate delivered goodput (EWMA)

    // --- Per-path transport state (length == numPaths) -------------------- //
    double cwnd[kMaxPaths];               // congestion window (bytes)
    double srttMs[kMaxPaths];             // smoothed RTT per path
    double bufferOcc[kMaxPaths];          // app send-backlog not yet handed to TCP (bytes)
    double pathThroughputMbps[kMaxPaths]; // EWMA delivered goodput per path
    double pathLoss[kMaxPaths];           // network loss estimate per path [0,1]

    // Per-path liveness mask: 1 = live, 0 = churned out (its bytes are lost).
    // With dynamics disabled every slot < numPaths is 1. Mirrors
    // FrameObs.path_active in src/ns3env/dataplane.py.
    uint8_t pathActive[kMaxPaths];

    // --- Realized result of the most-recently-completed frame ------------- //
    // lastBytes == 0 before the first frame completes.
    double lastLatencyMs; // generation -> last-byte-received latency
    double lastJitterMs;  // |latency - previous frame latency|
    double lastLoss;      // 1.0 if that frame missed its deadline, else network loss
    uint32_t lastBytes;   // bytes delivered for that frame
};

// Action commands. One NS-3 process serves many episodes (the simulation is
// continuing), so Python drives episode boundaries in-band rather than by
// relaunching the process (ns3-ai allows only one shared-memory creator per
// Python process).
enum ActCommand : int32_t
{
    ACT_STEP = 0,      // generate/push the next frame using the fields below
    ACT_RESET = 1,     // start a new episode (reset counters; keep sim running)
    ACT_TERMINATE = 2, // end the NS-3 process
};

// Python -> C++: the agents' action for the next frame.
struct ActStruct
{
    int32_t command;              // one of ActCommand
    double targetBitrateKbps;     // App action: encoder target (persists)
    double splitRatio[kMaxPaths]; // Transport action: per-path fractions (sum ~ 1)
};

#endif // REALTIME_MPQUIC_H
