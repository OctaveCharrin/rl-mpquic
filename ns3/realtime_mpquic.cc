/*
 * realtime_mpquic.cc — NS-3 scenario for RL-controlled real-time video
 * conferencing (WebRTC-like) over (abstracted) multipath QUIC, driven from
 * Python via the ns3-ai shared-memory message interface (struct-based).
 *
 * Role: this is the C++ "body" that the Python "brain" (src/ns3env + src/rl)
 * drives. It is intentionally THIN — all RL logic, observation normalization and
 * reward live in Python. C++ only:
 *   1) builds an N-path topology (multi-homed client <-> server), each path a
 *      stock single-path TCP subflow with its own bottleneck + AQM queue +
 *      time-varying UDP cross-traffic (abstracted multipath QUIC);
 *   2) generates a video frame every 1/fps seconds at the current target
 *      bitrate, SPLITS the frame's bytes across the subflows per the agent's
 *      ratio, and pushes each share on that path's persistent TCP connection;
 *   3) tracks per-frame delivery (generation -> last byte received) to derive
 *      latency / jitter / deadline-miss (real-time "loss"), and reports per-path
 *      transport state (cwnd, sRTT, send backlog, goodput, network loss).
 *
 * Decision epoch = ONE FRAME. The struct fields (realtime_mpquic.h) mirror the
 * Python snapshot so the Python Ns3DataPlane just marshals them across.
 *
 * Protocol (matches src/ns3env/dataplane.py::Ns3DataPlane, same as upstream
 * ns3-ai examples): C++ leads with a send. Per decision:
 *     FillObservation(env); CppSend(env); CppRecv(act);
 *     if TERMINATE: stop; if RESET or done: new episode; else apply+generate.
 * Frame delivery runs asynchronously between decisions (the simulator advances
 * 1/fps each step), and completed frames update the "last*" result fields.
 */

#include "realtime_mpquic.h"

#include <ns3/ai-module.h>
#include "ns3/applications-module.h"
#include "ns3/core-module.h"
#include "ns3/flow-monitor-module.h"
#include "ns3/internet-module.h"
#include "ns3/network-module.h"
#include "ns3/point-to-point-module.h"
#include "ns3/traffic-control-module.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <deque>
#include <set>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

using namespace ns3;

NS_LOG_COMPONENT_DEFINE("RealtimeMpquic");

// --------------------------------------------------------------------------- //
// RealtimeSource — server-side persistent bulk sender over one TCP subflow.
// Enqueue(bytes) appends to an application send backlog that is drained into the
// socket as send-buffer space frees (like BulkSendApplication, but re-armable
// and continuous). Tracks smoothed RTT and congestion window from socket traces.
// --------------------------------------------------------------------------- //

class RealtimeSource : public Application
{
  public:
    static TypeId GetTypeId()
    {
        static TypeId tid = TypeId("RealtimeMpquic::RealtimeSource")
                                .SetParent<Application>()
                                .SetGroupName("Applications")
                                .AddConstructor<RealtimeSource>();
        return tid;
    }

    void Configure(Address peer, uint32_t pathIdx)
    {
        m_peer = peer;
        m_pathIdx = pathIdx;
    }

    // Append `bytes` to the send backlog; push immediately if connected.
    void Enqueue(uint32_t bytes)
    {
        m_pending += bytes;
        if (m_connected)
        {
            Drain();
        }
    }

    double GetSrttMs() const { return m_srttMs; }
    double GetCwnd() const { return m_cwnd; }
    double GetBufferOcc() const { return static_cast<double>(m_pending); }

  private:
    void StartApplication() override
    {
        m_socket = Socket::CreateSocket(GetNode(), TcpSocketFactory::GetTypeId());
        m_socket->Bind();
        m_socket->Connect(m_peer);
        m_socket->SetConnectCallback(MakeCallback(&RealtimeSource::ConnectSucceeded, this),
                                     MakeCallback(&RealtimeSource::ConnectFailed, this));
        m_socket->SetSendCallback(MakeCallback(&RealtimeSource::OnSendPossible, this));
        // TcpSocketBase trace sources; connect once the object exists.
        m_socket->TraceConnectWithoutContext("RTT", MakeCallback(&RealtimeSource::RttTrace, this));
        m_socket->TraceConnectWithoutContext("CongestionWindow",
                                             MakeCallback(&RealtimeSource::CwndTrace, this));
    }

    void StopApplication() override
    {
        if (m_socket)
        {
            m_socket->Close();
            m_socket = nullptr;
        }
    }

    void ConnectSucceeded(Ptr<Socket>)
    {
        m_connected = true;
        if (m_pending > 0)
        {
            Drain();
        }
    }

    void ConnectFailed(Ptr<Socket>)
    {
        NS_LOG_WARN("RealtimeSource path " << m_pathIdx << " connect failed");
    }

    // Push as much of the backlog as the send buffer currently allows.
    void Drain()
    {
        while (m_pending > 0)
        {
            uint32_t avail = m_socket->GetTxAvailable();
            if (avail == 0)
            {
                break; // wait for OnSendPossible
            }
            uint32_t toSend = static_cast<uint32_t>(std::min<uint64_t>(m_pending, avail));
            int sent = m_socket->Send(Create<Packet>(toSend));
            if (sent <= 0)
            {
                break;
            }
            m_pending -= static_cast<uint64_t>(sent);
        }
    }

    void OnSendPossible(Ptr<Socket>, uint32_t)
    {
        if (m_pending > 0)
        {
            Drain();
        }
    }

    void RttTrace(Time, Time newRtt)
    {
        double sample = newRtt.GetSeconds() * 1000.0;
        m_srttMs = (m_srttMs <= 0.0) ? sample : 0.85 * m_srttMs + 0.15 * sample;
    }

    void CwndTrace(uint32_t, uint32_t newCwnd) { m_cwnd = static_cast<double>(newCwnd); }

    Ptr<Socket> m_socket;
    Address m_peer;
    uint32_t m_pathIdx = 0;
    uint64_t m_pending = 0; // application send backlog (bytes not yet handed to TCP)
    bool m_connected = false;
    double m_srttMs = 0.0;
    double m_cwnd = 0.0;
};

// --------------------------------------------------------------------------- //
// RealtimeSink — client-side receiver per path. Reports the cumulative bytes
// delivered (and the current time) on every read, so the controller can resolve
// which frame shares have completed by comparing against per-share watermarks.
// --------------------------------------------------------------------------- //

class RealtimeSink : public Application
{
  public:
    static TypeId GetTypeId()
    {
        static TypeId tid = TypeId("RealtimeMpquic::RealtimeSink")
                                .SetParent<Application>()
                                .SetGroupName("Applications")
                                .AddConstructor<RealtimeSink>();
        return tid;
    }

    void Configure(Address bindAddr,
                   uint32_t pathIdx,
                   Callback<void, uint32_t, uint64_t, Time> onBytes)
    {
        m_bind = bindAddr;
        m_pathIdx = pathIdx;
        m_onBytes = onBytes;
    }

  private:
    void StartApplication() override
    {
        m_listen = Socket::CreateSocket(GetNode(), TcpSocketFactory::GetTypeId());
        m_listen->Bind(m_bind);
        m_listen->Listen();
        m_listen->SetAcceptCallback(
            MakeNullCallback<bool, Ptr<Socket>, const Address&>(),
            MakeCallback(&RealtimeSink::HandleAccept, this));
    }

    void StopApplication() override
    {
        if (m_listen)
        {
            m_listen->Close();
            m_listen = nullptr;
        }
    }

    void HandleAccept(Ptr<Socket> s, const Address&)
    {
        s->SetRecvCallback(MakeCallback(&RealtimeSink::HandleRead, this));
    }

    void HandleRead(Ptr<Socket> s)
    {
        Ptr<Packet> pkt;
        Address from;
        while ((pkt = s->RecvFrom(from)))
        {
            uint32_t n = pkt->GetSize();
            if (n == 0)
            {
                break;
            }
            m_cumBytes += n;
            m_onBytes(m_pathIdx, m_cumBytes, Simulator::Now());
        }
    }

    Ptr<Socket> m_listen;
    Address m_bind;
    uint32_t m_pathIdx = 0;
    uint64_t m_cumBytes = 0; // cumulative bytes delivered since connection
    Callback<void, uint32_t, uint64_t, Time> m_onBytes;
};

// --------------------------------------------------------------------------- //
// Scenario configuration
// --------------------------------------------------------------------------- //

struct PathLink
{
    std::string rate;  // bottleneck data rate, e.g. "8Mbps"
    std::string delay; // one-way propagation delay, e.g. "10ms"
    double crossFrac;  // mean cross-traffic as a fraction of `rate`
};

struct ScenarioConfig
{
    // 4-path no-dominant topology (mirrors configs/four_path.yaml): no single
    // path saturates VMAF, so aggregation is mandatory and a state-aware split
    // beats single-best/proportional. Path 2 is a latency trap (good rate,
    // 40 ms delay).
    std::vector<PathLink> paths = {
        {"3Mbps", "10ms", 0.40},
        {"3Mbps", "15ms", 0.45},
        {"2.5Mbps", "40ms", 0.30},
        {"2Mbps", "20ms", 0.55},
    };
    double fps = 30.0;
    double episodeSeconds = 30.0;
    double appPeriodS = 1.0;
    double deadlineMs = 180.0; // a frame later than this is a real-time "loss"
    double initBitrateKbps = 1500.0;
    double minBitrateKbps = 300.0;
    double maxBitrateKbps = 6000.0;
    double frameSizeJitter = 0.25;
    uint32_t keyframeInterval = 30;
    uint32_t seed = 1;
    uint16_t basePort = 5000;

    // --- Optional non-stationary dynamics (mirrors DynamicsConfig in Python) --//
    // All OFF by default so the static scenario reproduces byte-for-byte: when
    // disabled the dynamics RNG is never drawn and per-path link rates are left
    // at their nominal values. Rates are per-second hazards, applied per frame
    // via 1 - exp(-rate*dt) exactly like the mock.
    bool dynamicsEnabled = false;
    bool churn = false;
    double churnUpRate = 0.10;   // down -> up hazard
    double churnDownRate = 0.05; // up -> down hazard
    uint32_t minActive = 1;      // floor on the live-path count
    bool regime = false;
    double regimeRate = 0.20; // change-points/s/path
    double regimeLo = 0.35;
    double regimeHi = 1.30;
    bool burst = false;
    double burstRate = 0.15;
    double burstIntensity = 0.25;
    double burstDurationS = 0.5;
    double corrRate = 0.05;
    double corrIntensity = 0.30;
    double corrDurationS = 1.0;
    // Correlated-failure groups, ':'-separated groups of ','-separated indices,
    // e.g. "4,5:0,1". Indices >= numPaths are dropped (this topology has 4
    // paths, so a 6-path config's out-of-range group is simply inactive).
    std::string corrGroups = "";
};

// --------------------------------------------------------------------------- //
// RealtimeController — owns the topology and drives the per-frame decision loop
// over the ns3-ai message interface.
// --------------------------------------------------------------------------- //

class RealtimeController
{
  public:
    explicit RealtimeController(const ScenarioConfig& cfg) : m_cfg(cfg) {}

    void Build()
    {
        const uint32_t n = NumPaths();
        NS_ABORT_MSG_IF(n == 0 || n > kMaxPaths, "path count out of range");

        m_client.Create(1);
        m_server.Create(1);
        InternetStackHelper internet;
        internet.Install(m_client);
        internet.Install(m_server);

        m_sources.resize(n);
        m_sinks.resize(n);
        m_clientAddr.resize(n);
        m_pathEnq.assign(n, 0);
        m_shareQ.resize(n);
        m_pathThrEwma.assign(n, 0.0);
        m_pathLossCache.assign(n, 0.0);
        m_curSplit.assign(n, 1.0 / n);
        m_curBitrateKbps = m_cfg.initBitrateKbps;
        m_devs.resize(n);
        m_baseRate.resize(n);
        m_baseRttMs.assign(n, 0.0);
        m_maxBufferBytes.assign(n, 0.0);

        Ipv4AddressHelper addr;
        for (uint32_t i = 0; i < n; ++i)
        {
            PointToPointHelper p2p;
            p2p.SetDeviceAttribute("DataRate", StringValue(m_cfg.paths[i].rate));
            p2p.SetChannelAttribute("Delay", StringValue(m_cfg.paths[i].delay));
            NetDeviceContainer dev = p2p.Install(m_client.Get(0), m_server.Get(0));
            m_devs[i] = dev;
            m_baseRate[i] = DataRate(m_cfg.paths[i].rate);
            // Base RTT ~ 2 x one-way propagation delay (ms); a neutral sRTT
            // placeholder reported for churned-out paths (mirrors the mock).
            m_baseRttMs[i] = 2.0 * Time(m_cfg.paths[i].delay).GetSeconds() * 1000.0;
            // WebRTC-style backlog bound: at most one deadline's worth of bytes at
            // this path's nominal rate. A stalled/reconnecting/throttled path (e.g.
            // mid-churn) drops fresh shares instead of accumulating unbounded stale
            // backlog that bursts out and desyncs delivery once it recovers.
            m_maxBufferBytes[i] =
                (m_cfg.deadlineMs / 1000.0) * m_baseRate[i].GetBitRate() / 8.0;

            // Per-path AQM so loss + queueing delay emerge under load.
            TrafficControlHelper tch;
            tch.SetRootQueueDisc("ns3::FqCoDelQueueDisc");
            tch.Install(dev);

            std::ostringstream net;
            net << "10.1." << (i + 1) << ".0";
            addr.SetBase(net.str().c_str(), "255.255.255.0");
            Ipv4InterfaceContainer ifc = addr.Assign(dev);
            Ipv4Address clientIp = ifc.GetAddress(0); // dev.Get(0) == client side
            m_clientAddr[i] = clientIp;

            // Seed the per-path goodput estimate at nominal link rate (Mbps).
            DataRate dr(m_cfg.paths[i].rate);
            m_pathThrEwma[i] = dr.GetBitRate() / 1e6;

            BuildPathApps(i, clientIp);
            BuildCrossTraffic(i, clientIp, dr);
        }

        Ipv4GlobalRoutingHelper::PopulateRoutingTables();

        // m_fmh is a member so the monitor outlives Simulator::Run().
        m_monitor = m_fmh.InstallAll();
        m_classifier = DynamicCast<Ipv4FlowClassifier>(m_fmh.GetClassifier());

        InitDynamics();
    }

    void AttachInterface(Ns3AiMsgInterfaceImpl<EnvStruct, ActStruct>* msg) { m_msg = msg; }

    void SetSelftest(bool on) { m_selftest = on; }

    void Start(Time warmup)
    {
        m_episodeStartS = warmup.GetSeconds();
        Simulator::Schedule(warmup, &RealtimeController::Decide, this);
    }

    uint32_t NumPaths() const { return static_cast<uint32_t>(m_cfg.paths.size()); }

  private:
    struct ShareEntry
    {
        uint64_t watermark; // cumulative path bytes at which this share is delivered
        uint64_t frameId;
        uint32_t bytes;
    };

    struct FrameRec
    {
        uint32_t sharesRemaining;
        double genTimeS;
        double completeTimeS;
        uint64_t bytes;
        uint64_t droppedBytes; // bytes routed onto churned-out paths (lost)
    };

    uint32_t FramesPerEpisode() const
    {
        return static_cast<uint32_t>(std::lround(m_cfg.fps * m_cfg.episodeSeconds));
    }

    uint32_t FramesPerApp() const
    {
        return std::max<uint32_t>(1, static_cast<uint32_t>(std::lround(m_cfg.fps * m_cfg.appPeriodS)));
    }

    void BuildPathApps(uint32_t i, Ipv4Address clientIp)
    {
        uint16_t port = m_cfg.basePort + i;

        Ptr<RealtimeSink> sink = CreateObject<RealtimeSink>();
        sink->Configure(InetSocketAddress(Ipv4Address::GetAny(), port),
                        i,
                        MakeCallback(&RealtimeController::OnPathBytes, this));
        m_client.Get(0)->AddApplication(sink);
        sink->SetStartTime(Seconds(0.0));
        m_sinks[i] = sink;

        Ptr<RealtimeSource> src = CreateObject<RealtimeSource>();
        src->Configure(InetSocketAddress(clientIp, port), i);
        m_server.Get(0)->AddApplication(src);
        src->SetStartTime(Seconds(0.1)); // after the sink is listening
        m_sources[i] = src;
    }

    // Time-varying UDP cross-traffic competing with the video flow on path i.
    void BuildCrossTraffic(uint32_t i, Ipv4Address clientIp, DataRate linkRate)
    {
        uint16_t port = m_cfg.basePort + 100 + i;

        PacketSinkHelper csink("ns3::UdpSocketFactory",
                               InetSocketAddress(Ipv4Address::GetAny(), port));
        ApplicationContainer ca = csink.Install(m_client.Get(0));
        ca.Start(Seconds(0.0));

        double crossBps = m_cfg.paths[i].crossFrac * linkRate.GetBitRate();
        OnOffHelper onoff("ns3::UdpSocketFactory", InetSocketAddress(clientIp, port));
        onoff.SetAttribute("DataRate", DataRateValue(DataRate(static_cast<uint64_t>(crossBps))));
        onoff.SetAttribute("PacketSize", UintegerValue(1200));
        // Bursty on/off, phase-shifted per path so available bandwidth varies and
        // paths peak at different times.
        std::ostringstream on, off;
        on << "ns3::ExponentialRandomVariable[Mean=" << (0.6 + 0.2 * i) << "]";
        off << "ns3::ExponentialRandomVariable[Mean=" << (0.8 + 0.3 * i) << "]";
        onoff.SetAttribute("OnTime", StringValue(on.str()));
        onoff.SetAttribute("OffTime", StringValue(off.str()));
        ApplicationContainer co = onoff.Install(m_server.Get(0));
        co.Start(Seconds(0.2));
        m_crossApps.Add(co);
    }

    // -- non-stationary dynamics ------------------------------------------- //
    // Mirror MockRealtimeDataPlane's churn/regime/burst/correlated-failure
    // machines (src/ns3env/dataplane.py). Everything is a no-op when
    // `dynamicsEnabled` is false, so the static scenario is byte-identical and
    // the dynamics RNG is never drawn.

    bool PathLive(uint32_t i) const { return m_active[i] != 0; }

    // Parse "4,5:0,1" into index sets (dropping out-of-range members: this
    // topology has NumPaths() paths, so a larger config's group is inactive).
    void InitDynamics()
    {
        const uint32_t n = NumPaths();
        m_active.assign(n, 1);
        m_regimeMult.assign(n, 1.0);
        m_burstUntil.assign(n, -1.0);
        m_corrMembers.clear();
        m_corrUntil.clear();
        if (!m_cfg.dynamicsEnabled)
        {
            return;
        }
        std::stringstream gs(m_cfg.corrGroups);
        std::string grp;
        while (std::getline(gs, grp, ':'))
        {
            std::set<uint32_t> members;
            std::stringstream is(grp);
            std::string idx;
            while (std::getline(is, idx, ','))
            {
                if (idx.empty()) continue;
                long v = std::strtol(idx.c_str(), nullptr, 10);
                if (v >= 0 && v < static_cast<long>(n)) members.insert(static_cast<uint32_t>(v));
            }
            if (!members.empty()) m_corrMembers.push_back(members);
        }
        ResetDynamicsState();
    }

    // Fresh-episode dynamic state: all paths live, regime multipliers resampled,
    // burst/correlation timers cleared. Continues the RNG stream across episodes
    // (like the mock's reset with seed=None) so runs stay deterministic per seed.
    void ResetDynamicsState()
    {
        const uint32_t n = NumPaths();
        std::fill(m_active.begin(), m_active.end(), 1);
        std::fill(m_burstUntil.begin(), m_burstUntil.end(), -1.0);
        std::fill(m_corrUntil.begin(), m_corrUntil.end(), -1.0);
        m_corrUntil.assign(m_corrMembers.size(), -1.0);
        for (uint32_t i = 0; i < n; ++i)
        {
            m_regimeMult[i] = m_cfg.regime
                                  ? m_dynRng->GetValue(m_cfg.regimeLo, m_cfg.regimeHi)
                                  : 1.0;
        }
        for (uint32_t i = 0; i < n; ++i) ApplyPathRate(i);
    }

    // Current dynamic capacity multiplier for path i (regime x burst x corr),
    // 1.0 when disabled. Mirrors MockRealtimeDataPlane._cap_mult.
    double CapMult(uint32_t i) const
    {
        if (!m_cfg.dynamicsEnabled) return 1.0;
        double m = m_regimeMult[i];
        double now = Simulator::Now().GetSeconds();
        if (m_burstUntil[i] > now) m *= m_cfg.burstIntensity;
        for (size_t g = 0; g < m_corrMembers.size(); ++g)
        {
            if (m_corrUntil[g] > now && m_corrMembers[g].count(i)) m *= m_cfg.corrIntensity;
        }
        return m;
    }

    // Push the current effective capacity onto the NS-3 bottleneck. A churned-out
    // path collapses to ~0 so its throughput/RTT reflect the outage; its bytes
    // are additionally dropped at the app layer in GenerateFrame.
    void ApplyPathRate(uint32_t i)
    {
        if (!m_cfg.dynamicsEnabled) return;
        double mult = PathLive(i) ? CapMult(i) : 1e-4;
        uint64_t bps =
            static_cast<uint64_t>(std::max(1e3, m_baseRate[i].GetBitRate() * mult));
        DataRateValue drv{DataRate(bps)};
        m_devs[i].Get(0)->SetAttribute("DataRate", drv);
        m_devs[i].Get(1)->SetAttribute("DataRate", drv);
    }

    // Step the machines one frame. Event probabilities convert per-second hazards
    // over one frame via 1 - exp(-rate*dt); draw order (regime, burst, corr,
    // churn) matches the mock for parity of behavior.
    void AdvanceDynamics()
    {
        if (!m_cfg.dynamicsEnabled) return;
        const uint32_t n = NumPaths();
        const double t = Simulator::Now().GetSeconds();
        const double dt = 1.0 / m_cfg.fps;

        if (m_cfg.regime)
        {
            double p = 1.0 - std::exp(-m_cfg.regimeRate * dt);
            for (uint32_t i = 0; i < n; ++i)
                if (m_dynRng->GetValue(0.0, 1.0) < p)
                    m_regimeMult[i] = m_dynRng->GetValue(m_cfg.regimeLo, m_cfg.regimeHi);
        }
        if (m_cfg.burst)
        {
            double p = 1.0 - std::exp(-m_cfg.burstRate * dt);
            for (uint32_t i = 0; i < n; ++i)
                if (m_dynRng->GetValue(0.0, 1.0) < p)
                    m_burstUntil[i] = t + m_cfg.burstDurationS;
        }
        if (!m_corrMembers.empty())
        {
            double p = 1.0 - std::exp(-m_cfg.corrRate * dt);
            for (size_t g = 0; g < m_corrMembers.size(); ++g)
                if (m_dynRng->GetValue(0.0, 1.0) < p)
                    m_corrUntil[g] = t + m_cfg.corrDurationS;
        }
        if (m_cfg.churn)
        {
            double pUp = 1.0 - std::exp(-m_cfg.churnUpRate * dt);
            double pDown = 1.0 - std::exp(-m_cfg.churnDownRate * dt);
            std::vector<uint8_t> nxt = m_active;
            for (uint32_t i = 0; i < n; ++i)
            {
                if (m_active[i])
                {
                    if (m_dynRng->GetValue(0.0, 1.0) < pDown) nxt[i] = 0;
                }
                else if (m_dynRng->GetValue(0.0, 1.0) < pUp)
                {
                    nxt[i] = 1;
                }
            }
            // Never let the live set fall below minActive (bring lowest idx up).
            uint32_t live = 0;
            for (auto v : nxt) live += v;
            for (uint32_t i = 0; i < n && live < m_cfg.minActive; ++i)
                if (!nxt[i]) { nxt[i] = 1; ++live; }
            m_active = nxt;
        }
        // Reflect the updated multipliers/liveness on the links for the next frame.
        for (uint32_t i = 0; i < n; ++i) ApplyPathRate(i);
    }

    // Register a whole-frame loss (all shares routed onto dead/empty paths), the
    // C++ analogue of the mock's "frame on dead paths -> deadline miss".
    void RegisterLostFrame(double now)
    {
        (void)now;
        m_appLossEwma = 0.9 * m_appLossEwma + 0.1 * 1.0;
        m_lastLatencyMs = 2.0 * m_cfg.deadlineMs;
        m_lastJitterMs = 0.0;
        m_lastLoss = 1.0;
        m_lastBytes = 0;
    }

    // -- decision loop ------------------------------------------------------ //

    void Decide()
    {
        const double now = Simulator::Now().GetSeconds();
        ExpireLateFrames(now);

        const bool appDue = (m_frameInEpisode % FramesPerApp() == 0);
        if (appDue)
        {
            RefreshNetworkLoss(); // FlowMonitor sweep only on the 1 s cadence (lean per-frame)
        }

        EnvStruct env{};
        FillObservation(env, now, appDue);
        env.done = (m_frameInEpisode >= FramesPerEpisode()) ? 1 : 0;

        ActStruct act{};
        if (m_selftest)
        {
            if (env.done)
            {
                std::cerr << "[selftest ep end] completed=" << m_dbgCompleted
                          << " expired=" << m_dbgExpired
                          << " firstLatMs=" << m_dbgFirstCompletedLatMs
                          << " meanLatMs=" << (m_dbgCompleted > 0 ? m_dbgSumLatMs / m_dbgCompleted : 0.0)
                          << " lossRate=" << (double)m_dbgExpired / (m_dbgCompleted + m_dbgExpired)
                          << "\n";
                Simulator::Stop();
                return;
            }
            act.command = ACT_STEP;
            act.targetBitrateKbps = m_cfg.initBitrateKbps;
            for (uint32_t i = 0; i < NumPaths(); ++i)
            {
                act.splitRatio[i] = 1.0 / NumPaths(); // even split
            }
        }
        else
        {
            m_msg->CppSendBegin();
            *m_msg->GetCpp2PyStruct() = env;
            m_msg->CppSendEnd();

            m_msg->CppRecvBegin();
            act = *m_msg->GetPy2CppStruct();
            m_msg->CppRecvEnd();
        }

        if (act.command == ACT_TERMINATE)
        {
            Simulator::Stop();
            return;
        }
        if (act.command == ACT_RESET || env.done)
        {
            ResetEpisode(now);
            Simulator::ScheduleNow(&RealtimeController::Decide, this);
            return;
        }

        GenerateFrame(act, now);
        ++m_frameInEpisode;
        // Advance the dynamic state so the next frame's delivery + observation
        // share it (mirrors the mock, which advances after each step_frame).
        AdvanceDynamics();
        Simulator::Schedule(Seconds(1.0 / m_cfg.fps), &RealtimeController::Decide, this);
    }

    // Apply the action and push one frame's bytes, split across the subflows.
    void GenerateFrame(const ActStruct& act, double now)
    {
        const uint32_t n = NumPaths();

        double br = act.targetBitrateKbps;
        if (br < m_cfg.minBitrateKbps) br = m_cfg.minBitrateKbps;
        if (br > m_cfg.maxBitrateKbps) br = m_cfg.maxBitrateKbps;
        m_curBitrateKbps = br;

        // Normalize the split (clamp negatives; fall back to even if degenerate).
        double sum = 0.0;
        for (uint32_t i = 0; i < n; ++i)
        {
            m_curSplit[i] = std::max(0.0, act.splitRatio[i]);
            sum += m_curSplit[i];
        }
        if (sum <= 1e-9)
        {
            for (uint32_t i = 0; i < n; ++i) m_curSplit[i] = 1.0 / n;
        }
        else
        {
            for (uint32_t i = 0; i < n; ++i) m_curSplit[i] /= sum;
        }

        // Frame size in bytes: bitrate per frame, with I-frame burst and jitter.
        double baseBytes = (br * 1000.0 / 8.0) / m_cfg.fps;
        double kf = (m_cfg.keyframeInterval > 0 && m_frameTotal % m_cfg.keyframeInterval == 0)
                        ? 2.5
                        : 1.0;
        double jit = 1.0 + m_cfg.frameSizeJitter * (2.0 * m_uniform->GetValue() - 1.0);
        uint32_t frameBytes = static_cast<uint32_t>(std::max<long>(1L, std::lround(baseBytes * kf * jit)));

        // Split into per-path shares; fix rounding drift on the largest share.
        std::vector<uint32_t> shares(n, 0);
        uint32_t assigned = 0;
        uint32_t largest = 0;
        for (uint32_t i = 0; i < n; ++i)
        {
            shares[i] = static_cast<uint32_t>(std::lround(frameBytes * m_curSplit[i]));
            assigned += shares[i];
            if (m_curSplit[i] > m_curSplit[largest]) largest = i;
        }
        // Reconcile so the shares sum exactly to frameBytes.
        if (assigned < frameBytes) shares[largest] += (frameBytes - assigned);
        else if (assigned > frameBytes)
        {
            uint32_t excess = assigned - frameBytes;
            shares[largest] = (shares[largest] > excess) ? shares[largest] - excess : 0;
        }

        const uint64_t fId = m_frameTotal++;
        uint32_t activeShares = 0;
        uint64_t droppedBytes = 0;
        for (uint32_t i = 0; i < n; ++i)
        {
            if (shares[i] == 0) continue;
            if (m_cfg.dynamicsEnabled && !PathLive(i))
            {
                // Bytes routed onto a churned-out path never arrive: count them
                // as loss (the penalty that teaches respect for the mask).
                droppedBytes += shares[i];
                m_pathLossCache[i] = 1.0;
                continue;
            }
            if (m_sources[i]->GetBufferOcc() > m_maxBufferBytes[i])
            {
                // Path already has more than one deadline's worth of unsent app
                // backlog (e.g. mid-churn/regime stall or still reconnecting after
                // an episode reset): drop this frame's share instead of piling on
                // more stale bytes that would only arrive late anyway.
                droppedBytes += shares[i];
                m_pathLossCache[i] = 1.0;
                continue;
            }
            ++activeShares;
            m_pathEnq[i] += shares[i];
            m_shareQ[i].push_back(ShareEntry{m_pathEnq[i], fId, shares[i]});
            m_sources[i]->Enqueue(shares[i]);
        }
        if (activeShares == 0)
        {
            // Whole frame routed onto dead (or empty) paths: an immediate loss.
            RegisterLostFrame(now);
            return;
        }

        m_frames[fId] = FrameRec{activeShares, now, now, frameBytes, droppedBytes};
        m_pendingFrames.push_back(fId);
        if (m_selftest && fId < 30)
        {
            std::ostringstream ss;
            for (uint32_t i = 0; i < n; ++i)
                ss << (PathLive(i) ? "L" : "D") << shares[i] << " ";
            std::cerr << "[gen] fid=" << fId << " t=" << now
                      << " bytes=" << frameBytes << " active=" << activeShares
                      << " paths:[" << ss.str() << "]\n";
        }
    }

    // Sink reports cumulative delivered bytes on path `pathIdx`; resolve any
    // frame shares whose watermark is now reached.
    void OnPathBytes(uint32_t pathIdx, uint64_t cumDelivered, Time when)
    {
        auto& q = m_shareQ[pathIdx];
        const double t = when.GetSeconds();
        while (!q.empty() && q.front().watermark <= cumDelivered)
        {
            ShareEntry e = q.front();
            q.pop_front();
            auto it = m_frames.find(e.frameId);
            if (it == m_frames.end())
            {
                continue; // frame already completed or expired
            }
            FrameRec& fr = it->second;
            fr.completeTimeS = std::max(fr.completeTimeS, t);

            // Per-path realized goodput for this share.
            double shareDur = std::max(t - fr.genTimeS, 1e-6);
            double gp = (e.bytes * 8.0) / (shareDur * 1e6);
            m_pathThrEwma[pathIdx] = 0.6 * m_pathThrEwma[pathIdx] + 0.4 * gp;

            if (--fr.sharesRemaining == 0)
            {
                CompleteFrame(it);
            }
        }
    }

    void CompleteFrame(std::unordered_map<uint64_t, FrameRec>::iterator it)
    {
        const FrameRec& fr = it->second;
        double latencyMs = (fr.completeTimeS - fr.genTimeS) * 1000.0;
        bool late = latencyMs > m_cfg.deadlineMs;

        double jitterMs = (m_prevLatencyMs >= 0.0) ? std::abs(latencyMs - m_prevLatencyMs) : 0.0;
        m_prevLatencyMs = latencyMs;

        // Partial loss from bytes shed onto churned-out paths (mirrors the mock's
        // dropped_frac): app loss = max(deadline-miss, dropped fraction).
        double droppedFrac = (fr.bytes > 0)
                                 ? static_cast<double>(fr.droppedBytes) / static_cast<double>(fr.bytes)
                                 : 0.0;
        double frameLoss = std::min(1.0, std::max(late ? 1.0 : 0.0, droppedFrac));
        uint64_t delivered = late ? 0 : (fr.bytes - fr.droppedBytes);

        double goodputMbps = (fr.bytes * 8.0) / (std::max(latencyMs / 1000.0, 1e-6) * 1e6);

        // Episode EWMAs feeding the aggregate observation.
        m_jitterEwma = 0.7 * m_jitterEwma + 0.3 * jitterMs;
        m_appLossEwma = 0.9 * m_appLossEwma + 0.1 * frameLoss;
        m_thrEwma = 0.7 * m_thrEwma + 0.3 * goodputMbps;
        m_rttEwma = (m_rttEwma <= 0.0) ? latencyMs : 0.8 * m_rttEwma + 0.2 * latencyMs;

        // Most-recently-completed frame result.
        m_lastLatencyMs = latencyMs;
        m_lastJitterMs = jitterMs;
        m_lastLoss = frameLoss;
        m_lastBytes = static_cast<uint32_t>(delivered);

        if (m_selftest)
        {
            ++m_dbgCompleted;
            m_dbgSumLatMs += latencyMs;
            if (m_dbgFirstCompletedLatMs < 0.0) m_dbgFirstCompletedLatMs = latencyMs;
            if (m_dbgCompleted + m_dbgExpired <= 30)
                std::cerr << "[frame] COMPLETE fid=" << fr.genTimeS
                          << " latMs=" << latencyMs << " loss=" << frameLoss << "\n";
        }
        m_frames.erase(it);
    }

    // Drop frames that blew their deadline without fully arriving: count them as
    // real-time losses. Their straggler shares are skipped on later delivery.
    void ExpireLateFrames(double now)
    {
        const double deadlineS = m_cfg.deadlineMs / 1000.0;
        while (!m_pendingFrames.empty())
        {
            uint64_t fId = m_pendingFrames.front();
            auto it = m_frames.find(fId);
            if (it == m_frames.end())
            {
                m_pendingFrames.pop_front(); // already completed
                continue;
            }
            if (it->second.genTimeS + deadlineS >= now)
            {
                break; // ordered by genTime: nothing older is late yet
            }
            // Late: register as a lost frame.
            m_pendingFrames.pop_front();
            double elapsedMs = (now - it->second.genTimeS) * 1000.0;
            m_appLossEwma = 0.9 * m_appLossEwma + 0.1 * 1.0;
            m_lastLatencyMs = elapsedMs;
            m_lastJitterMs = 0.0;
            m_lastLoss = 1.0;
            m_lastBytes = 0;
            if (m_selftest) {
                ++m_dbgExpired;
                if (m_dbgCompleted + m_dbgExpired <= 30)
                    std::cerr << "[frame] EXPIRE fid=" << it->second.genTimeS
                              << " elapsedMs=" << elapsedMs
                              << " sharesLeft=" << it->second.sharesRemaining << "\n";
            }
            m_frames.erase(it);
        }
    }

    void FillObservation(EnvStruct& env, double now, bool appDue)
    {
        const uint32_t n = NumPaths();
        env.numPaths = n;
        env.clockS = now;
        env.appDecisionDue = appDue ? 1 : 0;

        env.currentBitrateKbps = m_curBitrateKbps;
        env.jitterMs = m_jitterEwma;
        env.loss = m_appLossEwma;
        env.throughputMbps = m_thrEwma;

        // Aggregate RTT: split-weighted per-path sRTT (fall back to frame EWMA).
        double rttW = 0.0;
        double wsum = 0.0;
        for (uint32_t i = 0; i < n; ++i)
        {
            bool live = !m_cfg.dynamicsEnabled || PathLive(i);
            env.pathActive[i] = live ? 1 : 0;
            double s;
            if (live)
            {
                s = m_sources[i]->GetSrttMs();
                env.cwnd[i] = m_sources[i]->GetCwnd();
                env.bufferOcc[i] = m_sources[i]->GetBufferOcc();
                env.pathThroughputMbps[i] = m_pathThrEwma[i];
                env.pathLoss[i] = m_pathLossCache[i];
            }
            else
            {
                // Churned-out path reports dead state; the mask lets the policy
                // exclude it. Neutral base RTT placeholder (masked downstream).
                s = m_baseRttMs[i];
                env.cwnd[i] = 0.0;
                env.bufferOcc[i] = 0.0;
                env.pathThroughputMbps[i] = 0.0;
                env.pathLoss[i] = 1.0;
            }
            env.srttMs[i] = s;
            rttW += m_curSplit[i] * s;
            wsum += m_curSplit[i];
        }
        env.rttMs = (wsum > 0.0 && rttW > 0.0) ? rttW / wsum : m_rttEwma;

        env.lastLatencyMs = m_lastLatencyMs;
        env.lastJitterMs = m_lastJitterMs;
        env.lastLoss = m_lastLoss;
        env.lastBytes = m_lastBytes;
    }

    // Per-path network loss from FlowMonitor: lost / (tx + lost) for the
    // server->client video flow on path i. Sampled on the app cadence only.
    void RefreshNetworkLoss()
    {
        if (!m_monitor || !m_classifier)
        {
            return;
        }
        m_monitor->CheckForLostPackets();
        auto stats = m_monitor->GetFlowStats();
        for (uint32_t i = 0; i < NumPaths(); ++i)
        {
            m_pathLossCache[i] = 0.0;
        }
        for (const auto& kv : stats)
        {
            Ipv4FlowClassifier::FiveTuple ft = m_classifier->FindFlow(kv.first);
            if (ft.protocol != 6)
            {
                continue; // TCP video flow only
            }
            for (uint32_t i = 0; i < NumPaths(); ++i)
            {
                if (ft.destinationAddress == m_clientAddr[i])
                {
                    uint64_t tx = kv.second.txPackets;
                    uint64_t lost = kv.second.lostPackets;
                    uint64_t denom = tx + lost;
                    m_pathLossCache[i] =
                        (denom == 0) ? 0.0
                                     : std::min(1.0, static_cast<double>(lost) / denom);
                }
            }
        }
    }

    void ResetEpisode(double now)
    {
        // Keep the network (and cumulative byte counters / share queues) warm so
        // congestion evolves across episodes; only reset episode-scoped state.
        // The per-path app backlog (RealtimeSource::m_pending) is bounded
        // continuously by GenerateFrame's deadline-based cap (see
        // m_maxBufferBytes), so it can no longer grow into the multi-MB
        // cross-episode backlog that used to stall delivery for an entire
        // episode — no explicit flush/socket-reset needed here.
        m_frames.clear();
        m_pendingFrames.clear();
        m_frameInEpisode = 0;
        m_episodeStartS = now;
        m_prevLatencyMs = -1.0;
        m_jitterEwma = 0.0;
        m_appLossEwma = 0.0;
        m_thrEwma = 0.0;
        m_rttEwma = 0.0;
        m_lastLatencyMs = 0.0;
        m_lastJitterMs = 0.0;
        m_lastLoss = 0.0;
        m_lastBytes = 0;

        // Fresh episode: all paths live again, regime multipliers resampled.
        if (m_cfg.dynamicsEnabled)
        {
            ResetDynamicsState();
        }
    }

    ScenarioConfig m_cfg;
    NodeContainer m_client;
    NodeContainer m_server;
    std::vector<Ptr<RealtimeSource>> m_sources;
    std::vector<Ptr<RealtimeSink>> m_sinks;
    std::vector<Ipv4Address> m_clientAddr;
    ApplicationContainer m_crossApps;

    // Per-path link handles + baselines for the dynamics machinery.
    std::vector<NetDeviceContainer> m_devs;
    std::vector<DataRate> m_baseRate;
    std::vector<double> m_baseRttMs;
    std::vector<double> m_maxBufferBytes; // per-path app backlog cap (bytes)

    // Non-stationary dynamics state (mirrors MockRealtimeDataPlane).
    std::vector<uint8_t> m_active;             // per-path liveness (churn)
    std::vector<double> m_regimeMult;          // per-path regime multiplier
    std::vector<double> m_burstUntil;          // per-path burst end time (s)
    std::vector<std::set<uint32_t>> m_corrMembers; // correlated-failure groups
    std::vector<double> m_corrUntil;           // per-group failure end time (s)

    // Frame delivery bookkeeping.
    std::vector<uint64_t> m_pathEnq;             // cumulative bytes pushed per path
    std::vector<std::deque<ShareEntry>> m_shareQ; // pending shares per path (FIFO)
    std::unordered_map<uint64_t, FrameRec> m_frames;
    std::deque<uint64_t> m_pendingFrames;        // frame ids in generation order

    // Episode-scoped aggregates.
    std::vector<double> m_pathThrEwma;
    std::vector<double> m_pathLossCache;
    std::vector<double> m_curSplit;
    double m_curBitrateKbps = 0.0;
    double m_prevLatencyMs = -1.0;
    double m_jitterEwma = 0.0;
    double m_appLossEwma = 0.0;
    double m_thrEwma = 0.0;
    double m_rttEwma = 0.0;
    double m_lastLatencyMs = 0.0;
    double m_lastJitterMs = 0.0;
    double m_lastLoss = 0.0;
    uint32_t m_lastBytes = 0;

    uint64_t m_frameTotal = 0;     // monotonic frame id (process lifetime)
    uint32_t m_frameInEpisode = 0; // resets each episode
    double m_episodeStartS = 0.0;

    // Diagnostic counters (selftest only).
    uint32_t m_dbgCompleted = 0;
    uint32_t m_dbgExpired = 0;
    double m_dbgFirstCompletedLatMs = -1.0;
    double m_dbgSumLatMs = 0.0;

    FlowMonitorHelper m_fmh;
    Ptr<FlowMonitor> m_monitor;
    Ptr<Ipv4FlowClassifier> m_classifier;
    Ns3AiMsgInterfaceImpl<EnvStruct, ActStruct>* m_msg = nullptr;
    Ptr<UniformRandomVariable> m_uniform = CreateObject<UniformRandomVariable>();
    // Dedicated RNG stream for dynamics so frame-size jitter is unperturbed.
    Ptr<UniformRandomVariable> m_dynRng = CreateObject<UniformRandomVariable>();
    bool m_selftest = false;
};

// --------------------------------------------------------------------------- //

// Parse a topology string ("rate,delay,cross:rate,delay,cross:...") from the CLI
// into PathLink entries. Mirrors _encode_topology in src/ns3env/dataplane.py.
static std::vector<PathLink>
ParsePaths(const std::string& s)
{
    std::vector<PathLink> out;
    std::stringstream ps(s);
    std::string tok;
    while (std::getline(ps, tok, ':'))
    {
        if (tok.empty()) continue;
        std::stringstream fs(tok);
        std::string rate, delay, cross;
        std::getline(fs, rate, ',');
        std::getline(fs, delay, ',');
        std::getline(fs, cross, ',');
        PathLink pl;
        pl.rate = rate;
        pl.delay = delay;
        pl.crossFrac = cross.empty() ? 0.4 : std::strtod(cross.c_str(), nullptr);
        out.push_back(pl);
    }
    return out;
}

int
main(int argc, char* argv[])
{
    ScenarioConfig cfg;
    bool selftest = false;
    std::string topology = ""; // empty => keep the hardcoded default path list

    CommandLine cmd;
    cmd.AddValue("fps", "Video frames per second", cfg.fps);
    cmd.AddValue("episodeSeconds", "Episode horizon (sim seconds)", cfg.episodeSeconds);
    cmd.AddValue("appPeriodS", "App-agent decision period (s)", cfg.appPeriodS);
    cmd.AddValue("deadlineMs", "Frame deadline; later frames count as loss", cfg.deadlineMs);
    cmd.AddValue("initBitrateKbps", "Initial encoder target (kbps)", cfg.initBitrateKbps);
    cmd.AddValue("minBitrateKbps", "Minimum encoder target (kbps)", cfg.minBitrateKbps);
    cmd.AddValue("maxBitrateKbps", "Maximum encoder target (kbps)", cfg.maxBitrateKbps);
    cmd.AddValue("seed", "RNG seed", cfg.seed);
    cmd.AddValue("selftest", "Run a self-contained even-split episode without the bridge",
                 selftest);
    cmd.AddValue("paths", "Topology: rate,delay,cross:... (empty => built-in default)", topology);
    // Non-stationary dynamics (default OFF -> static scenario unchanged).
    cmd.AddValue("dynamicsEnabled", "Enable non-stationary path dynamics", cfg.dynamicsEnabled);
    cmd.AddValue("churn", "Enable path churn (appear/disappear)", cfg.churn);
    cmd.AddValue("churnUpRate", "Churn down->up hazard (per s)", cfg.churnUpRate);
    cmd.AddValue("churnDownRate", "Churn up->down hazard (per s)", cfg.churnDownRate);
    cmd.AddValue("minActive", "Floor on the number of live paths", cfg.minActive);
    cmd.AddValue("regime", "Enable regime shifts (best-path swaps)", cfg.regime);
    cmd.AddValue("regimeRate", "Regime change-points (per s per path)", cfg.regimeRate);
    cmd.AddValue("regimeLo", "Regime capacity multiplier lower bound", cfg.regimeLo);
    cmd.AddValue("regimeHi", "Regime capacity multiplier upper bound", cfg.regimeHi);
    cmd.AddValue("burst", "Enable congestion bursts", cfg.burst);
    cmd.AddValue("burstRate", "Bursts (per s per path)", cfg.burstRate);
    cmd.AddValue("burstIntensity", "Capacity multiplier while bursting", cfg.burstIntensity);
    cmd.AddValue("burstDurationS", "Burst duration (s)", cfg.burstDurationS);
    cmd.AddValue("corrRate", "Correlated-failure events (per s per group)", cfg.corrRate);
    cmd.AddValue("corrIntensity", "Capacity multiplier for a failed group", cfg.corrIntensity);
    cmd.AddValue("corrDurationS", "Correlated-failure duration (s)", cfg.corrDurationS);
    cmd.AddValue("corrGroups", "Correlated groups, e.g. \"4,5:0,1\"", cfg.corrGroups);
    cmd.Parse(argc, argv);

    if (!topology.empty())
    {
        std::vector<PathLink> parsed = ParsePaths(topology);
        if (!parsed.empty())
        {
            cfg.paths = parsed;
        }
    }

    RngSeedManager::SetSeed(cfg.seed);

    RealtimeController controller(cfg);
    controller.Build();
    controller.SetSelftest(selftest);

    if (!selftest)
    {
        // ns3-ai struct-based message interface. Python is the memory creator,
        // so C++ passes isCreator=false. handleFinish=true makes the destructor
        // signal Python when the process ends.
        Ns3AiMsgInterface* interface = Ns3AiMsgInterface::Get();
        interface->SetIsMemoryCreator(false);
        interface->SetUseVector(false);
        interface->SetHandleFinish(true);
        Ns3AiMsgInterfaceImpl<EnvStruct, ActStruct>* msg =
            interface->GetInterface<EnvStruct, ActStruct>();
        controller.AttachInterface(msg);
    }

    // Warm up: let the per-path TCP connections establish before the first frame.
    controller.Start(Seconds(1.0));

    // Safety stop in case the bridge desyncs. The process normally runs until
    // Python sends ACT_TERMINATE; the simulation is continuing across episodes.
    Simulator::Stop(Seconds(1e9));
    Simulator::Run();
    Simulator::Destroy();
    return 0;
}
