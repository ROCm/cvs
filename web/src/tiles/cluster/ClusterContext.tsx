import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useRef,
  useState,
  type ReactNode,
} from "react";
import {
  getLatestMetrics,
  getProbeStatus,
  listClusterNodes,
  streamClusterMetrics,
  type ClusterNodes,
  type MetricsSnapshot,
} from "@/shared/api";

const PROBE_POLL_MS = 2000;

interface ClusterState {
  nodes: ClusterNodes | null;
  nodesError: string | null;
  snapshot: MetricsSnapshot | null;
  live: boolean;
  setLive: (v: boolean) => void;
  connected: boolean;
  refreshing: boolean;
  reload: () => void;
  probeReady: boolean;
  probeReachable: number;
  probeTotal: number;
}

const Ctx = createContext<ClusterState | null>(null);

// useCluster exposes the shared live snapshot + node grid to every cluster page,
// so the GPU/NIC/Nodes tabs all read one WebSocket stream and one node fetch.
export function useCluster(): ClusterState {
  const v = useContext(Ctx);
  if (!v) throw new Error("useCluster must be used within ClusterProvider");
  return v;
}

export function ClusterProvider({ children }: { children: ReactNode }) {
  const [nodes, setNodes] = useState<ClusterNodes | null>(null);
  const [nodesError, setNodesError] = useState<string | null>(null);
  const [snapshot, setSnapshot] = useState<MetricsSnapshot | null>(null);

  // Always start stopped; the user opts in by clicking "Start". Persisting
  // the live flag across reloads / container restarts caused the WebSocket to
  // fire before the SSH pool was ready and before the user was even on the
  // cluster page (now that ClusterProvider lives at the app level).
  const [live, setLiveState] = useState(false);
  const setLive = useCallback((v: boolean) => {
    setLiveState(v);
  }, []);

  const [connected, setConnected] = useState(false);
  const [refreshing, setRefreshing] = useState(false);

  // Probe-status polling: fires every 2 s until the SSH pool signals ready.
  const [probeReady, setProbeReady] = useState(false);
  const [probeReachable, setProbeReachable] = useState(0);
  const [probeTotal, setProbeTotal] = useState(0);
  const probeTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  const reload = useCallback(async () => {
    setRefreshing(true);
    try {
      setNodes(await listClusterNodes());
      setNodesError(null);
    } catch (e) {
      setNodesError(String(e));
    } finally {
      setRefreshing(false);
    }
  }, []);

  useEffect(() => {
    void reload();
    // Paint any cached snapshot immediately; the WS also sends latest on connect.
    getLatestMetrics()
      .then(setSnapshot)
      .catch(() => undefined);
  }, [reload]);

  // Poll GET /api/v1/inventory/probe-status until the pool is ready.
  useEffect(() => {
    let stopped = false;

    const poll = () => {
      if (stopped) return;
      getProbeStatus()
        .then((st) => {
          setProbeReachable(st.reachable);
          setProbeTotal(st.total);
          if (st.ready) {
            setProbeReady(true);
            return; // stop polling
          }
          probeTimerRef.current = setTimeout(poll, PROBE_POLL_MS);
        })
        .catch(() => {
          probeTimerRef.current = setTimeout(poll, PROBE_POLL_MS);
        });
    };

    poll();
    return () => {
      stopped = true;
      if (probeTimerRef.current !== null) {
        clearTimeout(probeTimerRef.current);
      }
    };
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // When monitoring starts, pre-warm the three on-demand server-side caches
  // (GPU SW, NIC SW, Logs) so they are ready by the time the user navigates to
  // those tabs. Responses are discarded — zero client memory impact.
  useEffect(() => {
    if (!live) return;
    const warm = [
      "/api/v1/clustermon/software/gpu",
      "/api/v1/clustermon/software/nic/devlink",
      "/api/v1/clustermon/logs/dmesg",
    ];
    for (const url of warm) {
      fetch(url).catch(() => undefined);
    }
  }, [live]);

  // One shared subscription to the server-side metrics broadcast. The server
  // owns the poll cadence (default 60s) and pushes each GPU+NIC snapshot; we
  // also refresh the node grid on each frame so reachability stays current.
  useEffect(() => {
    if (!live) {
      setConnected(false);
      return;
    }
    const close = streamClusterMetrics(
      (snap) => {
        setConnected(true);
        setSnapshot(snap);
        void reload();
      },
      () => setConnected(false),
    );
    return close;
  }, [live, reload]);

  return (
    <Ctx.Provider
      value={{ nodes, nodesError, snapshot, live, setLive, connected, refreshing, reload, probeReady, probeReachable, probeTotal }}
    >
      {children}
    </Ctx.Provider>
  );
}
