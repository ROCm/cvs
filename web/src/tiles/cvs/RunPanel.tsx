import { useCallback, useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { Eye, ExternalLink, Loader2, Pencil, Play, Plus, Trash2 } from "lucide-react";
import {
  createCluster,
  deleteCluster,
  executeTest,
  executionReportUrl,
  getClusterContent,
  isTerminal,
  listClusters,
  updateCluster,
  type Cluster,
} from "@/shared/api";
import { useInventory } from "@/shared/inventory";
import { useExecutionStream } from "./useExecutionStream";

interface RunPanelProps {
  suite: string;
  // Current edited config from the form (null when the suite has no config).
  config: unknown;
  // Unresolved <changeme> placeholders; running is blocked while > 0.
  remaining: number;
}

const STATUS_STYLES: Record<string, string> = {
  queued: "bg-muted text-muted-foreground",
  running: "bg-primary/10 text-primary",
  passed: "bg-green-500/10 text-green-600",
  failed: "bg-destructive/10 text-destructive",
  error: "bg-destructive/10 text-destructive",
  interrupted: "bg-amber-500/10 text-amber-600",
};

export default function RunPanel({ suite, config, remaining }: RunPanelProps) {
  const { status } = useInventory();
  const nodes = status?.inventory?.nodes ?? [];

  const [clusters, setClusters] = useState<Cluster[]>([]);
  const [clusterId, setClusterId] = useState<string>("");
  const [creating, setCreating] = useState(false);
  const [newName, setNewName] = useState("");
  const [selectedNodes, setSelectedNodes] = useState<string[]>([]);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const [executionId, setExecutionId] = useState<string>("");
  const { execution, logs } = useExecutionStream(executionId || null);

  // Expanded cluster_json preview (which cluster + its formatted content).
  const [viewId, setViewId] = useState<string>("");
  const [viewContent, setViewContent] = useState<string>("");

  // Inline cluster edit (which cluster + working name/node set).
  const [editId, setEditId] = useState<string>("");
  const [editName, setEditName] = useState<string>("");
  const [editNodes, setEditNodes] = useState<string[]>([]);

  const refreshClusters = useCallback(async () => {
    const cs = await listClusters();
    setClusters(cs);
    setClusterId((cur) => cur || (cs[0]?.id ?? ""));
  }, []);

  useEffect(() => {
    refreshClusters().catch((e) => setError(String(e)));
  }, [refreshClusters]);

  const toggleNode = (n: string) =>
    setSelectedNodes((prev) =>
      prev.includes(n) ? prev.filter((x) => x !== n) : [...prev, n],
    );

  async function handleCreate() {
    setError(null);
    if (!newName.trim()) {
      setError("Cluster name is required");
      return;
    }
    if (selectedNodes.length === 0) {
      setError("Select at least one node");
      return;
    }
    setBusy(true);
    try {
      const c = await createCluster({ name: newName.trim(), nodes: selectedNodes });
      await refreshClusters();
      setClusterId(c.id);
      setCreating(false);
      setNewName("");
      setSelectedNodes([]);
    } catch (e) {
      setError(String(e));
    } finally {
      setBusy(false);
    }
  }

  async function handleView(id: string) {
    if (viewId === id) {
      setViewId("");
      return;
    }
    setError(null);
    try {
      const raw = await getClusterContent(id);
      let pretty = raw;
      try {
        pretty = JSON.stringify(JSON.parse(raw), null, 2);
      } catch {
        // Not JSON (e.g. YAML) — show as-is.
      }
      setViewContent(pretty);
      setViewId(id);
    } catch (e) {
      setError(String(e));
    }
  }

  function startEdit(c: Cluster) {
    setViewId("");
    setEditId(c.id);
    setEditName(c.name);
    setEditNodes([...c.nodes]);
  }

  const toggleEditNode = (n: string) =>
    setEditNodes((prev) =>
      prev.includes(n) ? prev.filter((x) => x !== n) : [...prev, n],
    );

  async function handleEditSave() {
    setError(null);
    if (!editName.trim()) {
      setError("Cluster name is required");
      return;
    }
    if (editNodes.length === 0) {
      setError("Select at least one node");
      return;
    }
    setBusy(true);
    try {
      await updateCluster(editId, { name: editName.trim(), nodes: editNodes });
      await refreshClusters();
      setEditId("");
    } catch (e) {
      setError(String(e));
    } finally {
      setBusy(false);
    }
  }

  async function handleDelete(id: string) {
    if (!window.confirm("Delete this saved cluster?")) return;
    try {
      await deleteCluster(id);
      if (clusterId === id) setClusterId("");
      await refreshClusters();
    } catch (e) {
      setError(String(e));
    }
  }

  async function handleRun() {
    setError(null);
    if (!clusterId) {
      setError("Select or create a cluster first");
      return;
    }
    setBusy(true);
    try {
      const ex = await executeTest({
        suite,
        cluster_id: clusterId,
        config: config ?? undefined,
      });
      setExecutionId(ex.id);
    } catch (e) {
      setError(String(e));
    } finally {
      setBusy(false);
    }
  }

  const canRun = !!clusterId && remaining === 0 && !busy;

  return (
    <div className="mt-6 border-t border-border pt-5">
      <h2 className="mb-3 text-lg font-semibold">Run</h2>

      {error && (
        <div className="mb-3 rounded border border-destructive/30 bg-destructive/10 p-2 text-sm text-destructive">
          {error}
        </div>
      )}

      {/* Cluster selection */}
      <div className="mb-4 rounded-xl border border-border bg-card p-4">
        <div className="mb-2 flex items-center justify-between">
          <span className="text-sm font-medium">Cluster</span>
          <button
            type="button"
            onClick={() => setCreating((v) => !v)}
            className="inline-flex items-center gap-1 rounded border border-dashed border-border px-2 py-1 text-xs text-muted-foreground hover:border-primary hover:text-primary"
          >
            <Plus className="h-3 w-3" /> {creating ? "Cancel" : "Create new"}
          </button>
        </div>

        {!creating && clusters.length === 0 && (
          <p className="text-sm text-muted-foreground">
            No saved clusters yet — create one from your inventory nodes.
          </p>
        )}

        {!creating && clusters.length > 0 && (
          <div className="space-y-1">
            {clusters.map((c) => (
              <div key={c.id}>
                <div className="flex items-center justify-between gap-2 rounded px-2 py-1 hover:bg-muted">
                  <label className="flex items-center gap-2 text-sm">
                    <input
                      type="radio"
                      name="cluster"
                      checked={clusterId === c.id}
                      onChange={() => setClusterId(c.id)}
                    />
                    <span className="font-medium">{c.name}</span>
                    <span className="text-xs text-muted-foreground">
                      {c.nodes.length} node{c.nodes.length === 1 ? "" : "s"}
                    </span>
                  </label>
                  <span className="flex items-center gap-1">
                    <button
                      type="button"
                      onClick={() => handleView(c.id)}
                      title="View cluster_json"
                      className={`rounded border p-1 ${
                        viewId === c.id
                          ? "border-primary text-primary"
                          : "border-border text-muted-foreground hover:border-primary hover:text-primary"
                      }`}
                    >
                      <Eye className="h-3.5 w-3.5" />
                    </button>
                    <button
                      type="button"
                      onClick={() => (editId === c.id ? setEditId("") : startEdit(c))}
                      title="Edit cluster"
                      className={`rounded border p-1 ${
                        editId === c.id
                          ? "border-primary text-primary"
                          : "border-border text-muted-foreground hover:border-primary hover:text-primary"
                      }`}
                    >
                      <Pencil className="h-3.5 w-3.5" />
                    </button>
                    <button
                      type="button"
                      onClick={() => handleDelete(c.id)}
                      title="Delete cluster"
                      className="rounded border border-border p-1 text-muted-foreground hover:border-destructive hover:text-destructive"
                    >
                      <Trash2 className="h-3.5 w-3.5" />
                    </button>
                  </span>
                </div>
                {viewId === c.id && (
                  <div className="mt-1 mb-2 ml-6">
                    <p className="mb-1 font-mono text-[11px] text-muted-foreground">
                      {c.file_path}
                    </p>
                    <pre className="max-h-72 overflow-auto rounded-lg border border-border bg-muted p-3 font-mono text-xs">
                      {viewContent}
                    </pre>
                  </div>
                )}
                {editId === c.id && (
                  <div className="mb-2 ml-6 mt-1 space-y-3 rounded-lg border border-border p-3">
                    <input
                      value={editName}
                      onChange={(e) => setEditName(e.target.value)}
                      placeholder="Cluster name"
                      className="w-full rounded border border-input bg-background px-2 py-1.5 text-sm"
                    />
                    <div>
                      <p className="mb-1 text-xs font-medium text-muted-foreground">
                        Nodes ({editNodes.length}/{nodes.length})
                      </p>
                      <div className="max-h-40 space-y-1 overflow-auto rounded border border-border p-2">
                        {nodes.map((n) => (
                          <label key={n} className="flex items-center gap-2 text-sm">
                            <input
                              type="checkbox"
                              checked={editNodes.includes(n)}
                              onChange={() => toggleEditNode(n)}
                            />
                            <span className="font-mono">{n}</span>
                          </label>
                        ))}
                      </div>
                      <p className="mt-1 text-[11px] text-muted-foreground">
                        Changing nodes regenerates the cluster_json file.
                      </p>
                    </div>
                    <div className="flex items-center gap-2">
                      <button
                        type="button"
                        onClick={handleEditSave}
                        disabled={busy}
                        className="rounded-lg bg-primary px-3 py-1.5 text-sm font-medium text-primary-foreground hover:bg-primary/90 disabled:opacity-50"
                      >
                        {busy ? "Saving…" : "Save changes"}
                      </button>
                      <button
                        type="button"
                        onClick={() => setEditId("")}
                        className="rounded-lg border border-border px-3 py-1.5 text-sm text-muted-foreground hover:bg-muted"
                      >
                        Cancel
                      </button>
                    </div>
                  </div>
                )}
              </div>
            ))}
          </div>
        )}

        {creating && (
          <div className="space-y-3">
            <input
              value={newName}
              onChange={(e) => setNewName(e.target.value)}
              placeholder="Cluster name"
              className="w-full rounded border border-input bg-background px-2 py-1.5 text-sm"
            />
            <div>
              <p className="mb-1 text-xs font-medium text-muted-foreground">
                Select nodes ({selectedNodes.length}/{nodes.length})
              </p>
              <div className="max-h-40 space-y-1 overflow-auto rounded border border-border p-2">
                {nodes.map((n) => (
                  <label key={n} className="flex items-center gap-2 text-sm">
                    <input
                      type="checkbox"
                      checked={selectedNodes.includes(n)}
                      onChange={() => toggleNode(n)}
                    />
                    <span className="font-mono">{n}</span>
                  </label>
                ))}
              </div>
            </div>
            <button
              type="button"
              onClick={handleCreate}
              disabled={busy}
              className="rounded-lg bg-primary px-3 py-1.5 text-sm font-medium text-primary-foreground hover:bg-primary/90 disabled:opacity-50"
            >
              {busy ? "Creating…" : "Create cluster"}
            </button>
          </div>
        )}
      </div>

      {/* Run button */}
      <div className="flex items-center gap-3">
        <button
          type="button"
          onClick={handleRun}
          disabled={!canRun}
          title={remaining > 0 ? "Resolve required fields first" : !clusterId ? "Select a cluster" : "Run suite"}
          className="inline-flex items-center gap-2 rounded-lg bg-primary px-4 py-2 text-sm font-medium text-primary-foreground hover:bg-primary/90 disabled:cursor-not-allowed disabled:opacity-50"
        >
          {busy ? <Loader2 className="h-4 w-4 animate-spin" /> : <Play className="h-4 w-4" />}
          Run suite
        </button>
        {remaining > 0 && (
          <span className="text-xs text-destructive">
            {remaining} required field{remaining === 1 ? "" : "s"} still to set
          </span>
        )}
      </div>

      {/* Execution status + logs */}
      {execution && (
        <div className="mt-4 rounded-xl border border-border bg-card p-4">
          <div className="mb-2 flex items-center gap-2">
            <span className="text-sm font-medium">Execution</span>
            <span className="font-mono text-xs text-muted-foreground">{execution.id}</span>
            <span
              className={`ml-auto rounded-full px-2 py-0.5 text-xs font-semibold ${
                STATUS_STYLES[execution.status] ?? "bg-muted"
              }`}
            >
              {execution.status}
              {!isTerminal(execution.status) && (
                <Loader2 className="ml-1 inline h-3 w-3 animate-spin" />
              )}
            </span>
          </div>
          {execution.error && (
            <p className="mb-2 text-xs text-destructive">{execution.error}</p>
          )}
          <pre className="max-h-72 overflow-auto rounded-lg border border-border bg-muted p-3 font-mono text-xs">
            {logs || "(waiting for output…)"}
          </pre>
          {execution.has_report && (
            <a
              href={executionReportUrl(execution.id)}
              target="_blank"
              rel="noreferrer"
              className="mt-3 inline-flex items-center gap-1 rounded-lg border border-border px-3 py-1.5 text-sm text-foreground hover:border-primary hover:text-primary"
            >
              <ExternalLink className="h-3.5 w-3.5" /> View HTML report
            </a>
          )}
        </div>
      )}

      <div className="mt-4">
        <Link to="/cvs/history" className="text-sm text-primary hover:underline">
          View execution history →
        </Link>
      </div>
    </div>
  );
}
