import { useEffect, useRef, useState } from "react";
import {
  Cpu,
  Loader2,
  RadioTower,
  Save,
  Server,
  Trash2,
  Upload,
  Wifi,
  WifiOff,
} from "lucide-react";
import {
  deleteInventory,
  listKeys,
  probeInventory,
  saveInventory,
  uploadSshKey,
  type AuthMethod,
} from "@/shared/api";
import { useInventory } from "@/shared/inventory";

const inputClass =
  "w-full rounded-lg border border-input bg-card px-3 py-2 focus:border-primary focus:outline-none focus:ring-2 focus:ring-ring";

export default function InventoryPage() {
  const { status, refresh } = useInventory();
  const existing = status?.inventory;

  const [nodeText, setNodeText] = useState("");
  const [username, setUsername] = useState("");
  const [authMethod, setAuthMethod] = useState<AuthMethod>("key");
  const [keyName, setKeyName] = useState("");
  const [availableKeys, setAvailableKeys] = useState<string[]>([]);

  type Banner = { type: "success" | "error"; text: string } | null;
  const [banner, setBanner] = useState<Banner>(null);
  const [saving, setSaving] = useState(false);
  const [probing, setProbing] = useState(false);
  const [deleting, setDeleting] = useState(false);

  // Prevent double-submit: useRef blocks synchronously before React re-renders.
  const submittingRef = useRef(false);

  // Hide node cards until the page's own refresh() resolves. This prevents
  // stale data from the InventoryProvider's earlier fetch (which happened when
  // the tab first opened) from flashing before we have up-to-date results.
  // useEffect fires after paint, so we start false and flip to true only after
  // the round-trip completes.
  const [pageReady, setPageReady] = useState(false);
  useEffect(() => {
    void refresh().finally(() => setPageReady(true));
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []); // intentional: run once on mount only

  // Prefill from any existing inventory + load uploaded keys.
  useEffect(() => {
    if (existing) {
      setNodeText(existing.nodes.join("\n"));
      setUsername(existing.username);
      setAuthMethod(existing.auth_method);
      setKeyName(existing.key_name ?? "");
    }
  }, [existing]);

  useEffect(() => {
    void listKeys().then(setAvailableKeys).catch(() => undefined);
  }, []);

  const handleNodesFile = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = (ev) => setNodeText((ev.target?.result as string) ?? "");
    reader.readAsText(file);
  };

  const handleKeyUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    try {
      const { key_name } = await uploadSshKey(file);
      setKeyName(key_name);
      setAvailableKeys(await listKeys());
      setBanner({ type: "success", text: `Uploaded key '${key_name}'.` });
    } catch (err) {
      setBanner({
        type: "error",
        text: `Key upload failed: ${err instanceof Error ? err.message : "unknown error"}`,
      });
    }
  };

  const parsedNodes = nodeText
    .split("\n")
    .map((l) => l.trim())
    .filter((l) => l && !l.startsWith("#"));

  // Nodes in the inventory that haven't received a probe result yet.
  // Derived from the persisted node list so it survives page refreshes.
  const pendingNodes = (existing?.nodes ?? []).filter(
    (n) => !existing?.statuses?.find((s) => s.host === n && s.checked_at),
  );
  const hasPending = pendingNodes.length > 0;

  // Auto-poll every 2 s while there are unresolved nodes. Starts automatically
  // on page load if nodes are still probing (e.g. after a browser refresh).
  useEffect(() => {
    if (!hasPending) return;
    const id = setInterval(() => { void refresh(); }, 2000);
    return () => clearInterval(id);
  }, [hasPending, refresh]);

  const handleSave = async () => {
    if (submittingRef.current) return; // double-submit guard
    if (parsedNodes.length === 0) {
      setBanner({ type: "error", text: "Enter at least one node IP or hostname." });
      return;
    }
    if (!username.trim()) {
      setBanner({ type: "error", text: "SSH username is required." });
      return;
    }
    if (authMethod === "key" && !keyName) {
      setBanner({ type: "error", text: "Upload (or select) an SSH private key." });
      return;
    }

    submittingRef.current = true;
    setSaving(true);
    setBanner(null);
    try {
      await saveInventory({
        nodes: parsedNodes,
        username: username.trim(),
        auth_method: authMethod,
        key_name: authMethod === "key" ? keyName : undefined,
        jump_host: null,
      });
      await refresh();
      setBanner({ type: "success", text: "Inventory saved. Probing new nodes below…" });
    } catch (err) {
      setBanner({
        type: "error",
        text: `Failed to save inventory: ${err instanceof Error ? err.message : "unknown error"}`,
      });
    } finally {
      setSaving(false);
      submittingRef.current = false;
    }
  };

  const handleProbe = async () => {
    setProbing(true);
    setBanner(null);
    try {
      const res = await probeInventory();
      await refresh();
      const reachable = res.inventory?.statuses?.filter((s) => s.reachable).length ?? 0;
      const total = res.inventory?.statuses?.length ?? 0;
      setBanner({ type: "success", text: `Probe complete: ${reachable}/${total} node(s) reachable.` });
    } catch (err) {
      setBanner({
        type: "error",
        text: `Probe failed: ${err instanceof Error ? err.message : "unknown error"}`,
      });
    } finally {
      setProbing(false);
    }
  };

  const handleDeleteInventory = async () => {
    if (!existing) return;
    if (
      !window.confirm(
        "Remove the saved fleet inventory from this server? Uploaded SSH keys are not deleted.",
      )
    ) {
      return;
    }
    setDeleting(true);
    setBanner(null);
    try {
      await deleteInventory();
      setNodeText("");
      setUsername("");
      setAuthMethod("key");
      setKeyName("");
      await refresh();
      setBanner({ type: "success", text: "Inventory cleared. Tiles are gated until you save again." });
    } catch (err) {
      setBanner({
        type: "error",
        text: `Failed to delete inventory: ${err instanceof Error ? err.message : "unknown error"}`,
      });
    } finally {
      setDeleting(false);
    }
  };

  return (
    <div className="mx-auto max-w-6xl space-y-6">
      <div>
        <h1 className="text-2xl font-semibold">Fleet Inventory</h1>
        <p className="mt-1 text-muted-foreground">
          Configure your nodes and SSH access. The tiles unlock once an inventory is saved.
        </p>
      </div>

      {banner && (
        <div
          className={`rounded-lg border p-3 text-sm ${
            banner.type === "success"
              ? "border-green-200 bg-green-50 text-green-800"
              : "border-destructive/30 bg-destructive/10 text-destructive"
          }`}
        >
          {banner.text}
        </div>
      )}

      {/* Configuration: nodes + SSH auth side by side */}
      <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">
        {/* Nodes */}
        <section className="rounded-xl border border-border bg-card p-5 shadow-sm">
          <h2 className="mb-1 flex items-center gap-2 font-semibold">
            <Server className="h-5 w-5" /> Cluster Nodes
          </h2>
          <p className="mb-4 text-sm text-muted-foreground">
            One IP or hostname per line. Lines starting with <code>#</code> are ignored.
          </p>

          <label
            htmlFor="nodes-file"
            className="mb-3 inline-flex cursor-pointer items-center gap-2 rounded-lg bg-primary px-4 py-2 text-sm font-medium text-primary-foreground hover:bg-primary/90"
          >
            <Upload className="h-4 w-4" /> Upload nodes.txt
          </label>
          <input
            id="nodes-file"
            type="file"
            accept=".txt"
            onChange={handleNodesFile}
            className="hidden"
          />

          <textarea
            value={nodeText}
            onChange={(e) => setNodeText(e.target.value)}
            rows={8}
            placeholder={"10.0.0.10\n10.0.0.11\nnode1.cluster.local"}
            className={`${inputClass} font-mono text-sm`}
          />
          <p className="mt-1 text-xs text-muted-foreground">{parsedNodes.length} node(s) detected.</p>
        </section>

        {/* SSH auth */}
        <section className="rounded-xl border border-border bg-card p-5 shadow-sm">
          <h2 className="mb-4 font-semibold">SSH Authentication</h2>

          <div className="mb-4">
            <label className="mb-1 block text-sm font-medium">SSH Username</label>
            <input
              type="text"
              value={username}
              onChange={(e) => setUsername(e.target.value)}
              placeholder="amd"
              className={inputClass}
            />
          </div>

          <div className="mb-4">
            <span className="mb-1 block text-sm font-medium">Authentication Method</span>
            <div className="flex gap-4">
              {(["key", "password"] as AuthMethod[]).map((m) => (
                <label key={m} className="flex cursor-pointer items-center gap-2">
                  <input
                    type="radio"
                    checked={authMethod === m}
                    onChange={() => setAuthMethod(m)}
                  />
                  <span className="capitalize">{m === "key" ? "SSH Key" : "Password"}</span>
                </label>
              ))}
            </div>
          </div>

          {authMethod === "key" && (
            <div className="space-y-3">
              <div>
                <label className="mb-1 block text-sm font-medium">Upload SSH Private Key</label>
                <input
                  type="file"
                  onChange={handleKeyUpload}
                  className={inputClass}
                />
              </div>
              {availableKeys.length > 0 && (
                <div>
                  <label className="mb-1 block text-sm font-medium">Or select an uploaded key</label>
                  <select
                    value={keyName}
                    onChange={(e) => setKeyName(e.target.value)}
                    className={inputClass}
                  >
                    <option value="">-- choose a key --</option>
                    {availableKeys.map((k) => (
                      <option key={k} value={k}>
                        {k}
                      </option>
                    ))}
                  </select>
                </div>
              )}
              {keyName && (
                <p className="text-xs text-green-700">Selected key: {keyName}</p>
              )}
            </div>
          )}

          {authMethod === "password" && (
            <p className="rounded-lg border border-amber-200 bg-amber-50 p-3 text-xs text-amber-800">
              Password auth is accepted for parity, but SSH keys are recommended. The password is
              not stored in the inventory document; key-based auth is the supported path.
            </p>
          )}
        </section>
      </div>

      {/* Action buttons — directly below the config forms */}
      <div className="flex flex-wrap items-center justify-end gap-2">
        {status?.configured && (
          <button
            type="button"
            onClick={() => void handleDeleteInventory()}
            disabled={deleting || saving}
            className="inline-flex items-center gap-2 rounded-lg border border-destructive/40 px-4 py-2.5 text-sm font-medium text-destructive hover:bg-destructive/10 disabled:cursor-not-allowed disabled:opacity-50"
          >
            <Trash2 className="h-4 w-4" />
            {deleting ? "Deleting…" : "Delete inventory"}
          </button>
        )}
        <button
          onClick={() => void handleSave()}
          disabled={saving}
          className="inline-flex items-center gap-2 rounded-lg bg-primary px-6 py-2.5 font-medium text-primary-foreground hover:bg-primary/90 disabled:cursor-not-allowed disabled:opacity-50"
        >
          <Save className="h-4 w-4" />
          {saving ? "Saving…" : "Save Inventory"}
        </button>
      </div>

      {/* Node Connectivity — card grid matching the cluster/nodes style */}
      {existing && pageReady && (
        <section>
          <div className="mb-4 flex items-center justify-between">
            <div>
              <h2 className="font-semibold">Node Connectivity</h2>
              <p className="text-sm text-muted-foreground">
                SSH reachability, GPU type, GPU count, and ROCm version per node.
              </p>
            </div>
            <button
              onClick={() => void handleProbe()}
              disabled={probing}
              className="inline-flex items-center gap-2 rounded-lg border border-input bg-card px-4 py-2 text-sm font-medium hover:bg-muted disabled:cursor-not-allowed disabled:opacity-50"
            >
              <RadioTower className={`h-4 w-4 ${probing ? "animate-pulse" : ""}`} />
              {probing ? "Probing…" : "Probe Nodes"}
            </button>
          </div>

          {existing.nodes && existing.nodes.length > 0 ? (
            <div className="grid grid-cols-1 gap-3 sm:grid-cols-2 lg:grid-cols-3">
              {existing.nodes.map((host) => {
                const s = existing.statuses?.find((st) => st.host === host);
                const isPending = !s?.checked_at;
                return (
                  <div key={host} className="rounded-xl border border-border bg-card p-4">
                    <div className="mb-3 flex items-center justify-between gap-2">
                      <span className="font-mono text-sm font-medium">{host}</span>
                      {isPending ? (
                        <span className="inline-flex items-center gap-1 rounded-full bg-muted px-2 py-0.5 text-xs font-medium text-muted-foreground">
                          <Loader2 className="h-3 w-3 animate-spin" /> Probing…
                        </span>
                      ) : s?.reachable ? (
                        <span className="inline-flex items-center gap-1 rounded-full bg-green-500/10 px-2 py-0.5 text-xs font-medium text-green-600">
                          <Wifi className="h-3 w-3" /> Reachable
                        </span>
                      ) : (
                        <span className="inline-flex items-center gap-1 rounded-full bg-destructive/10 px-2 py-0.5 text-xs font-medium text-destructive">
                          <WifiOff className="h-3 w-3" /> Unreachable
                        </span>
                      )}
                    </div>

                    {isPending ? (
                      <p className="text-sm text-muted-foreground">Connecting…</p>
                    ) : s?.reachable ? (
                      <dl className="space-y-1.5 text-sm">
                        <div className="flex items-center gap-2 text-muted-foreground">
                          <Cpu className="h-4 w-4" />
                          <span className="text-foreground">
                            {s.gpu_type || "Unknown GPU"}
                            {s.gpu_count ? ` ×${s.gpu_count}` : ""}
                          </span>
                        </div>
                        <div className="flex items-center justify-between">
                          <dt className="text-muted-foreground">ROCm</dt>
                          <dd className="font-mono text-xs">{s.rocm_version || "—"}</dd>
                        </div>
                        <div className="flex items-center justify-between">
                          <dt className="text-muted-foreground">Checked</dt>
                          <dd className="text-xs text-muted-foreground">
                            {s.checked_at ? new Date(s.checked_at).toLocaleTimeString() : "—"}
                          </dd>
                        </div>
                      </dl>
                    ) : (
                      <dl className="space-y-1.5 text-sm">
                        <div className="flex items-center justify-between">
                          <dt className="text-destructive">{s?.error || "Probe failed"}</dt>
                        </div>
                        <div className="flex items-center justify-between">
                          <dt className="text-muted-foreground">Checked</dt>
                          <dd className="text-xs text-muted-foreground">
                            {s?.checked_at ? new Date(s.checked_at).toLocaleTimeString() : "—"}
                          </dd>
                        </div>
                      </dl>
                    )}
                  </div>
                );
              })}
            </div>
          ) : (
            <p className="rounded-lg border border-dashed border-border p-4 text-center text-sm text-muted-foreground">
              No nodes configured. Add nodes above and save.
            </p>
          )}
        </section>
      )}
    </div>
  );
}
