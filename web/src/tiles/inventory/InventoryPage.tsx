import { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import {
  CheckCircle,
  RadioTower,
  Save,
  Server,
  Upload,
  XCircle,
} from "lucide-react";
import {
  listKeys,
  probeInventory,
  saveInventory,
  uploadSshKey,
  type AuthMethod,
  type JumpHost,
} from "@/shared/api";
import { useInventory } from "@/shared/inventory";

type Banner = { type: "success" | "error"; text: string } | null;

const inputClass =
  "w-full rounded-lg border border-input bg-card px-3 py-2 focus:border-primary focus:outline-none focus:ring-2 focus:ring-ring";

export default function InventoryPage() {
  const navigate = useNavigate();
  const { status, refresh } = useInventory();
  const existing = status?.inventory;

  const [nodeText, setNodeText] = useState("");
  const [username, setUsername] = useState("");
  const [authMethod, setAuthMethod] = useState<AuthMethod>("key");
  const [keyName, setKeyName] = useState("");
  const [availableKeys, setAvailableKeys] = useState<string[]>([]);

  const [useJump, setUseJump] = useState(false);
  const [jumpHost, setJumpHost] = useState("");
  const [jumpUser, setJumpUser] = useState("");
  const [jumpKeyName, setJumpKeyName] = useState("");

  const [banner, setBanner] = useState<Banner>(null);
  const [saving, setSaving] = useState(false);
  const [probing, setProbing] = useState(false);

  // Prefill from any existing inventory + load uploaded keys.
  useEffect(() => {
    if (existing) {
      setNodeText(existing.nodes.join("\n"));
      setUsername(existing.username);
      setAuthMethod(existing.auth_method);
      setKeyName(existing.key_name ?? "");
      if (existing.jump_host) {
        setUseJump(true);
        setJumpHost(existing.jump_host.host);
        setJumpUser(existing.jump_host.username);
        setJumpKeyName(existing.jump_host.key_name ?? "");
      }
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

  const handleKeyUpload = async (
    e: React.ChangeEvent<HTMLInputElement>,
    target: "node" | "jump",
  ) => {
    const file = e.target.files?.[0];
    if (!file) return;
    try {
      const { key_name } = await uploadSshKey(file);
      if (target === "node") setKeyName(key_name);
      else setJumpKeyName(key_name);
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

  const handleSave = async () => {
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

    const jump: JumpHost | null = useJump
      ? {
          host: jumpHost.trim(),
          username: jumpUser.trim(),
          auth_method: "key",
          key_name: jumpKeyName || undefined,
        }
      : null;

    if (jump && (!jump.host || !jump.username)) {
      setBanner({ type: "error", text: "Jump host requires a host and username." });
      return;
    }

    setSaving(true);
    setBanner(null);
    try {
      await saveInventory({
        nodes: parsedNodes,
        username: username.trim(),
        auth_method: authMethod,
        key_name: authMethod === "key" ? keyName : undefined,
        jump_host: jump,
      });
      await refresh();
      navigate("/");
    } catch (err) {
      setBanner({
        type: "error",
        text: `Failed to save inventory: ${err instanceof Error ? err.message : "unknown error"}`,
      });
      setSaving(false);
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

  return (
    <div className="mx-auto max-w-3xl space-y-6">
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
                onChange={(e) => handleKeyUpload(e, "node")}
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

      {/* Jump host (optional) */}
      <section className="rounded-xl border border-border bg-card p-5 shadow-sm">
        <label className="flex cursor-pointer items-center gap-2 font-semibold">
          <input
            type="checkbox"
            checked={useJump}
            onChange={(e) => setUseJump(e.target.checked)}
          />
          Use a jump host / bastion
        </label>
        {useJump && (
          <div className="mt-4 space-y-3">
            <div>
              <label className="mb-1 block text-sm font-medium">Jump Host</label>
              <input
                type="text"
                value={jumpHost}
                onChange={(e) => setJumpHost(e.target.value)}
                placeholder="bastion.example.com"
                className={inputClass}
              />
            </div>
            <div>
              <label className="mb-1 block text-sm font-medium">Jump Host Username</label>
              <input
                type="text"
                value={jumpUser}
                onChange={(e) => setJumpUser(e.target.value)}
                className={inputClass}
              />
            </div>
            <div>
              <label className="mb-1 block text-sm font-medium">Jump Host Key</label>
              <input
                type="file"
                onChange={(e) => handleKeyUpload(e, "jump")}
                className={inputClass}
              />
              {jumpKeyName && (
                <p className="mt-1 text-xs text-green-700">Selected key: {jumpKeyName}</p>
              )}
            </div>
          </div>
        )}
      </section>

      {/* Node connectivity (F2 probe): reachability + GPU type/count + ROCm */}
      {existing && (
        <section className="rounded-xl border border-border bg-card p-5 shadow-sm">
          <div className="mb-3 flex items-center justify-between">
            <div>
              <h2 className="font-semibold">Node Connectivity</h2>
              <p className="text-sm text-muted-foreground">
                Probe TCP reachability and collect GPU type, GPU count, and ROCm version.
              </p>
            </div>
            <button
              onClick={handleProbe}
              disabled={probing}
              className="inline-flex items-center gap-2 rounded-lg border border-input bg-card px-4 py-2 text-sm font-medium hover:bg-muted disabled:cursor-not-allowed disabled:opacity-50"
            >
              <RadioTower className={`h-4 w-4 ${probing ? "animate-pulse" : ""}`} />
              {probing ? "Probing…" : "Probe Nodes"}
            </button>
          </div>

          {existing.statuses && existing.statuses.length > 0 ? (
            <div className="space-y-2">
              {existing.statuses.map((s) => (
                <div
                  key={s.host}
                  className="flex items-center justify-between rounded-lg border border-border p-3"
                >
                  <div className="flex items-center gap-3">
                    {s.reachable ? (
                      <CheckCircle className="h-5 w-5 text-green-500" />
                    ) : (
                      <XCircle className="h-5 w-5 text-red-500" />
                    )}
                    <div>
                      <div className="font-medium">{s.host}</div>
                      <div className="text-xs text-muted-foreground">
                        {s.reachable
                          ? `${s.gpu_type || "GPU n/a"} ×${s.gpu_count ?? "?"} • ROCm ${s.rocm_version || "n/a"}`
                          : s.error || "unreachable"}
                      </div>
                    </div>
                  </div>
                  {s.checked_at && (
                    <span className="text-xs text-muted-foreground">
                      {new Date(s.checked_at).toLocaleTimeString()}
                    </span>
                  )}
                </div>
              ))}
            </div>
          ) : (
            <p className="rounded-lg border border-dashed border-border p-4 text-center text-sm text-muted-foreground">
              No probe results yet. Click “Probe Nodes” to check connectivity.
            </p>
          )}
        </section>
      )}

      <div className="flex justify-end">
        <button
          onClick={handleSave}
          disabled={saving}
          className="inline-flex items-center gap-2 rounded-lg bg-primary px-6 py-3 font-medium text-primary-foreground hover:bg-primary/90 disabled:cursor-not-allowed disabled:opacity-50"
        >
          <Save className="h-4 w-4" />
          {saving ? "Saving…" : "Save Inventory"}
        </button>
      </div>
    </div>
  );
}
