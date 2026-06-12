// Minimal fetch wrapper for the CVS API. All tiles share one origin (the Go
// daemon), so paths are relative.

export async function apiGet<T>(path: string): Promise<T> {
  const res = await fetch(path, { headers: { Accept: "application/json" } });
  if (!res.ok) {
    throw new Error(`GET ${path} failed: ${res.status}`);
  }
  return (await res.json()) as T;
}

async function errorDetail(res: Response): Promise<string> {
  try {
    const body = (await res.json()) as { error?: string };
    return body?.error ?? "";
  } catch {
    return "";
  }
}

export async function apiPost<T>(path: string, body: unknown): Promise<T> {
  const res = await fetch(path, {
    method: "POST",
    headers: { "Content-Type": "application/json", Accept: "application/json" },
    body: JSON.stringify(body),
  });
  if (!res.ok) {
    throw new Error((await errorDetail(res)) || `POST ${path} failed: ${res.status}`);
  }
  return (await res.json()) as T;
}

// One CVS test suite, as reported by `cvs list --json`.
export interface Suite {
  name: string;
  module_path: string;
  module: string;
  package: string;
  category: string;
}

export async function listSuites(): Promise<Suite[]> {
  const data = await apiGet<{ suites: Suite[] }>("/api/v1/suites");
  return data.suites ?? [];
}

// A node of the inferred config form (from GET /suites/{suite}/schema).
export interface Field {
  name: string;
  type: "object" | "array" | "string" | "number" | "boolean" | "null";
  description?: string;
  fields?: Field[];
  item?: Field;
}

export interface SuiteSchema {
  suite: string;
  test_name: string;
  has_config: boolean;
  config_path: string | null;
  format?: string;
  schema: Field[];
  example: unknown;
}

export async function getSuiteSchema(suite: string): Promise<SuiteSchema> {
  return apiGet<SuiteSchema>(`/api/v1/suites/${encodeURIComponent(suite)}/schema`);
}

// --- Inventory (the inventory-first gate shared by all tiles) ---

export type AuthMethod = "key" | "password";

export interface JumpHost {
  host: string;
  username: string;
  auth_method: AuthMethod;
  key_name?: string;
  node_username?: string;
  node_key_file?: string;
}

export interface NodeStatus {
  host: string;
  reachable: boolean;
  gpu_type?: string;
  gpu_count?: number;
  rocm_version?: string;
  error?: string;
  checked_at?: string;
}

export interface Inventory {
  nodes: string[];
  username: string;
  auth_method: AuthMethod;
  key_name?: string;
  jump_host?: JumpHost | null;
  statuses?: NodeStatus[];
  updated_at: string;
}

export interface InventoryStatus {
  configured: boolean;
  inventory?: Inventory;
}

export interface SaveInventoryRequest {
  nodes: string[];
  username: string;
  auth_method: AuthMethod;
  key_name?: string;
  jump_host?: JumpHost | null;
}

export async function getInventory(): Promise<InventoryStatus> {
  return apiGet<InventoryStatus>("/api/v1/inventory");
}

export async function saveInventory(req: SaveInventoryRequest): Promise<InventoryStatus> {
  return apiPost<InventoryStatus>("/api/v1/inventory", req);
}

// Runs the F2 connectivity + basic-info sweep and returns the updated inventory
// (with per-node reachability, GPU type/count, and ROCm version).
export async function probeInventory(): Promise<InventoryStatus> {
  return apiPost<InventoryStatus>("/api/v1/inventory/probe", {});
}

export async function listKeys(): Promise<string[]> {
  const data = await apiGet<{ keys: string[] }>("/api/v1/inventory/keys");
  return data.keys ?? [];
}

export async function uploadSshKey(file: File): Promise<{ key_name: string; path: string }> {
  const form = new FormData();
  form.append("key", file);
  const res = await fetch("/api/v1/inventory/keys", { method: "POST", body: form });
  if (!res.ok) {
    throw new Error((await errorDetail(res)) || `key upload failed: ${res.status}`);
  }
  return (await res.json()) as { key_name: string; path: string };
}
