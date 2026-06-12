import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useState,
  type ReactNode,
} from "react";
import { getInventory, type InventoryStatus } from "./api";

interface InventoryContextValue {
  status: InventoryStatus | null;
  loading: boolean;
  error: string | null;
  refresh: () => Promise<void>;
}

const InventoryContext = createContext<InventoryContextValue | undefined>(undefined);

// InventoryProvider loads the fleet inventory status once and exposes it to the
// app shell and the hard gate. `configured` drives whether the tiles are shown.
export function InventoryProvider({ children }: { children: ReactNode }) {
  const [status, setStatus] = useState<InventoryStatus | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const refresh = useCallback(async () => {
    setLoading(true);
    try {
      setStatus(await getInventory());
      setError(null);
    } catch (e) {
      setError(e instanceof Error ? e.message : "failed to load inventory");
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    void refresh();
  }, [refresh]);

  return (
    <InventoryContext.Provider value={{ status, loading, error, refresh }}>
      {children}
    </InventoryContext.Provider>
  );
}

export function useInventory(): InventoryContextValue {
  const ctx = useContext(InventoryContext);
  if (!ctx) {
    throw new Error("useInventory must be used within an InventoryProvider");
  }
  return ctx;
}
