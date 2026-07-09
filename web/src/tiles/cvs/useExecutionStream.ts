import { useEffect, useRef, useState } from "react";
import {
  getExecution,
  getExecutionLogs,
  isTerminal,
  streamExecution,
  type Execution,
} from "@/shared/api";

export interface ExecutionStream {
  execution: Execution | null;
  logs: string;
}

// useExecutionStream subscribes to an execution's live log + status over
// WebSocket and keeps both in state. Log lines are batched (flushed on an
// interval) so a heavy burst can't overwhelm React and get the subscriber
// dropped. When the socket closes it reconciles against the REST store — the
// authoritative source of the final status and full log — polling until the run
// reaches a terminal state. Pass null to disable.
export function useExecutionStream(executionId: string | null): ExecutionStream {
  const [execution, setExecution] = useState<Execution | null>(null);
  const [logs, setLogs] = useState<string>("");
  const bufRef = useRef<string[]>([]);

  useEffect(() => {
    if (!executionId) {
      setExecution(null);
      setLogs("");
      return;
    }
    let cancelled = false;
    bufRef.current = [];
    setLogs("");
    setExecution(null);

    const flush = window.setInterval(() => {
      if (cancelled || bufRef.current.length === 0) return;
      const chunk = bufRef.current.join("\n") + "\n";
      bufRef.current = [];
      setLogs((prev) => prev + chunk);
    }, 200);

    const reconcile = async () => {
      while (!cancelled) {
        let ex: Execution;
        try {
          ex = await getExecution(executionId);
        } catch {
          return;
        }
        if (cancelled) return;
        setExecution(ex);
        if (isTerminal(ex.status)) {
          try {
            const lg = await getExecutionLogs(executionId);
            if (!cancelled) setLogs(lg.logs); // authoritative full log
          } catch {
            // keep whatever streamed
          }
          return;
        }
        await new Promise((r) => setTimeout(r, 1500));
      }
    };

    const close = streamExecution(executionId, {
      onLog: (line) => bufRef.current.push(line),
      onStatus: (ex) => {
        if (!cancelled) setExecution(ex);
      },
      onClose: () => {
        void reconcile();
      },
    });

    return () => {
      cancelled = true;
      window.clearInterval(flush);
      close();
    };
  }, [executionId]);

  return { execution, logs };
}
