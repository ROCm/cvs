import { Link, NavLink, Outlet } from "react-router-dom";
import { Loader2, Server } from "lucide-react";
import { useEffect, useRef, useState } from "react";
import { TILES } from "@/tiles";
import { useInventory } from "@/shared/inventory";
import { getProbeStatus } from "@/shared/api";

const navLinkClass = ({ isActive }: { isActive: boolean }) =>
  `flex items-center gap-2 rounded-md px-3 py-1.5 text-sm font-medium transition-colors ${
    isActive
      ? "bg-primary/10 text-primary"
      : "text-muted-foreground hover:bg-muted hover:text-foreground"
  }`;

const disabledNavClass =
  "flex items-center gap-2 rounded-md px-3 py-1.5 text-sm font-medium cursor-not-allowed select-none text-muted-foreground/40";

export default function AppShell() {
  const { status } = useInventory();
  const configured = !!status?.configured;

  // Poll probe-status while the initial fleet sweep is in progress.
  // probeReady flips true only after the first full probe — never on incremental
  // add/remove saves — so this gate does not trigger when the user edits nodes.
  const [probeReady, setProbeReady] = useState(true);
  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null);

  useEffect(() => {
    if (!configured) {
      setProbeReady(true);
      return;
    }

    const check = async () => {
      try {
        const s = await getProbeStatus();
        // Only block when the pool exists (total > 0) but the initial sweep
        // hasn't finished — avoids blocking forever on a misconfigured key.
        const blocking = !s.ready;
        setProbeReady(!blocking);
        if (!blocking && pollRef.current) {
          clearInterval(pollRef.current);
          pollRef.current = null;
        }
      } catch {
        // Network error — fail open so we never permanently block.
        setProbeReady(true);
      }
    };

    void check();
    pollRef.current = setInterval(() => { void check(); }, 2000);
    return () => {
      if (pollRef.current) clearInterval(pollRef.current);
    };
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [configured]);

  const tilesBlocked = configured && !probeReady;

  return (
    <div className="flex h-full flex-col bg-background text-foreground">
      <header className="flex items-center gap-6 border-b border-border bg-card px-6 py-3 shadow-sm">
        <Link to="/" className="flex items-center gap-2 font-semibold">
          <span className="inline-block h-3 w-3 rounded-sm bg-primary" />
          CVS Platform
        </Link>
        <nav className="flex items-center gap-1">
          <NavLink to="/inventory" className={navLinkClass}>
            <Server className="h-4 w-4" />
            Inventory
            {tilesBlocked && (
              <Loader2 className="h-3 w-3 animate-spin text-muted-foreground" />
            )}
          </NavLink>
          {configured &&
            TILES.map((t) =>
              tilesBlocked ? (
                <span
                  key={t.id}
                  className={disabledNavClass}
                  title="Fleet probe in progress — available shortly"
                >
                  <t.icon className="h-4 w-4" />
                  {t.name}
                </span>
              ) : (
                <NavLink key={t.id} to={t.path} className={navLinkClass}>
                  <t.icon className="h-4 w-4" />
                  {t.name}
                </NavLink>
              ),
            )}
        </nav>
      </header>
      <main className="flex-1 overflow-auto p-6">
        {!configured && (
          <div className="mx-auto mb-6 max-w-3xl rounded-lg border border-amber-200 bg-amber-50 p-3 text-sm text-amber-800">
            Configure your fleet inventory below to unlock the Test Execution, Cluster Monitor,
            and Fleet Metrics tiles.
          </div>
        )}
        {tilesBlocked && (
          <div className="mx-auto mb-6 max-w-3xl rounded-lg border border-blue-200 bg-blue-50 p-3 text-sm text-blue-800">
            <span className="inline-flex items-center gap-2">
              <Loader2 className="h-4 w-4 animate-spin" />
              Fleet probe in progress — checking SSH reachability across all nodes. Tiles will unlock shortly.
            </span>
          </div>
        )}
        <Outlet context={{ tilesBlocked }} />
      </main>
    </div>
  );
}
