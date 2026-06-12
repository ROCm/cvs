import { Link, NavLink, Outlet } from "react-router-dom";
import { Server } from "lucide-react";
import { TILES } from "@/tiles";
import { useInventory } from "@/shared/inventory";

const navLinkClass = ({ isActive }: { isActive: boolean }) =>
  `flex items-center gap-2 rounded-md px-3 py-1.5 text-sm font-medium transition-colors ${
    isActive
      ? "bg-primary/10 text-primary"
      : "text-muted-foreground hover:bg-muted hover:text-foreground"
  }`;

export default function AppShell() {
  const { status } = useInventory();
  const configured = !!status?.configured;

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
          </NavLink>
          {configured &&
            TILES.map((t) => (
              <NavLink key={t.id} to={t.path} className={navLinkClass}>
                <t.icon className="h-4 w-4" />
                {t.name}
              </NavLink>
            ))}
        </nav>
      </header>
      <main className="flex-1 overflow-auto p-6">
        {!configured && (
          <div className="mx-auto mb-6 max-w-3xl rounded-lg border border-amber-200 bg-amber-50 p-3 text-sm text-amber-800">
            Configure your fleet inventory below to unlock the Test Execution, Cluster Monitor,
            and Fleet Metrics tiles.
          </div>
        )}
        <Outlet />
      </main>
    </div>
  );
}
