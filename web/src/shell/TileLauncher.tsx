import { Link } from "react-router-dom";
import { TILES } from "@/tiles";

export default function TileLauncher() {
  return (
    <div className="mx-auto max-w-5xl">
      <h1 className="mb-2 text-2xl font-semibold">CVS Unified Platform</h1>
      <p className="mb-8 text-muted-foreground">Choose a tile to get started.</p>
      <div className="grid grid-cols-1 gap-6 sm:grid-cols-2 lg:grid-cols-3">
        {TILES.map((t) => (
          <Link
            key={t.id}
            to={t.path}
            className="group rounded-xl border border-border bg-card p-6 shadow-sm transition-all hover:-translate-y-0.5 hover:border-primary/40 hover:shadow-md"
          >
            <div className="mb-4 inline-flex rounded-lg bg-primary/10 p-3 text-primary">
              <t.icon className="h-6 w-6" />
            </div>
            <h2 className="mb-1 text-lg font-semibold group-hover:text-primary">{t.name}</h2>
            <p className="text-sm text-muted-foreground">{t.description}</p>
          </Link>
        ))}
      </div>
    </div>
  );
}
