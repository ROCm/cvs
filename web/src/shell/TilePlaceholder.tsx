import type { TileMeta } from "@/tiles";

export default function TilePlaceholder({ tile }: { tile: TileMeta }) {
  return (
    <div className="mx-auto max-w-3xl">
      <div className="mb-4 inline-flex rounded-lg bg-primary/10 p-3 text-primary">
        <tile.icon className="h-6 w-6" />
      </div>
      <h1 className="mb-2 text-2xl font-semibold">{tile.name}</h1>
      <p className="mb-6 text-muted-foreground">{tile.description}</p>
      <div className="rounded-lg border border-dashed border-border bg-card p-8 text-center text-muted-foreground">
        Coming soon. This tile is a placeholder in the S0 walking skeleton.
      </div>
    </div>
  );
}
