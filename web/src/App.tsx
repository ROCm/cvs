import type { ComponentType } from "react";
import { Navigate, Route, Routes } from "react-router-dom";
import AppShell from "@/shell/AppShell";
import TileLauncher from "@/shell/TileLauncher";
import TilePlaceholder from "@/shell/TilePlaceholder";
import CvsTile from "@/tiles/cvs/CvsTile";
import InventoryPage from "@/tiles/inventory/InventoryPage";
import { TILES } from "@/tiles";
import { InventoryProvider, useInventory } from "@/shared/inventory";

// Tiles with a real implementation; others fall back to a placeholder page.
const TILE_COMPONENTS: Record<string, ComponentType> = {
  cvs: CvsTile,
};

function GatedRoutes() {
  const { loading, status } = useInventory();

  // Block first paint until we know whether an inventory exists, so a tile deep
  // link doesn't bounce to /inventory and get stuck there once it resolves.
  if (loading && !status) {
    return (
      <div className="flex h-full items-center justify-center text-muted-foreground">
        Loading…
      </div>
    );
  }

  const configured = !!status?.configured;

  return (
    <Routes>
      <Route element={<AppShell />}>
        <Route
          index
          element={configured ? <TileLauncher /> : <Navigate to="/inventory" replace />}
        />
        <Route path="/inventory" element={<InventoryPage />} />
        {TILES.map((t) => {
          const Impl = TILE_COMPONENTS[t.id];
          const page = Impl ? <Impl /> : <TilePlaceholder tile={t} />;
          return (
            <Route
              key={t.id}
              path={`${t.path}/*`}
              element={configured ? page : <Navigate to="/inventory" replace />}
            />
          );
        })}
        <Route path="*" element={<Navigate to="/" replace />} />
      </Route>
    </Routes>
  );
}

export default function App() {
  return (
    <InventoryProvider>
      <GatedRoutes />
    </InventoryProvider>
  );
}
