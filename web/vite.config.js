import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import path from "node:path";
// The Go daemon serves the built bundle from "/" (embedded). During local dev,
// the Vite server proxies API + WS calls to the running daemon on :8080.
export default defineConfig({
    plugins: [react()],
    resolve: {
        alias: {
            "@": path.resolve(__dirname, "./src"),
        },
    },
    server: {
        port: 3000,
        proxy: {
            "/api": { target: "http://localhost:8080", changeOrigin: true },
            "/ws": { target: "ws://localhost:8080", ws: true },
        },
    },
    build: {
        outDir: "dist",
    },
});
