import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// Local dev backend (change if needed)
const DEV_BACKEND = process.env.VITE_DEV_BACKEND || "http://localhost:8000";

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      "/api": {
        target: DEV_BACKEND,
        changeOrigin: true
      }
    }
  },
  build: { outDir: "dist" }
});
