import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  server: {
    host: "127.0.0.1",
    port: 5173
  },
  build: {
    rollupOptions: {
      output: {
        // Keep the charting vendor code isolated for better caching and
        // independent loading from application code.
        manualChunks: {
          charts: ["recharts"]
        }
      }
    }
  }
});
