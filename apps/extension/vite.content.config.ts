import { resolve } from "path";
import { defineConfig } from "vite";

export default defineConfig({
  publicDir: false,
  build: {
    outDir: "dist",
    emptyOutDir: false,
    lib: {
      entry: resolve(__dirname, "src/content/index.ts"),
      name: "CognitwinContent",
      formats: ["iife"],
      fileName: () => "content.js",
    },
    rollupOptions: {
      output: {
        extend: true,
      },
    },
  },
});
