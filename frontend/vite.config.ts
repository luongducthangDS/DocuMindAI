import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  build: {
    outDir: "../dist",
    emptyOutDir: true,
  },
  server: {
    port: 5174,
    // ws: true bắt buộc để Vite proxy cả kết nối WebSocket (/api/v1/ws/...),
    // không chỉ HTTP thường — thiếu nó thì streaming chỉ chạy được khi build
    // production (frontend/backend cùng domain hoặc qua VITE_API_URL), không
    // chạy được ở dev local qua proxy này.
    proxy: { "/api": { target: "http://localhost:8081", ws: true } },
  },
});
