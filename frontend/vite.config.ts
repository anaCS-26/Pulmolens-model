import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [react()],
  define: {
    "import.meta.env.VITE_API_BASE_URL": JSON.stringify("https://pulmolens-container.jollymushroom-d4a6f563.canadacentral.azurecontainerapps.io"),
  },
});
