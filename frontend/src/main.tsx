import "./index.css";
import "./telemetry"; // Azure Application Insights: must load before app renders
import React from "react";
import ReactDOM from "react-dom/client";
import App from "./App";

console.log("[PulmoLens] Initializing application...");
ReactDOM.createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>
);
