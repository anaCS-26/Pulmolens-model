export type Predictions = Record<string, number>;
export interface AnalyzeResp {
  inferenceId?: string; // kept for compatibility if needed
  predictions?: Record<string, number>;
  heatmap?: string;
  imageId?: string;
  report?: string; // Caught from RAG backend!
  sources?: string[]; // Metadata citations
}

// Use the local env var for testing, or default to localhost since we are running uvicorn locally now!
// When you re-deploy to Azure, this should point back to the Azure Container App URL.
const API_BASE_URL = "https://pulmolens-container.jollymushroom-d4a6f563.canadacentral.azurecontainerapps.io";

export async function uploadAndAnalyze(file: File) {
  const fd = new FormData();
  fd.append("file", file);
  // The backend endpoint is /predict
  const res = await fetch(`${API_BASE_URL}/predict`, { method: "POST", body: fd });
  if (!res.ok) {
    const errText = await res.text();
    throw new Error(`Analyze failed: ${res.status} ${errText}`);
  }
  return (await res.json()) as AnalyzeResp;
}

export async function submitFeedback(file: File, rating: "good" | "bad", predictions?: Record<string, number>, details?: string) {
  const fd = new FormData();
  fd.append("file", file);
  fd.append("rating", rating);
  if (details) fd.append("details", details);
  if (predictions) fd.append("predictions", JSON.stringify(predictions));

  const response = await fetch(`${API_BASE_URL}/feedback`, {
    method: "POST",
    body: fd,
  });

  if (!response.ok) {
    throw new Error(`Feedback failed: ${response.statusText}`);
  }

  return response.json();
}
