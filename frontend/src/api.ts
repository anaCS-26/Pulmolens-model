export type Predictions = Record<string, number>;
export interface AnalyzeResp {
  inferenceId?: string; // kept for compatibility if needed
  predictions?: Record<string, number>;
  heatmap?: string;
  imageId?: string;
  report?: string; // Caught from RAG backend!
  sources?: string[]; // Metadata citations
}

const API_BASE_URL =
  import.meta.env.VITE_API_BASE_URL?.trim() || "http://localhost:8000";

export async function warmup() {
  try {
    const res = await fetch(`${API_BASE_URL}/warmup`);
    if (res.ok) console.log("Backend warmed up");
  } catch (e) {
    console.warn("Warmup ping failed (might still be spinning up)", e);
  }
}

export async function uploadAndAnalyze(file: File, retryCount = 0): Promise<AnalyzeResp> {
  const MAX_RETRIES = 2;
  const RETRY_DELAY = 2000; // 2s

  const fd = new FormData();
  fd.append("file", file);
  
  try {
    const res = await fetch(`${API_BASE_URL}/predict`, { 
      method: "POST", 
      body: fd,
      // No standard timeout in fetch, but we can use AbortController if we wanted a hard cutoff
    });

    if (!res.ok) {
      const errText = await res.text();
      // If it's a 503 (Model not loaded) or 504 (Gateway/Spin-up issue), retry
      if ((res.status === 503 || res.status === 504 || res.status === 502) && retryCount < MAX_RETRIES) {
        console.warn(`Attempt ${retryCount + 1} failed with ${res.status}. Retrying in ${RETRY_DELAY}ms...`);
        await new Promise(r => setTimeout(r, RETRY_DELAY));
        return uploadAndAnalyze(file, retryCount + 1);
      }
      throw new Error(`Analyze failed: ${res.status} ${errText}`);
    }
    return (await res.json()) as AnalyzeResp;
  } catch (e: any) {
    if (retryCount < MAX_RETRIES && !e.message?.includes("400")) {
       console.warn(`Network error or timeout. Retrying ${retryCount + 1}...`, e);
       await new Promise(r => setTimeout(r, RETRY_DELAY));
       return uploadAndAnalyze(file, retryCount + 1);
    }
    throw e;
  }
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
