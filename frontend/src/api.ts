export type Predictions = Record<string, number>;
export interface AnalyzeResp {
  predictions: Record<string, number>;
  heatmap: string;
  imageId: string;
  version?: string;
  report?: string; // Kept for legacy compatibility
  sources?: string[]; 
}

export interface SummarizeResp {
  report: string;
  sources: string[];
}

const API_BASE_URL =
  import.meta.env.VITE_API_BASE_URL?.trim() || "http://localhost:8000";

const IS_DEMO = import.meta.env.VITE_DEMO_MODE === 'true';

export async function warmup() {
  if (IS_DEMO) {
    console.log("[PulmoLens] Demo Mode: Skipping backend warmup");
    return;
  }
  try {
    const res = await fetch(`${API_BASE_URL}/warmup`);
    if (res.ok) console.log("Backend warmed up");
  } catch (e) {
    console.warn("Warmup ping failed (might still be spinning up)", e);
  }
}

export async function uploadAndAnalyze(file: File, retryCount = 0): Promise<AnalyzeResp> {
  if (IS_DEMO) {
    console.log("[PulmoLens] Demo Mode: Mocking analysis for", file.name);
    // Simulate network latency
    await new Promise(resolve => setTimeout(resolve, 2000));
    
    return {
      predictions: {
        "Pneumonia": 0.82,
        "Effusion": 0.45,
        "Infiltration": 0.15,
        "Atelectasis": 0.05
      },
      report: "**Radiographic Signature**: There is a focal area of consolidation in the right lower lobe, consistent with pneumonia. No large pleural effusions or pneumothorax identified.\n\n**Clinical Management**: Initiate antibiotics as per local community-acquired pneumonia (CAP) guidelines. Consider CURB-65 score for severity assessment.\n\n**Patient Summary**: The scan shows a localized area of lung inflammation (pneumonia) in the lower right lung. This typically requires a course of antibiotics and follow-up to ensure clearance.",
      heatmap: "", // Empty string means UI will show fallback gradient or no heatmap
      imageId: "demo_mode_id",
      sources: ["BTS Guidelines for CAP", "NICE Chest Infection Pathway"]
    };
  }

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

export async function summarizeAI(predictions: Record<string, number>, heatmap: string): Promise<SummarizeResp> {
  if (IS_DEMO) {
    await new Promise(resolve => setTimeout(resolve, 3500)); // Longer wait for the "thinking" loader demo
    return {
      report: "**Radiographic Signature**: There is a focal area of consolidation in the right lower lobe, consistent with pneumonia. No large pleural effusions or pneumothorax identified.\n\n**Clinical Management**: Initiate antibiotics as per local community-acquired pneumonia (CAP) guidelines. Consider CURB-65 score for severity assessment.\n\n**Patient Summary**: The scan shows a localized area of lung inflammation (pneumonia) in the lower right lung. This typically requires a course of antibiotics and follow-up to ensure clearance.",
      sources: ["BTS Guidelines for CAP (2024)", "NICE Chest Infection Pathway"]
    };
  }

  const res = await fetch(`${API_BASE_URL}/summarize`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ predictions, heatmap })
  });

  if (!res.ok) {
    const err = await res.text();
    throw new Error(`Summarize failed: ${err}`);
  }

  return (await res.json()) as SummarizeResp;
}

export async function submitFeedback(file: File, rating: "good" | "bad", predictions?: Record<string, number>, details?: string) {
  if (IS_DEMO) {
    console.log("[PulmoLens] Demo Mode: Mocking feedback submission", { rating, predictions, details });
    await new Promise(resolve => setTimeout(resolve, 800));
    return { status: "received", message: "Demo Mode: Feedback received (mocked)" };
  }

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
