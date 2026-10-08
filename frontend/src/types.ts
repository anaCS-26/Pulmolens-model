export type Step = "landing" | "about" | "upload" | "processing" | "results";

/** One model output row. `threshold` is the operating point it is judged against. */
export interface Prediction {
  label: string;
  prob: number;
  threshold: number;
}
