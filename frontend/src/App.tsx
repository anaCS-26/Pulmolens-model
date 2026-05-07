import React, { useEffect, useMemo, useState } from "react";
import { predict, summarizeAI, warmup } from "./api";
import { LABELS, CLINICIAN_COPY, GUIDELINE_TAGS, THRESHOLDS } from "./data/constants";
import { Step } from "./types";

// Layout
import { Header } from "./components/layout/Header";
import { Footer } from "./components/layout/Footer";
import { Stepper } from "./components/layout/Stepper";

// Pages
import { Landing } from "./pages/Landing";
import { About } from "./pages/About";
import { Consent } from "./pages/Consent";
import { UploadPanel } from "./pages/UploadPanel";
import { Processing } from "./pages/Processing";
import { Results } from "./pages/Results";

// dev self-check
function runDevChecks() {
  const msgs: string[] = [];
  LABELS.forEach((l) => {
    if (typeof CLINICIAN_COPY[l] !== "string") msgs.push(`Missing copy: ${l}`);
    if (!Array.isArray(GUIDELINE_TAGS[l])) msgs.push(`Missing tags: ${l}`);
  });
  if (msgs.length) console.warn("[PulmoLens DevCheck]", msgs);
}

export default function App() {
  const [step, setStep] = useState<Step>("landing");
  const [agreed, setAgreed] = useState(false);
  const [file, setFile] = useState<File | null>(null);
  const [imageURL, setImageURL] = useState<string | null>(null);
  const [progress, setProgress] = useState(0);
  const [searchTerm, setSearchTerm] = useState("");
  const [heatmapOpacity, setHeatmapOpacity] = useState<number>(0.5);
  const [heatmap, setHeatmap] = useState<string | null>(null);
  const [imageId, setImageId] = useState<string | null>(null);
  const [report, setReport] = useState<string | null>(null);
  const [sources, setSources] = useState<string[]>([]);

  const [showPatientSheet, setShowPatientSheet] = useState(false);

  // server inference state
  const [serverPreds, setServerPreds] = useState<Record<string, number> | null>(null);
  const [isSummarizing, setIsSummarizing] = useState(false);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);

  useEffect(() => { 
    runDevChecks(); 
    // Warmup the backend as soon as the app loads to mitigate cold starts
    warmup();
  }, []);

  // predictions — ONLY from server
  const predictions = useMemo(() => {
    if (!serverPreds) return [] as { label: string; prob: number }[];
    return Object.entries(serverPreds)
      .map(([label, prob]) => ({ label, prob }))
      .sort((a, b) => b.prob - a.prob);
  }, [serverPreds]);

  // faux progress bar while waiting for server
  useEffect(() => {
    if (step !== "processing") return;
    setProgress(5);
    const id = setInterval(() => {
      setProgress((p) => Math.min(p + Math.random() * 22, 95));
    }, 500);
    return () => clearInterval(id);
  }, [step]);

  // image preview
  useEffect(() => {
    if (!file) return;
    const url = URL.createObjectURL(file);
    setImageURL(url);
    return () => URL.revokeObjectURL(url);
  }, [file]);

  const filtered = useMemo(
    () => predictions.filter((p) => p.label.toLowerCase().includes(searchTerm.toLowerCase())),
    [predictions, searchTerm]
  );
  const actionable = filtered.filter((p) => p.prob >= (THRESHOLDS[p.label] || 0.5));

  // upload & start inference
  const handleFile = async (f: File) => {
    setFile(f);
    setServerPreds(null);
    setErrorMsg(null);
    setStep("processing");
    try {
      const { predictions, heatmap, image_id } = await predict(f);
      
      setServerPreds(predictions || null);
      setHeatmap(heatmap || null);
      setImageId(image_id || null);
      setProgress(100);
      setStep("results");

      // Stage 2: Trigger AI Summarization in the background
      const generateReport = async (preds: Record<string, number>, heatmapB64: string) => {
        setIsSummarizing(true);
        try {
            const data = await summarizeAI(preds, heatmapB64);
            setReport(data.report);
            setSources(data.sources);
        } catch (err) {
            console.error("Report generation failed:", err);
        } finally {
            setIsSummarizing(false);
        }
      };
      if (predictions && heatmap) {
        generateReport(predictions, heatmap);
      }
    } catch (e: any) {
      console.error(e);
      setErrorMsg(`Upload failed: ${e?.message || e}`);
      setStep("results");
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 to-slate-100 text-slate-900">
      <Header step={step} setStep={setStep} agreed={agreed} hasFile={!!file} />
      <main className="mx-auto max-w-6xl px-4 pb-24">
        <Stepper step={step} setStep={setStep} agreed={agreed} hasFile={!!file} />
        {step === "landing" && (
          <Landing onStart={() => setStep("consent")} onLearnMore={() => setStep("about")} />
        )}
        {step === "about" && <About onBack={() => setStep("landing")} />}
        {step === "consent" && (
          <Consent
            agreed={agreed}
            setAgreed={setAgreed}
            onContinue={() => setStep("upload")}
            onBack={() => setStep("landing")}
          />
        )}
        {step === "upload" && <UploadPanel onFile={handleFile} onBack={() => setStep("consent")} />}
        {step === "processing" && <Processing progress={progress} />}
        {step === "results" && (
          <ErrorBoundary>
            <Results
              file={file}
              imageURL={imageURL}
              searchTerm={searchTerm}
              setSearchTerm={setSearchTerm}
              predictions={filtered}
              actionable={actionable}
              heatmapOpacity={heatmapOpacity}
              setHeatmapOpacity={setHeatmapOpacity}
              onRestart={() => {
                setFile(null);
                setImageURL(null);
                setServerPreds(null);
                setHeatmap(null);
                setImageId(null);
                setReport(null);
                setSources([]);
                setErrorMsg(null);
                setStep("upload");
              }}
              showPatientSheet={showPatientSheet}
              setShowPatientSheet={setShowPatientSheet}
              errorMsg={errorMsg}
              heatmap={heatmap}
              imageId={imageId}
              report={report}
              sources={sources}
              isSummarizing={isSummarizing}
            />
          </ErrorBoundary>
        )}
      </main>
      <Footer />
    </div>
  );
}

// Simple internal ErrorBoundary component
class ErrorBoundary extends React.Component<{ children: React.ReactNode }, { hasError: boolean; error: any }> {
  constructor(props: any) {
    super(props);
    this.state = { hasError: false, error: null };
  }
  static getDerivedStateFromError(error: any) {
    return { hasError: true, error };
  }
  render() {
    if (this.state.hasError) {
      return (
        <div className="p-8 bg-rose-50 border border-rose-200 rounded-3xl text-rose-900">
          <h2 className="text-xl font-bold">App Rendering Error</h2>
          <pre className="mt-4 text-xs overflow-auto">{this.state.error?.toString()}</pre>
          <button onClick={() => window.location.reload()} className="mt-4 bg-rose-900 text-white px-4 py-2 rounded-xl">Reload Page</button>
        </div>
      );
    }
    return this.props.children;
  }
}
