import React, { useEffect, useMemo, useRef, useState } from "react";
import { AnimatePresence, MotionConfig, motion } from "framer-motion";
import { uploadAndAnalyze, summarizeAIStream } from "./api";
import { LABELS, CLINICIAN_COPY, GUIDELINE_TAGS, THRESHOLDS } from "./data/constants";
import { Prediction, Step } from "./types";

// Layout
import { Header } from "./components/layout/Header";
import { Footer } from "./components/layout/Footer";

// Pages
import { Landing } from "./pages/Landing";
import { About } from "./pages/About";
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

const CONSENT_KEY = "pulmolens.consent";

export default function App() {
  const [step, setStep] = useState<Step>("landing");
  // Consent lasts for the browser session so a second analysis doesn't re-ask.
  const [agreed, setAgreedState] = useState(() => sessionStorage.getItem(CONSENT_KEY) === "1");
  const setAgreed = (v: boolean) => {
    setAgreedState(v);
    if (v) sessionStorage.setItem(CONSENT_KEY, "1");
    else sessionStorage.removeItem(CONSENT_KEY);
  };
  const [file, setFile] = useState<File | null>(null);
  const [imageURL, setImageURL] = useState<string | null>(null);
  const [attentionOverlay, setAttentionOverlay] = useState<string | null>(null);
  const [report, setReport] = useState<string | null>(null);
  const [sources, setSources] = useState<string[]>([]);

  // server inference state
  const [serverPreds, setServerPreds] = useState<Record<string, number> | null>(null);
  const [isSummarizing, setIsSummarizing] = useState(false);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);

  // Each analysis gets an id so a stale request or stream can't write into a newer one.
  const runId = useRef(0);
  const stepRef = useRef(step);
  stepRef.current = step;

  useEffect(() => {
    runDevChecks();
    // Warmup the backend as soon as the app loads to mitigate cold starts
    import("./api").then(({ warmup }) => warmup());
  }, []);

  useEffect(() => {
    window.scrollTo({ top: 0, behavior: "instant" as ScrollBehavior });
  }, [step]);

  // predictions: ONLY from server, each paired with the threshold it is judged against.
  const predictions = useMemo<Prediction[]>(() => {
    if (!serverPreds) return [];
    return Object.entries(serverPreds)
      .map(([label, prob]) => ({ label, prob, threshold: THRESHOLDS[label] || 0.5 }))
      .sort((a, b) => b.prob - a.prob);
  }, [serverPreds]);

  // image preview
  useEffect(() => {
    if (!file) return;
    const url = URL.createObjectURL(file);
    setImageURL(url);
    return () => URL.revokeObjectURL(url);
  }, [file]);

  // upload & start inference
  const handleFile = async (f: File) => {
    const id = ++runId.current;
    const current = () => id === runId.current;

    setFile(f);
    setServerPreds(null);
    setAttentionOverlay(null);
    setReport(null);
    setSources([]);
    setErrorMsg(null);
    setIsSummarizing(false);
    setStep("processing");
    try {
      const { predictions, attentionOverlay } = await uploadAndAnalyze(f);
      if (!current()) return;

      setServerPreds(predictions || null);
      setAttentionOverlay(attentionOverlay || null);
      // Don't yank the user back if they navigated away while waiting.
      if (stepRef.current === "processing") setStep("results");

      // Stage 2: Trigger AI Summarization in the background
      if (predictions && attentionOverlay) {
        setIsSummarizing(true);
        setReport(""); // Clear previous report for streaming
        try {
          await summarizeAIStream(
            predictions,
            attentionOverlay,
            (chunk) => current() && setReport((prev) => (prev || "") + chunk),
            (sources) => current() && setSources(sources)
          );
        } catch (summErr) {
          console.error("Summarization background task failed:", summErr);
        } finally {
          if (current()) setIsSummarizing(false);
        }
      }
    } catch (e: any) {
      if (!current()) return;
      console.error(e);
      setErrorMsg(`Upload failed: ${e?.message || e}`);
      if (stepRef.current === "processing") setStep("results");
    }
  };

  const restart = () => {
    runId.current++;
    setFile(null);
    setImageURL(null);
    setServerPreds(null);
    setAttentionOverlay(null);
    setReport(null);
    setSources([]);
    setErrorMsg(null);
    setIsSummarizing(false);
    setStep("upload");
  };

  const hasResults = !!serverPreds || !!errorMsg;

  return (
    <MotionConfig reducedMotion="user">
      <div className="relative min-h-screen overflow-x-clip print:hidden">
        <Header step={step} setStep={setStep} hasResults={hasResults} />
        <main className="relative min-h-[calc(100vh-180px)]">
          <AnimatePresence mode="wait" initial={false}>
            <motion.div
              key={step}
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              transition={{ duration: 0.18 }}
            >
              {step === "landing" && (
                <Landing onStart={() => setStep("upload")} onLearnMore={() => setStep("about")} />
              )}
              {step === "about" && <About onBack={() => setStep("landing")} onStart={() => setStep("upload")} />}
              {step === "upload" && <UploadPanel agreed={agreed} onAgree={setAgreed} onFile={handleFile} />}
              {step === "processing" && <Processing imageURL={imageURL} fileName={file?.name} />}
              {step === "results" && (
                <ErrorBoundary>
                  <Results
                    file={file}
                    imageURL={imageURL}
                    predictions={predictions}
                    onRestart={restart}
                    onRetry={() => file && handleFile(file)}
                    errorMsg={errorMsg}
                    attentionOverlay={attentionOverlay}
                    report={report}
                    sources={sources}
                    isSummarizing={isSummarizing}
                  />
                </ErrorBoundary>
              )}
            </motion.div>
          </AnimatePresence>
        </main>
        <Footer />
      </div>
    </MotionConfig>
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
        <div className="mx-auto mt-16 max-w-2xl border-l-2 border-marker px-6 py-4">
          <h2 className="font-serif text-2xl">App Rendering Error</h2>
          <pre className="mt-4 overflow-auto font-mono text-xs text-marker-dark">{this.state.error?.toString()}</pre>
          <button onClick={() => window.location.reload()} className="mt-5 rounded bg-ink px-4 py-2 text-sm font-medium text-paper">Reload Page</button>
        </div>
      );
    }
    return this.props.children;
  }
}
