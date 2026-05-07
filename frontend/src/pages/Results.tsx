import React, { useEffect, useRef, useState } from "react";
import { Search, Image as ImageIcon, ThumbsUp, ThumbsDown, Maximize2, X, ZoomIn, ZoomOut, RotateCcw } from "lucide-react";
import { cn } from "../utils/cn";
import { submitFeedback } from "../api";
import { THRESHOLDS } from "../data/constants";
import { ResultRow } from "../components/results/ResultRow";
import { ReportPanel } from "../components/results/ReportPanel";
import { PatientSummarySheet } from "../components/results/PatientSummarySheet";
import { ThinkingLoader } from "../components/ui/ThinkingLoader";

interface ResultsProps {
    file: File | null;
    imageURL: string | null;
    searchTerm: string;
    setSearchTerm: (s: string) => void;
    predictions: { label: string; prob: number }[];
    actionable: { label: string; prob: number }[];
    heatmapOpacity: number;
    setHeatmapOpacity: (v: number) => void;
    onRestart: () => void;
    showPatientSheet: boolean;
    setShowPatientSheet: (v: boolean) => void;
    errorMsg: string | null;
    heatmap: string | null;
    imageId: string | null;
    report: string | null;
    sources: string[];
    isSummarizing?: boolean;
}

// Reveals `target` one character at a time. Adapts cadence when the
// streaming backlog grows so we never fall far behind the model.
function useTypewriter(target: string, baseCps = 70): string {
    const [shown, setShown] = useState("");
    const shownRef = useRef("");
    shownRef.current = shown;

    // If the upstream buffer was reset/replaced (no longer a prefix), restart.
    useEffect(() => {
        if (target.length === 0) {
            if (shownRef.current.length !== 0) setShown("");
            return;
        }
        if (!target.startsWith(shownRef.current)) {
            setShown("");
        }
    }, [target]);

    useEffect(() => {
        if (shown.length >= target.length) return;
        let raf = 0;
        let last = performance.now();
        const tick = (now: number) => {
            const dt = now - last;
            last = now;
            setShown((prev) => {
                if (prev.length >= target.length) return prev;
                const remaining = target.length - prev.length;
                // Catch-up: speed climbs with backlog so a 500-char burst drains in ~1s.
                const cps = baseCps + Math.max(0, remaining - 30) * 6;
                const advance = Math.max(1, Math.round((dt / 1000) * cps));
                return target.slice(0, Math.min(target.length, prev.length + advance));
            });
            raf = requestAnimationFrame(tick);
        };
        raf = requestAnimationFrame(tick);
        return () => cancelAnimationFrame(raf);
    }, [target, shown.length, baseCps]);

    return shown;
}

function MarkdownLite({ text, isStreaming }: { text: string; isStreaming?: boolean }) {
    const displayed = useTypewriter(text || "");
    const stillTyping = !!isStreaming || displayed.length < (text?.length || 0);

    if (!displayed) return null;
    const lines = displayed.split("\n");

    return (
        <div className="space-y-1.5">
            {lines.map((l, i) => {
                const isLast = i === lines.length - 1;
                const trimmed = l.trim();
                const isBullet = trimmed.startsWith("*") && !trimmed.startsWith("**");
                const content = isBullet ? trimmed.slice(1).trim() : trimmed;

                if (!content) return <div key={i} className="h-2" />;

                const parts = content.split(/(\*\*?.*?\*\*?)/);
                const rendered = parts.map((p, j) => {
                    if (/^\*\*?.*?\*\*?$/.test(p)) {
                        const cleanText = p.replace(/\*/g, "");
                        return <strong key={j} className="font-semibold text-slate-900 not-italic">{cleanText}</strong>;
                    }
                    return <React.Fragment key={j}>{p}</React.Fragment>;
                });

                const caret = stillTyping && isLast ? (
                    <span className="inline-block w-[2px] h-[1em] bg-slate-400 ml-0.5 align-[-0.15em] animate-caret-blink" />
                ) : null;

                if (isBullet) {
                    return (
                        <div key={i} className="flex items-start gap-2 pl-2">
                            <div className="h-1.5 w-1.5 rounded-full bg-slate-400 mt-2 shrink-0" />
                            <div>{rendered}{caret}</div>
                        </div>
                    );
                }
                return <div key={i} className="mb-2 last:mb-0">{rendered}{caret}</div>;
            })}
            <style dangerouslySetInnerHTML={{ __html: `
                @keyframes caret-blink {
                    0%, 50% { opacity: 1; }
                    50.01%, 100% { opacity: 0; }
                }
                .animate-caret-blink { animation: caret-blink 1s steps(1) infinite; }
            `}} />
        </div>
    );
}

export function Results({
    file,
    imageURL,
    searchTerm,
    setSearchTerm,
    predictions,
    actionable,
    heatmapOpacity,
    setHeatmapOpacity,
    onRestart,
    showPatientSheet,
    setShowPatientSheet,
    errorMsg,
    heatmap,
    imageId,
    report,
    sources,
    isSummarizing,
}: ResultsProps) {
    const title = file ? file.name : "demo_cxr.jpg";
    const above = actionable.filter((a) => a.label !== "No findings");
    const [feedbackRating, setFeedbackRating] = useState<"good" | "bad" | null>(null);
    const [showFullScreen, setShowFullScreen] = useState(false);
    const [zoomLevel, setZoomLevel] = useState(1);

    const handleZoomIn = () => setZoomLevel(prev => Math.min(prev + 0.5, 4));
    const handleZoomOut = () => setZoomLevel(prev => Math.max(prev - 0.5, 1));
    const handleResetZoom = () => setZoomLevel(1);

    const handleFeedback = async (rating: "good" | "bad") => {
        if (feedbackRating) return; // already voted
        setFeedbackRating(rating);
        try {
            // Pass server predictions if available
            const preds = predictions.reduce((acc, p) => ({ ...acc, [p.label]: p.prob }), {});
            console.log("Submitting feedback with preds:", preds);

            if (file) {
                await submitFeedback(file, rating, preds);
            } else {
                console.error("No file to submit with feedback");
            }
        } catch (e) {
            console.error("Feedback failed", e);
        }
    };

    return (
        <section className="mt-8 grid grid-cols-1 lg:grid-cols-3 gap-8">
            <div className="lg:col-span-2 p-4 md:p-6 rounded-3xl bg-white shadow-sm border border-slate-200">
                <div className="flex items-center justify-between">
                    <div>
                        <h2 className="text-xl md:text-2xl font-semibold tracking-tight">Results</h2>
                        <div className="text-sm text-slate-500 mt-1">{title}</div>
                    </div>
                    <div className="flex items-center gap-3">
                        <div className="flex items-center gap-2">
                            <span className="text-sm text-slate-600">Overlay Opacity</span>
                            <input
                                type="range"
                                min="0"
                                max="1"
                                step="0.1"
                                value={heatmapOpacity}
                                onChange={(e) => setHeatmapOpacity(parseFloat(e.target.value))}
                                className="w-24 h-2 bg-slate-200 rounded-lg appearance-none cursor-pointer accent-slate-900"
                            />
                        </div>
                        <button onClick={onRestart} className="rounded-xl border border-slate-300 bg-white px-3 py-2 text-sm">
                            Analyse another image
                        </button>
                    </div>
                </div>

                <div className="mt-8 grid grid-cols-1 md:grid-cols-5 gap-4">
                    <div className="md:col-span-3">
                        <div className="relative aspect-[4/3] w-full rounded-2xl bg-slate-950 overflow-hidden">
                            {imageURL ? (
                                <img src={imageURL} alt="Uploaded CXR" className="absolute inset-0 h-full w-full object-contain" />
                            ) : (
                                <div className="absolute inset-0 grid place-items-center text-white/40">
                                    <ImageIcon className="h-12 w-12" />
                                </div>
                            )}
                            {heatmapOpacity > 0 && (
                                heatmap ? (
                                    <img src={heatmap} alt="Grad-CAM Heatmap" className="absolute inset-0 h-full w-full object-contain transition-opacity duration-200" style={{ opacity: heatmapOpacity }} />
                                ) : (
                                    <div
                                        className="absolute inset-0 pointer-events-none mix-blend-screen transition-opacity duration-200"
                                        style={{
                                            opacity: heatmapOpacity * 0.5,
                                            background:
                                                "radial-gradient(circle at 60% 40%, rgba(239,68,68,0.35), transparent 35%), radial-gradient(circle at 30% 70%, rgba(34,197,94,0.35), transparent 30%), radial-gradient(circle at 75% 80%, rgba(59,130,246,0.35), transparent 25%)",
                                        }}
                                    />
                                )
                            )}
                            <button
                                onClick={() => setShowFullScreen(true)}
                                className="absolute bottom-3 right-3 p-2 bg-black/50 hover:bg-black/70 text-white rounded-lg transition-colors backdrop-blur-sm"
                            >
                                <Maximize2 className="h-5 w-5" />
                            </button>
                        </div>
                        <div className="mt-3 text-xs text-slate-500">Overlay slot for Grad‑CAM/attention maps.</div>
                    </div>

                    <div className="md:col-span-2">
                        {errorMsg && (
                            <div className="rounded-xl border border-rose-300 bg-rose-50 text-rose-900 p-3 text-sm mb-3">
                                {errorMsg}
                            </div>
                        )}
                        <div className="rounded-2xl border border-slate-200 bg-slate-50 p-3">
                            <div className="flex items-center gap-2 text-sm"><Search className="h-4 w-4" /> Filter</div>
                            <input value={searchTerm} onChange={(e) => setSearchTerm(e.target.value)} placeholder="Search findings…" className="mt-2 w-full rounded-xl border border-slate-300 bg-white px-3 py-2 text-sm" />
                        </div>

                        <div className="mt-4 space-y-2 max-h-[22rem] overflow-auto pr-1">
                            {(!predictions || predictions.length === 0) && !errorMsg && (
                                <div className="text-sm text-slate-500">No predictions to display yet.</div>
                            )}
                            {(predictions || []).map((p) => (
                                <ResultRow key={p?.label || Math.random()} p={p} threshold={THRESHOLDS[p?.label] || 0.5} />
                            ))}
                        </div>
                    </div>
                </div>

                {/* AI RAG REPORT AREA - PLACED BELOW GRADCAM AS REQUESTED */}
                {isSummarizing && !(typeof report === 'string' && report.length > 0) && (
                    <div className="mt-8">
                        <ThinkingLoader />
                    </div>
                )}

                {typeof report === 'string' && report.length > 0 && (
                    <div className="mt-10 animate-in fade-in slide-in-from-bottom-2 duration-500">
                        <div className="border-l border-slate-300 pl-6 md:pl-8">
                            <div className="flex items-baseline justify-between mb-4 gap-4">
                                <div className="flex items-baseline gap-3 min-w-0">
                                    <span className="font-mono text-[10px] tracking-[0.2em] uppercase text-slate-400 shrink-0">§ Note</span>
                                    <h4 className="font-serif text-xl md:text-2xl text-slate-900 italic font-medium leading-none truncate">
                                        Clinical synthesis
                                    </h4>
                                </div>
                                {sources && sources.length > 0 && (
                                    <span className="font-mono text-[10px] tracking-wider text-slate-400 shrink-0">
                                        {sources.length} {sources.length === 1 ? 'ref.' : 'refs.'}
                                    </span>
                                )}
                            </div>

                            <div className="font-serif text-[15.5px] md:text-base text-slate-700 leading-[1.75] [&_strong]:text-slate-900 [&_strong]:font-semibold">
                                <MarkdownLite text={report} isStreaming={isSummarizing} />
                            </div>

                            {sources && sources.length > 0 && (
                                <div className="mt-6 pt-4 border-t border-slate-200/70">
                                    <div className="font-mono text-[10px] tracking-[0.18em] uppercase text-slate-400 mb-2.5">References</div>
                                    <ol className="space-y-1.5 list-none">
                                        {sources.map((s, idx) => (
                                            <li key={idx} className="font-serif text-[13px] text-slate-500 leading-snug flex gap-2.5">
                                                <span className="font-mono text-[11px] text-slate-400 shrink-0 pt-px">[{idx + 1}]</span>
                                                <span>{s}</span>
                                            </li>
                                        ))}
                                    </ol>
                                </div>
                            )}
                        </div>
                    </div>
                )}

                {/* Feedback Section - Moved here for better visibility */}
                <div className="mt-8 pt-6 border-t border-slate-100">
                    <div className="text-base font-medium text-slate-900 mb-3">Was this analysis helpful?</div>
                    <div className="flex items-center gap-3">
                        <button
                            onClick={() => handleFeedback("good")}
                            disabled={!!feedbackRating}
                            className={cn(
                                "flex-1 flex items-center justify-center gap-2 rounded-xl border px-4 py-3 text-sm font-medium transition-all",
                                feedbackRating === "good"
                                    ? "bg-emerald-100 border-emerald-300 text-emerald-800 shadow-inner"
                                    : "bg-white border-slate-200 hover:bg-slate-50 hover:border-slate-300 text-slate-700 shadow-sm"
                            )}
                        >
                            <ThumbsUp className="h-4 w-4" /> Yes, helpful
                        </button>
                        <button
                            onClick={() => handleFeedback("bad")}
                            disabled={!!feedbackRating}
                            className={cn(
                                "flex-1 flex items-center justify-center gap-2 rounded-xl border px-4 py-3 text-sm font-medium transition-all",
                                feedbackRating === "bad"
                                    ? "bg-rose-100 border-rose-300 text-rose-800 shadow-inner"
                                    : "bg-white border-slate-200 hover:bg-slate-50 hover:border-slate-300 text-slate-700 shadow-sm"
                            )}
                        >
                            <ThumbsDown className="h-4 w-4" /> Not accurate
                        </button>
                    </div>
                    {feedbackRating && (
                        <div className="mt-3 text-center text-sm text-slate-500 animate-in fade-in slide-in-from-top-1">
                            Thank you for your feedback! It helps us improve.
                        </div>
                    )}
                </div>
            </div>

            <div className="p-4 md:p-6 rounded-3xl bg-white shadow-sm border border-slate-200">
                <ReportPanel actionable={actionable} onPrintClinician={() => window.print()} onOpenPatient={() => setShowPatientSheet(true)} />
            </div>

            {
                showPatientSheet && (
                    <PatientSummarySheet findings={above.length ? above : actionable} onClose={() => setShowPatientSheet(false)} />
                )
            }
            {showFullScreen && (
                <div className="fixed inset-0 z-50 bg-black/95 flex flex-col animate-in fade-in duration-200">
                    <div className="flex items-center justify-between p-4 text-white">
                        <div className="text-lg font-medium">{title}</div>
                        <div className="flex items-center gap-4">
                            <div className="flex items-center gap-2 bg-white/10 rounded-lg p-1">
                                <button onClick={handleZoomOut} className="p-2 hover:bg-white/10 rounded-md transition-colors" disabled={zoomLevel <= 1}>
                                    <ZoomOut className="h-5 w-5" />
                                </button>
                                <span className="text-sm font-mono w-12 text-center">{Math.round(zoomLevel * 100)}%</span>
                                <button onClick={handleZoomIn} className="p-2 hover:bg-white/10 rounded-md transition-colors" disabled={zoomLevel >= 4}>
                                    <ZoomIn className="h-5 w-5" />
                                </button>
                                <div className="w-px h-6 bg-white/20 mx-1" />
                                <button onClick={handleResetZoom} className="p-2 hover:bg-white/10 rounded-md transition-colors" title="Reset Zoom">
                                    <RotateCcw className="h-5 w-5" />
                                </button>
                            </div>
                            <button onClick={() => setShowFullScreen(false)} className="p-2 hover:bg-white/10 rounded-full transition-colors">
                                <X className="h-6 w-6" />
                            </button>
                        </div>
                    </div>
                    <div className="flex-1 overflow-hidden flex items-center justify-center p-4">
                        <div
                            className="relative transition-transform duration-200 ease-out"
                            style={{ transform: `scale(${zoomLevel})` }}
                        >
                            {imageURL && <img src={imageURL} alt="Full screen" className="max-h-[85vh] max-w-[90vw] object-contain" />}
                            {heatmapOpacity > 0 && (
                                heatmap ? (
                                    <img src={heatmap} alt="Heatmap" className="absolute inset-0 h-full w-full object-contain transition-opacity duration-200" style={{ opacity: heatmapOpacity }} />
                                ) : (
                                    <div
                                        className="absolute inset-0 pointer-events-none mix-blend-screen transition-opacity duration-200"
                                        style={{
                                            opacity: heatmapOpacity * 0.5,
                                            background:
                                                "radial-gradient(circle at 60% 40%, rgba(239,68,68,0.35), transparent 35%), radial-gradient(circle at 30% 70%, rgba(34,197,94,0.35), transparent 30%), radial-gradient(circle at 75% 80%, rgba(59,130,246,0.35), transparent 25%)",
                                        }}
                                    />
                                )
                            )}
                        </div>
                    </div>
                    <div className="p-6 flex justify-center">
                        <div className="w-full max-w-md flex items-center gap-4 bg-black/50 backdrop-blur-md p-4 rounded-2xl border border-white/10">
                            <span className="text-sm text-white/80 whitespace-nowrap">Overlay Opacity</span>
                            <input
                                type="range"
                                min="0"
                                max="1"
                                step="0.1"
                                value={heatmapOpacity}
                                onChange={(e) => setHeatmapOpacity(parseFloat(e.target.value))}
                                className="w-full h-2 bg-white/20 rounded-lg appearance-none cursor-pointer accent-white"
                            />
                            <span className="text-sm font-mono text-white/80 w-12 text-right">{Math.round(heatmapOpacity * 100)}%</span>
                        </div>
                    </div>
                </div>
            )}
        </section >
    );
}
