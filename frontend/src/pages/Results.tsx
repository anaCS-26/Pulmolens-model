import React, { useState } from "react";
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

// Improved component to handle character-level streaming animation with a typewriter effect
function StreamingCharacterText({ text, isComplete }: { text: string; isComplete?: boolean }) {
    if (!text) return null;
    
    // We treat the text as a continuous sequence of characters
    // Using index-based delay to create a typewriter effect
    const characters = text.split("");
    const animationSpeed = 25; // ms per character

    return (
        <span className="inline">
            {characters.map((char, index) => (
                <span 
                    key={`char-${index}`}
                    className="inline-block animate-report-char"
                    style={{ 
                        animationDelay: `${index * animationSpeed}ms`,
                        animationFillMode: 'both'
                    }}
                >
                    {char === " " ? "\u00A0" : char}
                </span>
            ))}
            {!isComplete && (
                <span className="inline-block w-1.5 h-4 bg-indigo-500 ml-1 animate-pulse rounded-sm align-middle" />
            )}
        </span>
    );
}

function MarkdownLite({ text, isStreaming }: { text: string; isStreaming?: boolean }) {
    if (!text) return null;
    const lines = text.split("\n");
    return (
        <div className="space-y-1.5">
            {lines.map((l, i) => {
                const trimmed = l.trim();
                const isBullet = trimmed.startsWith("*") && !trimmed.startsWith("**");
                const content = isBullet ? trimmed.slice(1).trim() : trimmed;

                if (!content) return <div key={i} className="h-2" />;

                const parts = content.split(/(\*\*?.*?\*\*?)/);
                const rendered = parts.map((p, j) => {
                    if (/^\*\*?.*?\*\*?$/.test(p)) {
                        const cleanText = p.replace(/\*/g, "");
                        return <strong key={j} className="font-bold text-slate-900">{cleanText}</strong>;
                    }
                    // For non-bold text, apply character-level animation if streaming
                    return isStreaming ? <StreamingCharacterText key={j} text={p} isComplete={!isStreaming} /> : p;
                });

                if (isBullet) {
                    return (
                        <div key={i} className="flex items-start gap-2 pl-2">
                             <div className="h-1.5 w-1.5 rounded-full bg-indigo-500 mt-2 shrink-0" />
                             <div>{rendered}</div>
                        </div>
                    );
                }
                return <div key={i} className="mb-2 last:mb-0">{rendered}</div>;
            })}
            <style dangerouslySetInnerHTML={{ __html: `
                @keyframes report-char-in {
                    0% { opacity: 0; transform: translateY(8px); filter: blur(1px); }
                    100% { opacity: 1; transform: translateY(0); filter: blur(0); }
                }
                .animate-report-char {
                    animation: report-char-in 0.4s cubic-bezier(0.2, 0.8, 0.2, 1);
                }
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
                {isSummarizing && (
                    <div className="mt-8">
                        <ThinkingLoader />
                    </div>
                )}

                {typeof report === 'string' && report.length > 0 && !isSummarizing && (
                    <div className="mt-8 rounded-2xl bg-gradient-to-br from-indigo-50/50 to-blue-50/50 border border-indigo-100/20 overflow-hidden animate-in fade-in slide-in-from-bottom-2 duration-500 shadow-sm">
                        <div className="bg-white/40 backdrop-blur-sm px-5 py-3 border-b border-indigo-100/20 flex items-center justify-between">
                            <div className="flex items-center gap-2">
                                <span className="bg-indigo-100 text-indigo-700 p-1 rounded-lg text-lg">💡</span>
                                <h4 className="text-sm font-bold text-indigo-900 tracking-tight uppercase">Medical AI Agent Summary</h4>
                            </div>
                            <div className="flex items-center gap-2">
                                <span className="text-[10px] uppercase font-bold text-indigo-500/80 bg-indigo-100/30 px-2.5 py-1 rounded-full">RAG Grounded</span>
                            </div>
                        </div>
                        
                        <div className="p-6 text-sm md:text-base text-slate-800 leading-relaxed min-h-[100px]">
                            <MarkdownLite text={report} isStreaming={isSummarizing} />
                        </div>

                        {sources && sources.length > 0 && (
                             <div className="px-6 pb-6 mt-[-10px]">
                                <div className="flex flex-wrap gap-2 items-center">
                                    <span className="text-[10px] font-bold text-slate-400 uppercase tracking-tighter mr-2">Citations:</span>
                                    {sources.map((s, idx) => (
                                        <span key={idx} className="bg-white/60 border border-slate-100 px-2 py-0.5 rounded text-[9px] text-slate-500 shadow-sm">
                                            {s}
                                        </span>
                                    ))}
                                </div>
                             </div>
                        )}
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
