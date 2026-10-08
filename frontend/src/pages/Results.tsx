import React, { useEffect, useMemo, useState } from "react";
import { flushSync } from "react-dom";
import { AnimatePresence } from "framer-motion";
import { prettyLabel, pct } from "../utils/format";
import { submitFeedback } from "../api";
import { Prediction } from "../types";
import { Button } from "../components/ui/Button";
import { FlowProgress } from "../components/layout/FlowProgress";
import { XrayViewer, ViewMode } from "../components/viewer/XrayViewer";
import { FilmPanel } from "../components/viewer/FilmPanel";
import { Lightbox } from "../components/viewer/Lightbox";
import { FindingsTable } from "../components/results/FindingsList";
import { ClinicalReport } from "../components/results/ClinicalReport";
import { GuidedCard } from "../components/results/GuidedCard";
import { SafetyNet } from "../components/results/SafetyNet";
import { FeedbackBar } from "../components/results/FeedbackBar";
import { PatientSummarySheet } from "../components/results/PatientSummarySheet";
import { PrintReport, PrintMode } from "../components/results/PrintReport";

interface ResultsProps {
    file: File | null;
    imageURL: string | null;
    predictions: Prediction[];
    onRestart: () => void;
    onRetry: () => void;
    errorMsg: string | null;
    attentionOverlay: string | null;
    report: string | null;
    sources: string[];
    isSummarizing?: boolean;
}

const COUNT_WORDS = ["No", "One", "Two", "Three", "Four", "Five", "Six", "Seven", "Eight", "Nine"];
const countWord = (n: number) => COUNT_WORDS[n] ?? String(n);

function Block({ n, title, aside, children }: { n: number; title: string; aside?: React.ReactNode; children: React.ReactNode }) {
    return (
        <section className="mb-14">
            <h2 className="mb-5 flex items-baseline justify-between gap-4 border-t border-ink pt-3">
                <span className="label !text-ink"><span className="mr-3 text-ink-faint">{n}</span>{title}</span>
                {aside}
            </h2>
            {children}
        </section>
    );
}

export function Results({
    file,
    imageURL,
    predictions,
    onRestart,
    onRetry,
    errorMsg,
    attentionOverlay,
    report,
    sources,
    isSummarizing,
}: ResultsProps) {
    const title = file ? file.name : "demo_cxr.jpg";
    const [mode, setMode] = useState<ViewMode>("overlay");
    const [opacity, setOpacity] = useState(0.6);
    const [fullscreen, setFullscreen] = useState(false);
    const [showPatientSheet, setShowPatientSheet] = useState(false);
    const [printMode, setPrintMode] = useState<PrintMode>("clinician");
    const [readAt] = useState(() => new Date());

    const flagged = useMemo(() => predictions.filter((p) => p.label !== "No findings" && p.prob >= p.threshold), [predictions]);
    // The overlay is drawn for the highest-probability class.
    const top = predictions[0];

    const print = (m: PrintMode) => {
        flushSync(() => setPrintMode(m));
        window.print();
    };

    // "F" opens the fullscreen viewer when focus isn't in a text field.
    useEffect(() => {
        const onKey = (e: KeyboardEvent) => {
            const el = e.target as HTMLElement;
            if (el.tagName === "INPUT" || el.tagName === "TEXTAREA" || e.metaKey || e.ctrlKey) return;
            if (e.key.toLowerCase() === "f" && imageURL && !fullscreen && !showPatientSheet) setFullscreen(true);
        };
        window.addEventListener("keydown", onKey);
        return () => window.removeEventListener("keydown", onKey);
    }, [imageURL, fullscreen, showPatientSheet]);

    const handleFeedback = async (rating: "good" | "bad") => {
        if (!file) throw new Error("No file to submit with feedback");
        const preds = Object.fromEntries(predictions.map((p) => [p.label, p.prob]));
        await submitFeedback(file, rating, preds);
    };

    const meta = [title, file ? `${(file.size / 1024).toFixed(0)} KB` : null, readAt.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })]
        .filter(Boolean)
        .join(" · ");

    return (
        <section className="mx-auto max-w-[1200px] px-6 pb-24 pt-12">
            <FlowProgress current={errorMsg ? 2 : 4} />

            <div className="mt-6 flex flex-col gap-6 md:flex-row md:items-end md:justify-between">
                <div className="min-w-0">
                    <h1 className="font-serif text-[44px] font-normal leading-none tracking-[-0.02em] md:text-[56px]">Report</h1>
                    <div className="mt-3 truncate font-mono text-[12px] text-ink-muted">{meta}</div>
                </div>
                <div className="flex flex-wrap items-center gap-5">
                    <button onClick={onRestart} className="link text-sm">New analysis</button>
                    <Button variant="outline" onClick={() => print("clinician")} disabled={!!errorMsg}>Print report</Button>
                    <Button variant="primary" onClick={() => setShowPatientSheet(true)} disabled={!!errorMsg}>Patient summary</Button>
                </div>
            </div>

            {errorMsg ? (
                <div className="mt-10 grid grid-cols-1 gap-x-12 gap-y-10 border-t border-ink pt-8 lg:grid-cols-12">
                    <div className="lg:col-span-6">
                        <p className="font-serif text-[30px] leading-tight">The analysis didn't complete.</p>
                        <p className="mt-3 max-w-[48ch] text-[15px] leading-relaxed text-ink-soft">
                            The model server couldn't be reached or returned an error. PulmoLens never substitutes mock results, so
                            nothing is shown.
                        </p>
                        <pre className="mt-5 max-h-32 overflow-auto whitespace-pre-wrap border-l-2 border-marker bg-paper-raised px-4 py-3 font-mono text-[12px] text-marker-dark">{errorMsg}</pre>
                        <div className="mt-6 flex flex-wrap items-center gap-5">
                            {file && <Button variant="primary" onClick={onRetry}>Try again</Button>}
                            <button onClick={onRestart} className="link text-sm">Choose another image</button>
                        </div>
                    </div>
                    {imageURL && (
                        <FilmPanel className="aspect-[4/3] lg:col-span-6" tl={title} bl="Not analysed">
                            <img src={imageURL} alt="Uploaded chest X-ray" className="h-full w-full object-contain opacity-60" />
                        </FilmPanel>
                    )}
                </div>
            ) : (
                <>
                    {/* Impression */}
                    <div className="mt-10 grid grid-cols-1 gap-x-12 gap-y-3 border-t border-ink pt-5 md:grid-cols-12">
                        <div className="label !text-ink md:col-span-3">Impression</div>
                        <p className="font-serif text-[24px] leading-[1.35] md:col-span-9 md:text-[28px]">
                            {flagged.length === 0 ? (
                                <>
                                    No finding scored above its cutoff.
                                    {top && <> The highest was {prettyLabel(top.label).toLowerCase()}, at <span className="num font-mono text-[0.8em]">{pct(top.prob)}</span>.</>}
                                </>
                            ) : (
                                <>
                                    {countWord(flagged.length)} {flagged.length === 1 ? "finding" : "findings"} above cutoff:{" "}
                                    {flagged.map((f, i) => (
                                        <React.Fragment key={f.label}>
                                            {i > 0 && (i === flagged.length - 1 ? " and " : ", ")}
                                            {prettyLabel(f.label).toLowerCase()}{" "}
                                            <span className="num font-mono text-[0.8em] text-marker">{pct(f.prob)}</span>
                                        </React.Fragment>
                                    ))}
                                    .
                                </>
                            )}
                        </p>
                    </div>

                    <div className="mt-12 grid grid-cols-1 gap-x-12 gap-y-12 lg:grid-cols-12">
                        <div className="lg:sticky lg:top-20 lg:col-span-5 lg:self-start">
                            <XrayViewer
                                imageURL={imageURL}
                                overlay={attentionOverlay}
                                overlayFinding={top?.label}
                                fileName={title}
                                mode={mode}
                                setMode={setMode}
                                opacity={opacity}
                                setOpacity={setOpacity}
                                onExpand={() => setFullscreen(true)}
                            />
                            <div className="mt-8">
                                <FeedbackBar onSubmit={handleFeedback} disabled={!file} />
                            </div>
                        </div>

                        <div className="lg:col-span-7">
                            <Block n={1} title="Findings">
                                <FindingsTable predictions={predictions} />
                            </Block>

                            <Block
                                n={2}
                                title="Synthesis"
                                aside={isSummarizing ? <span className="font-mono text-[11px] text-marker">streaming<span className="animate-caret-blink">_</span></span> : <span className="font-mono text-[11px] text-ink-muted">AI-generated</span>}
                            >
                                <ClinicalReport report={report} sources={sources} isSummarizing={isSummarizing} hasOverlay={!!attentionOverlay} />
                            </Block>

                            <Block n={3} title="Guidance">
                                {flagged.length === 0 ? (
                                    <p className="text-[14.5px] text-ink-soft">No acute radiographic abnormality above the current threshold. Correlate with clinical picture.</p>
                                ) : (
                                    flagged.slice(0, 6).map((a) => <GuidedCard key={a.label} label={a.label} prob={a.prob} />)
                                )}
                            </Block>

                            <Block n={4} title="Safety net">
                                <SafetyNet />
                            </Block>
                        </div>
                    </div>
                </>
            )}

            <AnimatePresence>
                {fullscreen && imageURL && (
                    <Lightbox title={title} imageURL={imageURL} overlay={attentionOverlay} opacity={opacity} setOpacity={setOpacity} onClose={() => setFullscreen(false)} />
                )}
            </AnimatePresence>
            <AnimatePresence>
                {showPatientSheet && (
                    <PatientSummarySheet findings={flagged} onClose={() => setShowPatientSheet(false)} onPrint={() => print("patient")} />
                )}
            </AnimatePresence>
            {!errorMsg && (
                <PrintReport mode={printMode} fileName={title} predictions={predictions} flagged={flagged} report={report} sources={sources} />
            )}
        </section>
    );
}
