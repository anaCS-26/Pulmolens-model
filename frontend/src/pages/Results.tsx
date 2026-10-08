import React, { useEffect, useMemo, useState } from "react";
import { flushSync } from "react-dom";
import { AnimatePresence } from "framer-motion";
import { Printer } from "lucide-react";
import { cn } from "../utils/cn";
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

function Card({ title, aside, children, className }: { title: string; aside?: React.ReactNode; children: React.ReactNode; className?: string }) {
    return (
        <section className={cn("rounded-xl border border-line bg-surface-raised", className)}>
            <h2 className="flex items-center justify-between gap-4 border-b border-line px-5 py-3.5 text-[15px] font-semibold sm:px-6">
                {title}
                {aside}
            </h2>
            <div className="px-5 py-5 sm:px-6">{children}</div>
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
        <section className="mx-auto max-w-[1280px] px-4 pb-24 pt-10 sm:px-6 lg:px-10">
            <FlowProgress current={errorMsg ? 2 : 4} />

            <div className="mt-7 flex flex-col gap-6 md:flex-row md:items-end md:justify-between">
                <div className="min-w-0">
                    <h1 className="display text-[40px] leading-none md:text-[54px]">Report</h1>
                    <div className="mt-3 truncate text-[13.5px] text-fg-muted">{meta}</div>
                </div>
                <div className="flex flex-wrap items-center gap-3">
                    <Button variant="text" onClick={onRestart} className="mr-2 text-[14px]">New analysis</Button>
                    <Button variant="outline" onClick={() => print("clinician")} disabled={!!errorMsg}><Printer className="h-4 w-4" /> Print report</Button>
                    <Button variant="primary" onClick={() => setShowPatientSheet(true)} disabled={!!errorMsg}>Patient summary</Button>
                </div>
            </div>

            {errorMsg ? (
                <div className="mt-10 grid grid-cols-1 gap-x-12 gap-y-10 lg:grid-cols-12">
                    <div className="lg:col-span-6">
                        <p className="display text-[30px] leading-tight">The analysis didn't complete.</p>
                        <p className="mt-3 max-w-[48ch] text-[15.5px] leading-relaxed text-fg-soft">
                            The model server couldn't be reached or returned an error. PulmoLens never substitutes mock results, so
                            nothing is shown.
                        </p>
                        <pre className="mt-5 max-h-32 overflow-auto whitespace-pre-wrap rounded-lg border border-urgent/40 bg-urgent/[0.05] px-4 py-3 font-mono text-[12px] text-urgent">{errorMsg}</pre>
                        <div className="mt-6 flex flex-wrap items-center gap-5">
                            {file && <Button variant="primary" onClick={onRetry}>Try again</Button>}
                            <button onClick={onRestart} className="link text-[14px]">Choose another image</button>
                        </div>
                    </div>
                    {imageURL && (
                        <FilmPanel className="aspect-square lg:col-span-6" tl={title} bl="Not analysed">
                            <img src={imageURL} alt="Uploaded chest X-ray" className="h-full w-full object-contain opacity-60" />
                        </FilmPanel>
                    )}
                </div>
            ) : (
                <>
                    {/* Workstation: film on a black stage, impression and findings beside it */}
                    <div className="mt-10 grid grid-cols-1 overflow-hidden rounded-xl border border-line bg-surface-raised lg:grid-cols-[minmax(0,1fr)_minmax(0,470px)]">
                        <div className="flex items-center bg-film p-4 sm:p-5">
                            <XrayViewer
                                tone="film"
                                className="mx-auto w-full max-w-[620px]"
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
                        </div>
                        <div className="flex min-w-0 flex-col border-t border-line lg:border-l lg:border-t-0">
                            <div className="border-b border-line px-5 py-5 sm:px-6">
                                <div className="text-[12.5px] text-fg-faint">Impression</div>
                                <p className="mt-1.5 text-[20px] font-medium leading-snug md:text-[22px]">
                                    {flagged.length === 0 ? (
                                        <>
                                            No finding scored above its cutoff.
                                            {top && <> The highest was {prettyLabel(top.label).toLowerCase()}, at <span className="num">{pct(top.prob)}</span>.</>}
                                        </>
                                    ) : (
                                        <>
                                            {countWord(flagged.length)} {flagged.length === 1 ? "finding" : "findings"} above cutoff:{" "}
                                            {flagged.map((f, i) => (
                                                <React.Fragment key={f.label}>
                                                    {i > 0 && (i === flagged.length - 1 ? " and " : ", ")}
                                                    <span className="font-semibold text-accent-ink">{prettyLabel(f.label).toLowerCase()}</span>{" "}
                                                    <span className="num text-[0.8em] text-fg-muted">{pct(f.prob)}</span>
                                                </React.Fragment>
                                            ))}
                                            .
                                        </>
                                    )}
                                </p>
                            </div>
                            <div className="px-5 py-5 sm:px-6">
                                <FindingsTable predictions={predictions} />
                            </div>
                        </div>
                    </div>

                    <div className="mt-6 grid grid-cols-1 gap-6 lg:grid-cols-12 lg:items-start">
                        <Card
                            title="Summary"
                            className="lg:col-span-7"
                            aside={
                                isSummarizing ? (
                                    <span className="flex items-center gap-2 text-[12.5px] font-normal text-accent-ink">
                                        <span className="h-1.5 w-1.5 animate-pulse rounded-full bg-accent" /> Streaming
                                    </span>
                                ) : (
                                    <span className="text-[12.5px] font-normal text-fg-muted">AI-generated</span>
                                )
                            }
                        >
                            <ClinicalReport report={report} sources={sources} isSummarizing={isSummarizing} hasOverlay={!!attentionOverlay} />
                        </Card>

                        <div className="space-y-6 lg:col-span-5">
                            <Card title="Next steps" aside={<span className="text-[12.5px] font-normal text-fg-muted">{flagged.length} flagged</span>}>
                                {flagged.length === 0 ? (
                                    <p className="text-[14.5px] text-fg-soft">No acute radiographic abnormality above the current threshold. Correlate with the clinical picture.</p>
                                ) : (
                                    flagged.slice(0, 6).map((a) => <GuidedCard key={a.label} label={a.label} prob={a.prob} />)
                                )}
                            </Card>
                            <SafetyNet />
                        </div>
                    </div>

                    <div className="mt-6 rounded-xl border border-line px-5 py-4 sm:px-6">
                        <FeedbackBar onSubmit={handleFeedback} disabled={!file} />
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
