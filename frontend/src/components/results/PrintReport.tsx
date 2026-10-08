import React from "react";
import { createPortal } from "react-dom";
import { Prediction } from "../../types";
import { prettyLabel, pct } from "../../utils/format";
import { CLINICIAN_COPY } from "../../data/constants";
import { DEFAULT_GUIDANCE, GUIDANCE_BULLETS, SAFETY_NET } from "../../data/guidance";
import { PatientSummaryBody } from "./PatientSummarySheet";

export type PrintMode = "clinician" | "patient";

interface PrintReportProps {
    mode: PrintMode;
    fileName: string;
    predictions: Prediction[];
    flagged: Prediction[];
    report: string | null;
    sources: string[];
}

/** Paper layout that only exists for window.print(); the dark UI is hidden in print. */
export function PrintReport({ mode, fileName, predictions, flagged, report, sources }: PrintReportProps) {
    const date = new Date().toLocaleString();
    return createPortal(
        <div className="hidden print:block bg-white font-sans text-[11pt] leading-snug text-zinc-900">
            <header className="flex items-end justify-between border-b-2 border-zinc-900 pb-3">
                <div>
                    <div className="text-[18pt] font-semibold">PulmoLens · {mode === "clinician" ? "Structured report" : "Patient summary"}</div>
                    <div className="mt-1 text-[9pt] text-zinc-500">{fileName} · {date}</div>
                </div>
                <div className="text-right text-[8pt] uppercase tracking-wider text-zinc-500">Educational prototype<br />Not a diagnostic device</div>
            </header>

            {mode === "patient" ? (
                <div className="mt-6"><PatientSummaryBody findings={flagged} /></div>
            ) : (
                <>
                    <h2 className="mt-6 text-[12pt] font-semibold">Model output</h2>
                    <table className="mt-2 w-full border-collapse text-[10pt]">
                        <thead>
                            <tr className="border-b border-zinc-300 text-left text-zinc-500">
                                <th className="py-1 font-medium">Finding</th>
                                <th className="py-1 text-right font-medium">Probability</th>
                                <th className="py-1 text-right font-medium">Threshold</th>
                                <th className="py-1 text-right font-medium">Status</th>
                            </tr>
                        </thead>
                        <tbody>
                            {predictions.map((p) => {
                                const over = p.prob >= p.threshold;
                                return (
                                    <tr key={p.label} className="border-b border-zinc-100">
                                        <td className={`py-1 ${over ? "font-semibold" : ""}`}>{prettyLabel(p.label)}</td>
                                        <td className="py-1 text-right font-mono">{pct(p.prob, 1)}</td>
                                        <td className="py-1 text-right font-mono text-zinc-500">{pct(p.threshold, 1)}</td>
                                        <td className="py-1 text-right">{over ? "Flagged" : "—"}</td>
                                    </tr>
                                );
                            })}
                        </tbody>
                    </table>

                    {report && (
                        <>
                            <h2 className="mt-6 text-[12pt] font-semibold">Clinical synthesis (AI-generated)</h2>
                            <div className="mt-2 whitespace-pre-wrap text-[10.5pt]">{report.replace(/\*\*/g, "").replace(/^\s*\*\s+/gm, "• ")}</div>
                        </>
                    )}

                    <h2 className="mt-6 text-[12pt] font-semibold">Guidance for flagged findings</h2>
                    {flagged.length === 0 ? (
                        <p className="mt-2 text-[10.5pt]">No acute radiographic abnormality above the current threshold. Correlate with clinical picture.</p>
                    ) : (
                        flagged.map((f) => (
                            <div key={f.label} className="mt-3 break-inside-avoid">
                                <div className="font-semibold">{prettyLabel(f.label)} <span className="font-normal text-zinc-500">({pct(f.prob)})</span></div>
                                <div className="text-[9.5pt] text-zinc-600">{CLINICIAN_COPY[f.label]}</div>
                                <ul className="mt-1 list-disc pl-5 text-[10pt]">
                                    {(GUIDANCE_BULLETS[f.label] || DEFAULT_GUIDANCE).map((b) => <li key={b}>{b}</li>)}
                                </ul>
                            </div>
                        ))
                    )}

                    <h2 className="mt-6 text-[12pt] font-semibold">Safety-net advice</h2>
                    <ul className="mt-1 list-disc pl-5 text-[10pt]">
                        {SAFETY_NET.map((s) => <li key={s}>{s}</li>)}
                    </ul>

                    {sources.length > 0 && (
                        <>
                            <h2 className="mt-6 text-[12pt] font-semibold">References</h2>
                            <ol className="mt-1 list-decimal pl-5 text-[9.5pt] text-zinc-700">
                                {sources.map((s, i) => <li key={i}>{s}</li>)}
                            </ol>
                        </>
                    )}
                </>
            )}

            <footer className="mt-8 border-t border-zinc-300 pt-3 text-[8pt] text-zinc-500">
                Decision support only. Not medical advice and not for diagnostic or clinical decision-making. Always consult a
                qualified healthcare professional.
            </footer>
        </div>,
        document.body
    );
}
