import React, { useState } from "react";
import { AnimatePresence, motion } from "framer-motion";
import { cn } from "../../utils/cn";
import { prettyLabel, pct } from "../../utils/format";
import { CLINICIAN_COPY, GUIDELINE_TAGS } from "../../data/constants";
import { Prediction } from "../../types";
import { EASE_OUT } from "../ui/motion";

interface FindingsTableProps {
    predictions: Prediction[];
    /** Static specimen (landing page): no toggle, no expandable rows. */
    specimen?: boolean;
}

const COLS = "grid grid-cols-[minmax(8.5rem,1.4fr)_44px_minmax(40px,1fr)_44px_12px] items-center gap-x-3 sm:gap-x-4";

export function FindingsTable({ predictions, specimen }: FindingsTableProps) {
    const [flaggedOnly, setFlaggedOnly] = useState(false);
    const [open, setOpen] = useState<string | null>(null);

    const flaggedCount = predictions.filter((p) => p.prob >= p.threshold).length;
    const rows = flaggedOnly ? predictions.filter((p) => p.prob >= p.threshold) : predictions;

    return (
        <div>
            {!specimen && (
                <div className="mb-3 flex items-center gap-4 text-[13px]">
                    <span className="text-ink-muted">Show</span>
                    {[
                        [false, `All (${predictions.length})`],
                        [true, `Flagged (${flaggedCount})`],
                    ].map(([v, l]) => (
                        <button
                            key={String(v)}
                            onClick={() => setFlaggedOnly(v as boolean)}
                            aria-pressed={flaggedOnly === v}
                            className={cn(flaggedOnly === v ? "font-medium text-ink underline underline-offset-[5px]" : "text-ink-muted hover:text-ink")}
                        >
                            {l as string}
                        </button>
                    ))}
                </div>
            )}

            <div className={cn(COLS, "border-y border-ink py-2 label !text-[10.5px]")}>
                <span>Finding</span>
                <span className="text-right">Score</span>
                <span />
                <span className="text-right">Cutoff</span>
                <span />
            </div>

            <ul>
                {rows.map((p, i) => (
                    <Row key={p.label} p={p} index={i} specimen={specimen} open={open === p.label} onToggle={() => setOpen(open === p.label ? null : p.label)} />
                ))}
                {rows.length === 0 && (
                    <li className="border-b border-rule py-6 text-center text-sm text-ink-muted">
                        {predictions.length === 0 ? "No predictions to display yet." : "Nothing is above its cutoff."}
                    </li>
                )}
            </ul>
        </div>
    );
}

function Row({ p, index, open, onToggle, specimen }: { p: Prediction; index: number; open: boolean; onToggle: () => void; specimen?: boolean }) {
    const over = p.prob >= p.threshold;
    const tags = (GUIDELINE_TAGS[p.label] || []).filter((t) => t !== "None");

    const body = (
        <div className={cn(COLS, "py-2.5")}>
            <span className={cn("flex min-w-0 items-center gap-2 truncate text-[14px]", over ? "font-medium text-ink" : "text-ink-soft")}>
                {over && <span className="h-2 w-2 shrink-0 bg-marker" aria-label="Above cutoff" />}
                {prettyLabel(p.label)}
            </span>
            <span className={cn("num text-right font-mono text-[13px]", over ? "text-marker" : "text-ink-soft")}>{pct(p.prob)}</span>
            <span className="relative h-[5px] bg-paper-sunk">
                <motion.span
                    className={cn("absolute inset-y-0 left-0", over ? "bg-marker" : "bg-ink/45")}
                    initial={{ width: 0 }}
                    whileInView={{ width: `${Math.max(0.6, p.prob * 100)}%` }}
                    viewport={{ once: true }}
                    transition={{ duration: 0.7, delay: Math.min(index, 12) * 0.03, ease: EASE_OUT }}
                />
                <span className="absolute -bottom-[3px] -top-[3px] w-px bg-ink" style={{ left: `${p.threshold * 100}%` }} />
            </span>
            <span className="num text-right font-mono text-[12px] text-ink-muted">{pct(p.threshold)}</span>
            <span className="font-mono text-[13px] text-ink-faint">{specimen ? "" : open ? "−" : "+"}</span>
        </div>
    );

    if (specimen) return <li className="border-b border-rule">{body}</li>;

    return (
        <li className="border-b border-rule">
            <button onClick={onToggle} aria-expanded={open} className="block w-full text-left transition-colors hover:bg-paper-raised">
                {body}
            </button>
            <AnimatePresence initial={false}>
                {open && (
                    <motion.div
                        initial={{ height: 0, opacity: 0 }}
                        animate={{ height: "auto", opacity: 1 }}
                        exit={{ height: 0, opacity: 0 }}
                        transition={{ duration: 0.25, ease: EASE_OUT }}
                        className="overflow-hidden"
                    >
                        <div className="max-w-[60ch] pb-4 pl-4 text-[13.5px] leading-relaxed text-ink-soft">
                            <p>{CLINICIAN_COPY[p.label] || "No reference notes for this finding."}</p>
                            <p className="mt-2 font-mono text-[11.5px] text-ink-muted">
                                Score {pct(p.prob, 1)} against a cutoff of {pct(p.threshold, 1)}
                                {tags.length > 0 && <> · {tags.join("; ")}</>}
                            </p>
                        </div>
                    </motion.div>
                )}
            </AnimatePresence>
        </li>
    );
}
