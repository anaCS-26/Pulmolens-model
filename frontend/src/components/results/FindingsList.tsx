import React, { useState } from "react";
import { AnimatePresence, motion } from "framer-motion";
import { ChevronDown } from "lucide-react";
import { cn } from "../../utils/cn";
import { prettyLabel, pct } from "../../utils/format";
import { CLINICIAN_COPY, GUIDELINE_TAGS, LOW_RECALL_NOTE } from "../../data/constants";
import { Prediction } from "../../types";
import { Tabs } from "../ui/Tabs";
import { EASE_OUT } from "../ui/motion";

interface FindingsTableProps {
    predictions: Prediction[];
    /** Static specimen (landing page): no filter, no expandable rows. */
    specimen?: boolean;
    className?: string;
}

const COLS = "grid grid-cols-[minmax(0,1fr)_44px_minmax(56px,110px)_16px] items-center gap-x-3 sm:gap-x-4";

export function FindingsTable({ predictions, specimen, className }: FindingsTableProps) {
    const [flaggedOnly, setFlaggedOnly] = useState(false);
    const [open, setOpen] = useState<string | null>(null);

    const flaggedCount = predictions.filter((p) => p.prob >= p.threshold).length;
    const rows = flaggedOnly ? predictions.filter((p) => p.prob >= p.threshold) : predictions;

    return (
        <div className={className}>
            {!specimen && (
                <Tabs<"all" | "flagged">
                    label="Show findings"
                    className="mb-4"
                    value={flaggedOnly ? "flagged" : "all"}
                    onChange={(v) => setFlaggedOnly(v === "flagged")}
                    options={[
                        { value: "all", label: `All ${predictions.length}` },
                        { value: "flagged", label: `Above cutoff ${flaggedCount}` },
                    ]}
                />
            )}

            <div className={cn(COLS, "border-b border-line pb-2 text-[12px] text-fg-faint")}>
                <span>Finding</span>
                <span className="text-right">Score</span>
                <span>Against cutoff</span>
                <span />
            </div>

            <ul>
                {rows.map((p, i) => (
                    <Row key={p.label} p={p} index={i} specimen={specimen} open={open === p.label} onToggle={() => setOpen(open === p.label ? null : p.label)} />
                ))}
                {rows.length === 0 && (
                    <li className="border-b border-line-soft py-6 text-center text-[14px] text-fg-muted">
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
    const caveat = !over ? LOW_RECALL_NOTE[p.label] : undefined;

    const body = (
        <>
            <div className={cn(COLS, "py-2.5")}>
                <span className={cn("flex min-w-0 items-center gap-2 text-[14px]", over ? "font-semibold text-fg" : "text-fg-soft")}>
                    {over && <span className="h-1.5 w-1.5 shrink-0 rounded-full bg-accent" aria-label="Above cutoff" />}
                    <span className="truncate">{prettyLabel(p.label)}</span>
                </span>
                <span className={cn("num text-right font-mono text-[12.5px] [font-stretch:80%]", over ? "text-accent-ink" : "text-fg-muted")}>{pct(p.prob)}</span>
                <span className="relative h-1.5 rounded-full bg-line-soft">
                    <motion.span
                        className={cn("absolute inset-y-0 left-0 rounded-full", over ? "bg-accent" : "bg-bar")}
                        initial={{ width: 0 }}
                        whileInView={{ width: `${Math.max(1, p.prob * 100)}%` }}
                        viewport={{ once: true }}
                        transition={{ duration: 0.8, delay: Math.min(index, 12) * 0.03, ease: EASE_OUT }}
                    />
                    <span className="absolute -bottom-1 -top-1 w-[1.5px] bg-fg/75" style={{ left: `${p.threshold * 100}%` }} title={`Cutoff ${pct(p.threshold)}`} />
                </span>
                {specimen ? <span /> : <ChevronDown className={cn("h-3.5 w-3.5 text-fg-faint transition-transform", open && "rotate-180")} />}
            </div>
            {caveat && <p className="-mt-1 pb-2.5 text-[12.5px] text-fg-muted">{caveat}</p>}
        </>
    );

    const rowClass = cn("border-b border-line-soft", over && "-mx-3 bg-gradient-to-r from-accent/10 to-transparent px-3");

    if (specimen) return <li className={rowClass}>{body}</li>;

    return (
        <li className={rowClass}>
            <button onClick={onToggle} aria-expanded={open} className="block w-full text-left">
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
                        <div className="max-w-[60ch] pb-4 text-[13.5px] leading-relaxed text-fg-soft">
                            <p>{CLINICIAN_COPY[p.label] || "No reference notes for this finding."}</p>
                            <p className="mt-2 text-[12.5px] text-fg-muted">
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
