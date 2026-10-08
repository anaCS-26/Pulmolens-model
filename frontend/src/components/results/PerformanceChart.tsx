import React from "react";
import { cn } from "../../utils/cn";
import { prettyLabel } from "../../utils/format";
import { MODEL_PERFORMANCE } from "../../data/constants";

const COLS = "grid grid-cols-[96px_minmax(0,1fr)_44px] items-center gap-3 sm:grid-cols-[160px_minmax(0,1fr)_64px] sm:gap-4";

/** Per-finding recall (thick bar) and precision (thin bar) on the held-out test set. */
export function PerformanceChart({ className }: { className?: string }) {
    return (
        <figure className={className}>
            <div className="mb-3 flex flex-wrap gap-x-6 gap-y-1 text-[13px] text-fg-muted">
                <span className="flex items-center gap-2"><i className="h-2 w-3.5 rounded-sm bg-fg" />Recall</span>
                <span className="flex items-center gap-2"><i className="h-1 w-3.5 rounded-sm bg-fg-muted/70" />Precision</span>
                <span className="ml-auto">Positive cases</span>
            </div>
            <div role="table" aria-label="Recall and precision per finding">
                {MODEL_PERFORMANCE.map((r) => (
                    <div role="row" key={r.label} className={cn(COLS, "border-b border-line-soft py-1.5 text-[13px] sm:text-[14px]")}>
                        <span role="cell" className={cn("truncate", r.recall < 0.45 ? "font-semibold text-fg" : "text-fg-soft")}>{prettyLabel(r.label)}</span>
                        <span role="cell" className="relative h-5" title={`Recall ${r.recall}, precision ${r.precision}`}>
                            <i className="absolute left-0 top-[2px] h-2.5 rounded-sm bg-fg" style={{ width: `${r.recall * 100}%` }} />
                            <i className="absolute bottom-[2px] left-0 h-[5px] rounded-sm bg-fg-muted/70" style={{ width: `${r.precision * 100}%` }} />
                        </span>
                        <span role="cell" className="num text-right text-fg-faint">{r.support.toLocaleString()}</span>
                    </div>
                ))}
            </div>
            <div className={cn(COLS, "pt-2 text-[12px] text-fg-faint")} aria-hidden>
                <span />
                <span className="flex justify-between"><span>0</span><span>0.25</span><span>0.5</span><span>0.75</span><span>1</span></span>
                <span />
            </div>
        </figure>
    );
}
