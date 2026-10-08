import React from "react";
import { cn } from "../../utils/cn";

const STAGES = ["Consent", "Image", "Analysis", "Report"] as const;

/** Breadcrumb-style progress: "1 Consent ✓ — 2 Image — …". */
export function FlowProgress({ current }: { current: number }) {
    return (
        <ol className="flex flex-wrap items-center gap-x-3 gap-y-1 font-mono text-[12px] print:hidden" aria-label="Progress">
            {STAGES.map((label, i) => {
                const done = i < current;
                const active = i === current;
                return (
                    <li key={label} className="flex items-center gap-3" aria-current={active ? "step" : undefined}>
                        <span className={cn(active ? "text-ink" : done ? "text-ink-muted" : "text-ink-faint")}>
                            {i + 1} {label}
                            {done && <span className="ml-1.5 text-ink-muted">✓</span>}
                        </span>
                        {i < STAGES.length - 1 && <span className={cn("h-px w-6", done ? "bg-ink-muted" : "bg-rule-strong")} />}
                    </li>
                );
            })}
        </ol>
    );
}
