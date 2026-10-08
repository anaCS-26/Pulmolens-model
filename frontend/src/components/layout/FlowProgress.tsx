import React from "react";
import { Check } from "lucide-react";
import { cn } from "../../utils/cn";

const STAGES = ["Consent", "Image", "Analysis", "Report"] as const;

/** Four-stage stepper. The current stage is amber; finished ones get a tick. */
export function FlowProgress({ current }: { current: number }) {
    return (
        <ol className="flex flex-wrap items-center gap-x-3 gap-y-2 text-[13px] print:hidden" aria-label="Progress">
            {STAGES.map((label, i) => {
                const done = i < current;
                const active = i === current;
                return (
                    <li key={label} className="flex items-center gap-3" aria-current={active ? "step" : undefined}>
                        <span className={cn("flex items-center gap-2", active ? "text-fg" : done ? "text-fg-muted" : "text-fg-faint")}>
                            <span
                                className={cn(
                                    "grid h-5 w-5 place-items-center rounded-full border text-[11px] num",
                                    active ? "border-accent bg-accent text-black" : done ? "border-line bg-surface-sunk" : "border-line"
                                )}
                            >
                                {done ? <Check className="h-3 w-3" strokeWidth={2.5} /> : i + 1}
                            </span>
                            {label}
                        </span>
                        {i < STAGES.length - 1 && <span className={cn("h-px w-6", done ? "bg-fg-muted" : "bg-line")} />}
                    </li>
                );
            })}
        </ol>
    );
}
