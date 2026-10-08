import React from "react";
import { SAFETY_NET } from "../../data/guidance";

/** Urgent-care advice. The only place red is used outside the attention map. */
export function SafetyNet() {
    return (
        <div className="rounded-lg border border-urgent/40 bg-urgent/[0.04] px-5 py-4">
            <h3 className="flex items-center gap-2.5 text-[15px] font-semibold">
                <span className="h-2 w-2 rounded-full bg-urgent shadow-[0_0_0_4px_rgb(var(--urgent)/0.18)]" aria-hidden />
                Get urgent care first if
            </h3>
            <ul className="mt-2.5 space-y-1.5 pl-[18px] text-[14.5px] text-fg-soft">
                {SAFETY_NET.map((s) => (
                    <li key={s}>{s}</li>
                ))}
            </ul>
            <p className="mt-3 pl-[18px] text-[12.5px] text-fg-muted">Educational prototype. Not a diagnostic device.</p>
        </div>
    );
}
