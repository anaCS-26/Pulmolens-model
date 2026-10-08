import React from "react";
import { SAFETY_NET } from "../../data/guidance";

export function SafetyNet() {
    return (
        <div className="border-l-2 border-marker pl-5">
            <ul className="space-y-1.5 text-[14.5px]">
                {SAFETY_NET.map((s) => (
                    <li key={s}>{s}</li>
                ))}
            </ul>
            <p className="mt-2 font-mono text-[11.5px] text-ink-muted">Educational prototype. Not a diagnostic device.</p>
        </div>
    );
}
