import React from "react";
import { cn } from "../../utils/cn";
import { CLINICIAN_COPY } from "../../data/constants";

interface ResultRowProps {
    p: { label: string; prob: number };
    threshold: number;
}

export function ResultRow({ p, threshold }: ResultRowProps) {
    const over = p.prob >= threshold;
    return (
        <div className={cn("rounded-xl border px-3 py-2 text-sm flex items-center justify-between", over ? "border-emerald-300 bg-emerald-50 text-emerald-900" : "border-slate-200 bg-white text-slate-700")} title={CLINICIAN_COPY[p.label] || ""}>
            <div className="flex items-center gap-2">
                <span className="font-medium">{p.label}</span>
                <span className="text-xs text-slate-500">{(p.prob * 100).toFixed(0)}%</span>
            </div>
            <div className={cn("text-xs", over ? "text-emerald-700" : "text-slate-400")}>{over ? "above threshold" : "below threshold"}</div>
        </div>
    );
}
