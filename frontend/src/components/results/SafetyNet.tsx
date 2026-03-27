import React from "react";
import { AlertOctagon } from "lucide-react";

export function SafetyNet() {
    return (
        <div className="rounded-2xl border border-rose-200 bg-rose-50 p-4">
            <div className="flex items-center gap-2 text-rose-900 font-semibold">
                <AlertOctagon className="h-4 w-4" /> Safety‑net advice
            </div>
            <ul className="mt-2 text-sm text-rose-900 list-disc pl-5 space-y-1">
                <li>Severe breathlessness, chest pain, haemoptysis, confusion, or cyanosis → urgent medical attention.</li>
                <li>If symptoms worsen or fail to improve as expected, arrange prompt clinical review.</li>
            </ul>
            <div className="mt-2 text-xs text-rose-900/80">Educational prototype — not a diagnostic device.</div>
        </div>
    );
}
