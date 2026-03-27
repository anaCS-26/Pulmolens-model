import React from "react";
import { Printer } from "lucide-react";
import { GuidedCard } from "./GuidedCard";
import { SafetyNet } from "./SafetyNet";

interface ReportPanelProps {
    actionable: { label: string; prob: number }[];
    onPrintClinician: () => void;
    onOpenPatient: () => void;
    report: string | null;
}

export function ReportPanel({ actionable, onPrintClinician, onOpenPatient }: Omit<ReportPanelProps, 'report'>) {
    const hasFindings = actionable.some((a) => a.label !== "No findings");
    return (
        <div>
            <div className="flex items-center justify-between">
                <h3 className="text-lg font-semibold tracking-tight">Structured report</h3>
                <div className="flex items-center gap-2">
                    <button onClick={onPrintClinician} className="rounded-xl border border-slate-300 bg-white px-4 py-2 text-sm flex items-center gap-2" title="Print clinician view">
                        <Printer className="h-4 w-4" />
                        Print
                    </button>
                    <button onClick={onOpenPatient} className="rounded-xl bg-slate-900 text-white px-3 py-2 text-sm" title="Generate patient-friendly summary">
                        Patient-friendly summary
                    </button>
                </div>
            </div>

            <div className="mt-3 space-y-3">
                {!hasFindings && (
                    <div className="rounded-xl border border-emerald-200 bg-emerald-50 text-emerald-900 p-4 text-sm">
                        No acute radiographic abnormality above the current threshold. Correlate with clinical picture.
                    </div>
                )}

                {actionable.filter((a) => a.label !== "No findings").slice(0, 6).map((a) => (
                    <GuidedCard key={a.label} label={a.label} />
                ))}

                <SafetyNet />
            </div>
        </div>
    );
}
