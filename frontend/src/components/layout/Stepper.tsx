import React from "react";
import { CheckCircle2, Gauge, ChevronRight } from "lucide-react";
import { cn } from "../../utils/cn";
import { Step } from "../../types";

interface StepperProps {
    step: Step;
    setStep: (s: Step) => void;
    agreed: boolean;
    hasFile: boolean;
}

export function Stepper({ step, setStep, agreed, hasFile }: StepperProps) {
    const items: { id: Step; label: string }[] = [
        { id: "landing", label: "Intro" },
        { id: "consent", label: "Consent" },
        { id: "upload", label: "Upload" },
        { id: "processing", label: "Processing" },
        { id: "results", label: "Results" },
    ];
    const idx = items.findIndex((i) => i.id === step);

    const isStepDisabled = (id: Step) => {
        if (id === "upload" && !agreed) return true;
        if ((id === "processing" || id === "results") && !hasFile) return true;
        return false;
    };

    return (
        <div className="mt-6 mb-6">
            <ol className="grid grid-cols-5 gap-2">
                {items.map((it, i) => {
                    const disabled = isStepDisabled(it.id);
                    return (
                        <li key={it.id}>
                            <button
                                onClick={() => !disabled && setStep(it.id)}
                                disabled={disabled}
                                className={cn(
                                    "w-full group flex items-center gap-2 rounded-2xl border px-3 py-2 text-sm transition-colors",
                                    i < idx
                                        ? "bg-emerald-50 border-emerald-200 text-emerald-700"
                                        : i === idx
                                            ? "bg-white border-slate-300 shadow text-slate-900"
                                            : disabled
                                                ? "bg-slate-50 border-slate-200 text-slate-300 cursor-not-allowed"
                                                : "bg-slate-50 border-slate-200 text-slate-500 hover:bg-slate-100"
                                )}
                            >
                                <span className="flex-1 text-left">{it.label}</span>
                                {i < idx ? (
                                    <CheckCircle2 className="h-4 w-4" />
                                ) : i === idx ? (
                                    <Gauge className="h-4 w-4" />
                                ) : (
                                    <ChevronRight className="h-4 w-4" />
                                )}
                            </button>
                        </li>
                    );
                })}
            </ol>
        </div>
    );
}
