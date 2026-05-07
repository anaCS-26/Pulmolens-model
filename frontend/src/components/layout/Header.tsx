import React from "react";
import { Stethoscope } from "lucide-react";
import { cn } from "../../utils/cn";
import { Step } from "../../types";

interface HeaderProps {
    step: Step;
    setStep: (s: Step) => void;
    agreed: boolean;
    hasFile: boolean;
}

export function Header({ step, setStep, agreed, hasFile }: HeaderProps) {
    const NavBtn = ({ id, label, disabled }: { id: Step; label: string; disabled?: boolean }) => (
        <button
            onClick={() => !disabled && setStep(id)}
            disabled={disabled}
            className={cn(
                "text-sm px-3 py-1.5 rounded-lg border transition-colors",
                step === id
                    ? "bg-slate-900 text-white border-slate-900"
                    : disabled
                        ? "bg-slate-50 text-slate-300 border-slate-100 cursor-not-allowed"
                        : "bg-white border-slate-300 hover:bg-slate-50"
            )}
        >
            {label}
        </button>
    );

    return (
        <header className="sticky top-0 z-40 backdrop-blur supports-[backdrop-filter]:bg-white/80 border-b border-slate-200">
            <div className="mx-auto max-w-6xl px-4 py-3 flex items-center justify-between">
                <div className="flex items-center gap-3">
                    <div className="h-10 w-10 rounded-xl bg-slate-900 text-white flex items-center justify-center shadow">
                        <Stethoscope className="h-6 w-6" />
                    </div>
                    <div>
                        <div className="text-xl font-semibold tracking-tight">PulmoLens</div>
                        <div className="text-xs text-slate-500">
                            AI-assisted CXR Guidance {import.meta.env.VITE_DEMO_MODE === 'true' ? "(Mock Mode)" : "(Demo)"}
                        </div>
                    </div>
                </div>
                <nav className="flex items-center gap-2">
                    <NavBtn id="landing" label="Home" />
                    <NavBtn id="about" label="About" />
                    <NavBtn id="upload" label="Upload" disabled={!agreed} />
                    <NavBtn id="results" label="Results" disabled={!hasFile} />
                </nav>
            </div>
        </header>
    );
}
