import React from "react";
import { Loader2 } from "lucide-react";

interface ProcessingProps {
    progress: number;
}

export function Processing({ progress }: ProcessingProps) {
    return (
        <section className="mt-8">
            <div className="p-6 rounded-3xl bg-white shadow-sm border border-slate-200">
                <div className="flex items-center gap-3">
                    <Loader2 className="h-5 w-5 animate-spin text-slate-500" />
                    <h2 className="text-2xl font-semibold tracking-tight">Analysing image…</h2>
                </div>
                <div className="mt-4 w-full h-3 rounded-full bg-slate-100 overflow-hidden">
                    <div className="h-full bg-slate-900 transition-all" style={{ width: `${progress}%` }} />
                </div>
                <p className="mt-2 text-sm text-slate-600">
                    {import.meta.env.VITE_DEMO_MODE === 'true' ? "Simulating analysis pipeline..." : "Running your model on the server..."}
                </p>
                {progress > 60 && import.meta.env.VITE_DEMO_MODE !== 'true' && (
                    <p className="mt-4 text-xs text-amber-600 animate-pulse bg-amber-50 p-2 rounded-lg border border-amber-100 flex items-center gap-2">
                        <span>⚠️</span>
                        The server is currently warming up (this happens if it hasn't been used recently). Please stay on this screen.
                    </p>
                )}
            </div>
        </section>
    );
}
