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
                <p className="mt-2 text-sm text-slate-600">Running your model on the server…</p>
            </div>
        </section>
    );
}
