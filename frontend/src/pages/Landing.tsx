import React from "react";
import { Brain, ChevronRight, Info, ShieldCheck, FileText, Image as ImageIcon, Stethoscope } from "lucide-react";

interface LandingProps {
    onStart: () => void;
    onLearnMore: () => void;
}

export function Landing({ onStart, onLearnMore }: LandingProps) {
    return (
        <section className="mt-8 grid grid-cols-1 md:grid-cols-2 gap-8">
            <div className="p-6 rounded-3xl bg-white shadow-sm border border-slate-200">
                <div className="inline-flex items-center gap-2 rounded-full bg-slate-900 text-white px-3 py-1 text-xs">
                    <Brain className="h-3.5 w-3.5" /> Demo prototype
                </div>
                <h1 className="text-3xl md:text-4xl font-bold tracking-tight mt-4">
                    Clinician-first chest X-ray guidance
                </h1>
                <p className="text-slate-600 mt-3 text-base leading-relaxed">
                    Upload a de-identified chest X-ray to see structured, UK-guideline-aligned next steps in seconds.
                    This is an educational demo. It is not intended for diagnosis or patient care.
                </p>
                <div className="mt-6 flex items-center gap-3">
                    <button
                        onClick={onStart}
                        className="inline-flex items-center gap-2 rounded-2xl bg-slate-900 text-white px-5 py-3 shadow hover:shadow-md"
                    >
                        Get started <ChevronRight className="h-4 w-4" />
                    </button>
                    <button
                        onClick={onLearnMore}
                        className="text-slate-600 hover:text-slate-900 inline-flex items-center gap-2 border rounded-2xl px-5 py-3 bg-white"
                    >
                        Learn more <Info className="h-4 w-4" />
                    </button>
                </div>
                <ul className="mt-6 grid grid-cols-1 sm:grid-cols-2 gap-3 text-sm text-slate-600">
                    <li className="flex items-center gap-2"><ShieldCheck className="h-4 w-4" />Aligned with UK guidance</li>
                    <li className="flex items-center gap-2"><FileText className="h-4 w-4" />Shareable structured report</li>
                    <li className="flex items-center gap-2"><ImageIcon className="h-4 w-4" />Grad-CAM heatmap overlay</li>
                    <li className="flex items-center gap-2"><Stethoscope className="h-4 w-4" />Built for clinicians</li>
                </ul>
            </div>

            <div className="p-6 rounded-3xl bg-gradient-to-br from-slate-900 to-slate-800 text-white shadow-sm border border-slate-700">
                <div className="rounded-2xl bg-white/10 p-4 ring-1 ring-white/20">
                    <div className="text-xs uppercase tracking-wider text-white/70">Preview</div>
                    <div className="mt-2 h-64 rounded-xl bg-slate-950 flex items-center justify-center overflow-hidden relative">
                        <img src="/example1.png" alt="App preview" className="absolute inset-0 h-full w-full object-cover opacity-80" />
                        <div className="absolute inset-0 bg-gradient-to-t from-slate-950/80 to-transparent" />
                        <ImageIcon className="relative h-10 w-10 text-white/80" />
                    </div>
                    <p className="mt-3 text-sm text-white/80">
                        The prototype only displays live model output. If the backend is unavailable, you will see an error message rather than mock results.
                    </p>
                </div>
            </div>
        </section>
    );
}
