import React from "react";

export function Footer() {
    return (
        <footer className="mt-12 text-center text-xs text-slate-500 border-t border-slate-200 bg-white">
            <div className="mx-auto max-w-6xl px-4 py-8">
                <div className="font-medium text-slate-700 flex items-center justify-center gap-2">
                    © {new Date().getFullYear()} PulmoLens · Portfolio prototype.
                    {import.meta.env.VITE_DEMO_MODE === 'true' && (
                        <span className="bg-indigo-100 text-indigo-700 px-2 py-0.5 rounded-full text-[10px] font-bold uppercase tracking-wider">Demo Mode Active</span>
                    )}
                </div>
                <div className="mt-2 text-[11px] leading-relaxed max-w-3xl mx-auto text-slate-400">
                    <strong>DISCLAIMER:</strong> This application is a technical demonstration of AI Engineering and RAG capabilities. It is not approved by the FDA or any regulatory body. 
                    The predictions and generated reports are <strong>NOT medical advice</strong>, and must not be used for diagnostic or clinical decision-making. 
                    Always consult a qualified healthcare professional.
                </div>
            </div>
        </footer>
    );
}
