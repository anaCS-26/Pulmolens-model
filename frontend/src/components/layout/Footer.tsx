import React from "react";

export function Footer() {
    return (
        <footer className="border-t border-ink print:hidden">
            <div className="mx-auto grid max-w-[1200px] grid-cols-1 gap-6 px-6 py-10 md:grid-cols-12">
                <div className="md:col-span-3">
                    <div className="font-serif text-lg">PulmoLens</div>
                    <div className="mt-1 font-mono text-[11px] text-ink-muted">© {new Date().getFullYear()} · Portfolio prototype</div>
                </div>
                <p className="max-w-3xl text-[13px] leading-relaxed text-ink-muted md:col-span-9">
                    <span className="font-medium text-ink">Disclaimer.</span> This application is a technical demonstration of
                    AI engineering and RAG capabilities. It is not approved by the FDA or any regulatory body. The predictions and
                    generated reports are <span className="font-medium text-ink">not medical advice</span>, and must not be used for
                    diagnostic or clinical decision-making. Always consult a qualified healthcare professional.
                </p>
            </div>
        </footer>
    );
}
