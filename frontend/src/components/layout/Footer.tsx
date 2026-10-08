import React from "react";

export function Footer() {
    return (
        <footer className="border-t border-line print:hidden">
            <div className="mx-auto grid max-w-[1280px] grid-cols-1 gap-6 px-4 py-10 sm:px-6 md:grid-cols-[220px_1fr] lg:px-10">
                <div className="text-[13.5px] text-fg-muted">
                    <div className="font-semibold text-fg [font-stretch:112%]">PulmoLens</div>
                    <div className="mt-1">© {new Date().getFullYear()} · Portfolio prototype</div>
                </div>
                <p className="max-w-[70em] text-[13.5px] leading-relaxed text-fg-muted">
                    A technical demonstration of model engineering and retrieval-augmented generation. It is not approved by the
                    FDA or any regulatory body. Predictions and generated reports are <span className="text-fg">not medical advice</span> and
                    must not be used for diagnostic or clinical decision-making. Always consult a qualified healthcare professional.
                </p>
            </div>
        </footer>
    );
}
