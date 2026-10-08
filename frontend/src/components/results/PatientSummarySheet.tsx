import React, { useEffect } from "react";
import { createPortal } from "react-dom";
import { motion } from "framer-motion";
import { toLayTerm, SAFETY_NET_PATIENT } from "../../data/guidance";
import { Button } from "../ui/Button";
import { capitalize } from "../../utils/format";

interface PatientSummarySheetProps {
    findings: { label: string; prob: number }[];
    onClose: () => void;
    onPrint: () => void;
}

/** Right-hand slide-over in plain English, styled like the paper it prints to. */
export function PatientSummarySheet({ findings, onClose, onPrint }: PatientSummarySheetProps) {
    useEffect(() => {
        const prev = document.body.style.overflow;
        document.body.style.overflow = "hidden";
        const onKey = (e: KeyboardEvent) => e.key === "Escape" && onClose();
        window.addEventListener("keydown", onKey);
        return () => { document.body.style.overflow = prev; window.removeEventListener("keydown", onKey); };
    }, [onClose]);

    return createPortal(
        <div className="fixed inset-0 z-[60] print:hidden" role="dialog" aria-modal="true" aria-label="Patient-friendly summary">
            <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }} className="absolute inset-0 bg-ink/40" onClick={onClose} />
            <motion.aside
                initial={{ x: "100%" }}
                animate={{ x: 0 }}
                exit={{ x: "100%" }}
                transition={{ duration: 0.3, ease: [0.16, 1, 0.3, 1] }}
                className="absolute inset-y-0 right-0 flex w-full max-w-lg flex-col border-l border-ink bg-paper"
            >
                <div className="flex items-center justify-between border-b border-ink px-6 py-3">
                    <h3 className="label !text-ink">Patient-friendly summary</h3>
                    <div className="flex items-center gap-5">
                        <Button size="sm" variant="primary" onClick={onPrint}>Print</Button>
                        <button onClick={onClose} className="link text-[13px]" aria-label="Close">Close</button>
                    </div>
                </div>

                <div className="flex-1 overflow-y-auto px-6 py-8">
                    <PatientSummaryBody findings={findings} />
                </div>
            </motion.aside>
        </div>,
        document.body
    );
}

/** Shared by the on-screen sheet and the printed page. */
export function PatientSummaryBody({ findings }: { findings: { label: string }[] }) {
    return (
        <div className="text-[14px] leading-relaxed">
            <div className="font-serif text-[34px] leading-tight">Your chest X-ray</div>
            <p className="mt-3">
                <strong>What this means:</strong> your chest X-ray suggests the findings listed below. This summary is to aid
                understanding and does not replace medical advice.
            </p>
            <ul className="mt-4 space-y-2">
                {findings.length === 0 && <li className="border-l-2 border-ink pl-3">{capitalize(toLayTerm("No findings"))}.</li>}
                {findings.map((f) => (
                    <li key={f.label} className="border-l-2 border-ink pl-3">
                        <strong>{capitalize(toLayTerm(f.label))}:</strong> please follow the plan agreed with your clinician.
                    </li>
                ))}
            </ul>
            <h4 className="mt-6 font-semibold">What should happen next?</h4>
            <p className="mt-1">
                Depending on your symptoms and history, your clinician may arrange blood tests, a repeat X-ray, additional scans,
                or treatment.
            </p>
            <h4 className="mt-6 font-semibold text-marker-dark">Get urgent help if you develop:</h4>
            <ul className="mt-2 list-disc space-y-1 pl-5">
                {SAFETY_NET_PATIENT.map((s) => <li key={s}>{s}</li>)}
            </ul>
            <p className="mt-6 border-t border-rule pt-4 text-xs text-ink-muted">
                Disclaimer: decision support only. Not a diagnosis. Imaging must always be interpreted in clinical context.
            </p>
        </div>
    );
}
