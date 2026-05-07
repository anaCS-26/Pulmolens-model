import React from "react";

function toLayTerm(label: string): string {
    const map: Record<string, string> = {
        Atelectasis: "partial lung collapse",
        Cardiomegaly: "enlarged heart",
        Effusion: "fluid around the lungs",
        Infiltration: "patchy lung changes",
        Mass: "larger spot that needs checking",
        Nodule: "small spot that needs checking",
        Pneumonia: "lung infection",
        Pneumothorax: "air leak (collapsed lung)",
        Consolidation: "solid-looking lung area (often infection)",
        Edema: "fluid in the lungs",
        Emphysema: "damaged air sacs (COPD)",
        Fibrosis: "scarring of the lungs",
        Pleural_Thickening: "thickening of the lining around the lung",
        Hernia: "abnormal organ movement",
        "No findings": "no significant problems seen",
    };
    return map[label] ?? label;
}

interface PatientSummarySheetProps {
    findings: { label: string; prob: number }[];
    onClose: () => void;
}

export function PatientSummarySheet({ findings, onClose }: PatientSummarySheetProps) {
    return (
        <div role="dialog" aria-label="Patient-friendly summary" className="fixed inset-0 z-50 grid place-items-center bg-black/30 p-4">
            <div className="w-full max-w-2xl rounded-xl border bg-white p-5 shadow print:block">
                <div className="mb-4 flex items-center justify-between print-hidden">
                    <h3 className="text-lg font-semibold">Patient-friendly summary</h3>
                    <div className="flex gap-2">
                        <button className="rounded border px-3 py-2 text-sm" onClick={() => window.print()}>Print summary</button>
                        <button className="rounded px-3 py-2 text-sm" onClick={onClose}>Close</button>
                    </div>
                </div>

                <div className="prose max-w-none text-sm text-zinc-800">
                    <p><strong>What this means:</strong> your chest X-ray suggests the findings listed below. This summary is to aid understanding and does not replace medical advice.</p>
                    <ul className="mt-3">
                        {findings.map((f) => (
                            <li key={f.label}><strong>{toLayTerm(f.label)}:</strong> please follow the plan agreed with your clinician.</li>
                        ))}
                    </ul>
                    <h4 className="mt-4">What should happen next?</h4>
                    <p>Depending on your symptoms and history, your clinician may arrange blood tests, a repeat X-ray, additional scans, or treatment.</p>
                    <h4 className="mt-4">Get urgent help if you develop:</h4>
                    <ul>
                        <li>Severe breathlessness or chest pain</li>
                        <li>Very low oxygen levels or fainting</li>
                        <li>Coughing up blood</li>
                        <li>Rapidly worsening symptoms</li>
                    </ul>
                    <p className="mt-4 text-xs text-zinc-500">Disclaimer: decision support only. Not a diagnosis. Imaging must always be interpreted in clinical context.</p>
                </div>
            </div>
        </div>
    );
}
