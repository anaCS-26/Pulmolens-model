import React from "react";
import { ChevronLeft } from "lucide-react";

interface AboutProps {
    onBack: () => void;
}

export function About({ onBack }: AboutProps) {
    return (
        <section className="mt-8">
            <div className="p-6 rounded-3xl bg-white shadow-sm border border-slate-200">
                <h2 className="text-2xl font-semibold tracking-tight">About PulmoLens</h2>
                <div className="mt-4 grid md:grid-cols-2 gap-6 text-sm text-slate-700">
                    <div>
                        <h3 className="font-semibold">Aims</h3>
                        <ul className="mt-2 list-disc pl-5 space-y-1">
                            <li>Support clinicians and students with concise, guideline-anchored chest X-ray summaries.</li>
                            <li>Standardise first-line investigations and safety-net advice for common thoracic findings.</li>
                            <li>Speed up learning through transparent, citable references.</li>
                        </ul>
                        <h3 className="mt-6 font-semibold">Initiatives</h3>
                        <ul className="mt-2 list-disc pl-5 space-y-1">
                            <li>Integrate UK guidance (NICE, BTS) and institutional documents.</li>
                            <li>Offer structured clinician reports and a one-click patient summary.</li>
                            <li>Design for accessibility, privacy, and auditability.</li>
                        </ul>
                    </div>
                    <div>
                        <h3 className="font-semibold">Warnings and disclaimers</h3>
                        <ul className="mt-2 list-disc pl-5 space-y-1">
                            <li><strong>Decision support only:</strong> not a diagnosis, and not a substitute for clinical judgement.</li>
                            <li><strong>Context matters:</strong> history, examination, labs, and prior imaging all change management.</li>
                            <li><strong>Urgent symptoms:</strong> severe breathlessness, chest pain, haemoptysis, or hypoxia warrant urgent care.</li>
                            <li><strong>Data protection:</strong> use de-identified images only.</li>
                            <li><strong>Model limits:</strong> performance varies with device, positioning, and image quality.</li>
                        </ul>
                    </div>
                </div>
                <div className="mt-6">
                    <button onClick={onBack} className="inline-flex items-center gap-2 px-4 py-2 rounded-xl border border-slate-300 bg-white">
                        <ChevronLeft className="h-4 w-4" /> Back
                    </button>
                </div>
            </div>
        </section>
    );
}
