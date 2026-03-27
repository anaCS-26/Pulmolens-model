import React from "react";
import { CLINICIAN_COPY, GUIDELINE_TAGS } from "../../data/constants";

interface GuidedCardProps {
    label: string;
}

export function GuidedCard({ label }: GuidedCardProps) {
    const bullets: Record<string, string[]> = {
        Consolidation: [
            "Assess severity (e.g., CURB-65); consider sepsis criteria",
            "Baseline bloods; microbiology if severe/systemic",
            "Empiric antibiotics per BTS pneumonia pathway",
            "Consider repeat CXR in 6 weeks if >50y or high risk",
        ],
        Cardiomegaly: [
            "Correlate with HF symptoms/signs; BNP if new",
            "ECG; consider echo; review edema features",
            "Optimise HF therapy per NICE as indicated",
        ],
        Effusion: [
            "Bedside US; diagnostic tap if exudate suspected",
            "Drain if complicated; antibiotics if parapneumonic",
            "Consider CT and malignancy work-up as appropriate",
        ],
        Pneumothorax: [
            "Quantify size/symptoms; aspiration vs ICC per BTS",
            "Immediate decompression if tension physiology",
        ],
        Nodule: [
            "Apply BTS nodule risk model; compare prior imaging",
            "Plan surveillance vs PET-CT/biopsy depending on risk",
        ],
        Mass: ["2-week wait lung cancer referral", "Staging CT and MDT discussion"],
        Atelectasis: [
            "Treat precipitant (analgesia, physio, mobilisation)",
            "Consider bronchoscopy if mucus plug suspected and severe",
        ],
        Edema: [
            "Diuretics/HF optimisation; treat triggers (AF, infection)",
            "Escalate if hypoxic/hemodynamically unstable",
        ],
        Emphysema: [
            "Correlate with spirometry (COPD); smoking cessation, vaccinations",
            "Consider referral to pulmonary rehab",
        ],
        Fibrosis: ["If suspected ILD, discuss HRCT and ILD clinic referral"],
        Infiltration: [
            "Integrate with clinical context: infection, edema, hemorrhage",
            "Further imaging if uncertainty persists",
        ],
        Pleural_Thickening: [
            "Occupational history; consider CT and mesothelioma work-up if concerning",
        ],
        Hernia: ["If acute compromise, urgent surgical review", "CT to define anatomy"],
        Pneumonia: [
            "BTS antibiotic pathway; assess for admission criteria",
            "Safety-net and follow-up imaging if indicated",
        ],
    };

    const blurb = CLINICIAN_COPY[label] || "";
    const tags = GUIDELINE_TAGS[label] || [];

    return (
        <div className="rounded-2xl border border-slate-200 p-4 bg-white">
            <div className="flex items-start justify-between gap-3">
                <div>
                    <div className="font-semibold">{label}</div>
                    <div className="text-xs text-slate-500 mt-0.5">{blurb}</div>
                </div>
                <span className="rounded-full bg-slate-100 text-slate-700 px-2 py-1 text-xs">Guidance</span>
            </div>
            <ul className="mt-3 list-disc pl-5 text-sm text-slate-800 space-y-1">
                {(bullets[label] || ["Review with clinical context."]).map((t, i) => (
                    <li key={i}>{t}</li>
                ))}
            </ul>
            {tags.length > 0 && (
                <div className="mt-3 flex flex-wrap gap-2">
                    {tags.map((g) => (
                        <span key={g} className="rounded-full bg-slate-50 border border-slate-200 px-2.5 py-1 text-xs text-slate-600">
                            {g}
                        </span>
                    ))}
                </div>
            )}
        </div>
    );
}
