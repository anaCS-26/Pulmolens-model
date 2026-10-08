/** First-line next steps per finding, shown in the Guidance tab and the printed report. */
export const GUIDANCE_BULLETS: Record<string, string[]> = {
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

export const DEFAULT_GUIDANCE = ["Review with clinical context."];

const LAY_TERMS: Record<string, string> = {
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

export function toLayTerm(label: string): string {
    return LAY_TERMS[label] ?? label;
}

export const SAFETY_NET = [
    "Severe breathlessness, chest pain, haemoptysis, confusion, or cyanosis warrant urgent medical attention.",
    "If symptoms worsen or fail to improve as expected, arrange prompt clinical review.",
];

export const SAFETY_NET_PATIENT = [
    "Severe breathlessness or chest pain",
    "Very low oxygen levels or fainting",
    "Coughing up blood",
    "Rapidly worsening symptoms",
];
