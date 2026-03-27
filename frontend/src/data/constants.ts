export const LABELS: string[] = [
    "Atelectasis",
    "Cardiomegaly",
    "Effusion",
    "Infiltration",
    "Mass",
    "Nodule",
    "Pneumonia",
    "Pneumothorax",
    "Consolidation",
    "Edema",
    "Emphysema",
    "Fibrosis",
    "Pleural_Thickening",
    "Hernia",
    "No findings",
];

export const GUIDELINE_TAGS: Record<string, string[]> = {
    Atelectasis: ["StatPearls (Atelectasis)", "BTS Pneumonia (context)"],
    Cardiomegaly: ["NICE Chronic Heart Failure", "NICE Acute HF"],
    Effusion: ["BTS Pleural Disease", "NICE Lung Cancer (ddx context)"],
    Infiltration: ["BTS Pneumonia", "Immunocompromised Infiltrates"],
    Mass: ["NICE Lung Cancer"],
    Nodule: ["BTS Pulmonary Nodules", "NICE Lung Cancer"],
    Pneumonia: ["BTS Pneumonia", "NICE COPD (exacerbation ddx)"],
    Pneumothorax: ["BTS Pleural Disease"],
    Consolidation: ["BTS Pneumonia"],
    Edema: ["NICE Heart Failure"],
    Emphysema: ["NICE COPD"],
    Fibrosis: ["NICE Idiopathic Pulmonary Fibrosis"],
    Pleural_Thickening: ["BTS Pleural Disease"],
    Hernia: ["WSES/EAST Diaphragmatic Hernia", "NHS CDH leaflet (context)"],
    "No findings": ["—"],
};

export const CLINICIAN_COPY: Record<string, string> = {
    Atelectasis:
        "Volume loss with increased density ± plate-like opacities; correlate with post-op status, mucus plugging, pain-splinting.",
    Cardiomegaly:
        "CTR > 0.5 on PA film suggests enlargement; consider HF context; review edema signs.",
    Effusion:
        "Blunting of costophrenic angles/meniscus; quantify if large; assess for septations on US.",
    Infiltration:
        "Non-specific interstitial/alveolar change; integrate clinical picture (infection, edema, hemorrhage).",
    Mass: "Discrete opacity >3 cm; compare with prior; 2ww if suspicious.",
    Nodule:
        "Solitary/multiple <3 cm; risk stratify (age, smoking, morphology); follow BTS nodule pathway.",
    Pneumonia:
        "Lobar/segmental consolidation; use CURB-65 for severity; antibiotics per BTS.",
    Pneumothorax:
        "Visceral pleural line with peripheral lucency; quantify; decompress if tension.",
    Consolidation:
        "Airspace opacification ± air bronchograms; often infection but consider aspiration/hemorrhage.",
    Edema:
        "Interstitial/alveolar edema: Kerley B lines, perihilar haze; correlate with HF.",
    Emphysema:
        "Hyperinflation, flattened diaphragms, vascular pruning; correlate with spirometry (COPD).",
    Fibrosis:
        "Reticular opacities, volume loss; consider HRCT and ILD clinic if indicated.",
    Pleural_Thickening:
        "Pleural line irregularity/calcification; occupational history; differentiate from effusion.",
    Hernia:
        "Intrathoracic bowel/stomach with air-fluid levels; urgent surgical review if compromise.",
    "No findings": "No acute radiographic abnormality identified.",
};

export const THRESHOLDS: Record<string, number> = {
    "Atelectasis": 0.6177,
    "Cardiomegaly": 0.5790,
    "Effusion": 0.5706,
    "Infiltration": 0.5690,
    "Mass": 0.5896,
    "Nodule": 0.5581,
    "Pneumonia": 0.4369,
    "Pneumothorax": 0.6355,
    "Consolidation": 0.5533,
    "Edema": 0.5551,
    "Emphysema": 0.6208,
    "Fibrosis": 0.5494,
    "Pleural_Thickening": 0.5883,
    "Hernia": 0.7249,
    "No findings": 0.5, // Fallback
};
