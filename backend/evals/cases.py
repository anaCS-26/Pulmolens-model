from dataclasses import dataclass, field
from typing import Dict, List


CLASS_NAMES = [
    "Atelectasis", "Cardiomegaly", "Effusion", "Infiltration", "Mass",
    "Nodule", "Pneumonia", "Pneumothorax", "Consolidation", "Edema",
    "Emphysema", "Fibrosis", "Pleural_Thickening", "Hernia",
]


@dataclass
class EvalCase:
    name: str
    predictions: Dict[str, float]
    primary_finding: str = ""
    expect_short_circuit: bool = False
    must_contain: List[str] = field(default_factory=list)
    must_not_contain: List[str] = field(default_factory=list)
    urgent: bool = False
    notes: str = ""


def with_prob(**overrides) -> Dict[str, float]:
    base = {c: 0.02 for c in CLASS_NAMES}
    base.update(overrides)
    return base


REQUIRED_SECTIONS = ["Radiographic Signature", "Clinical Management", "Patient Summary"]


PROMPT_LEAK_FORBIDDEN = [
    "INTERNAL DATA",
    "DO NOT REFERENCE",
    "FINDINGS:",
    "RELEVANT_GUIDELINES",
    "raw data",
    "guidelines are missing",
    "INSTRUCTIONS for Senior Radiographic Consultant",
    "KNOWLEDGE FALLBACK",
    "VISUAL GROUNDING",
    "GUIDELINE ADHERENCE",
    "PRIMARY FINDING:",
]


CASES: List[EvalCase] = [
    EvalCase(
        name="high_confidence_pneumonia",
        predictions=with_prob(Pneumonia=0.92, Infiltration=0.61, Consolidation=0.55),
        primary_finding="Pneumonia",
        must_contain=REQUIRED_SECTIONS + ["Pneumonia"],
    ),
    EvalCase(
        name="cardiomegaly_dominant",
        predictions=with_prob(Cardiomegaly=0.88, Effusion=0.42),
        primary_finding="Cardiomegaly",
        must_contain=REQUIRED_SECTIONS + ["Cardiomegaly"],
    ),
    EvalCase(
        name="urgent_pneumothorax",
        predictions=with_prob(Pneumothorax=0.94),
        primary_finding="Pneumothorax",
        must_contain=REQUIRED_SECTIONS + ["Pneumothorax"],
        urgent=True,
        notes="Life-threatening; judge should flag if urgency missing.",
    ),
    EvalCase(
        name="mass_oncology_workup",
        predictions=with_prob(Mass=0.81, Nodule=0.58),
        primary_finding="Mass",
        must_contain=REQUIRED_SECTIONS + ["Mass"],
    ),
    EvalCase(
        name="multi_finding_complex",
        predictions=with_prob(
            Pneumonia=0.78, Effusion=0.71, Consolidation=0.66, Atelectasis=0.52
        ),
        primary_finding="Pneumonia",
        must_contain=REQUIRED_SECTIONS + ["Pneumonia"],
        notes="Multiple co-occurring findings; primary must remain Pneumonia (highest prob).",
    ),
    EvalCase(
        name="rare_hernia_fallback",
        predictions=with_prob(Hernia=0.83),
        primary_finding="Hernia",
        must_contain=REQUIRED_SECTIONS + ["Hernia"],
        must_not_contain=["guidelines are missing", "raw data", "I don't have", "I do not have"],
        notes="Tests KNOWLEDGE FALLBACK rule when pathology absent from BTS/NICE.",
    ),
    EvalCase(
        name="sub_threshold_short_circuit",
        predictions=with_prob(Atelectasis=0.32, Effusion=0.28),
        expect_short_circuit=True,
        must_contain=["below the RAG threshold"],
        notes="Should bypass LLM entirely.",
    ),
    EvalCase(
        name="just_above_threshold",
        predictions=with_prob(Effusion=0.51),
        primary_finding="Effusion",
        must_contain=REQUIRED_SECTIONS + ["Effusion"],
    ),
    EvalCase(
        name="emphysema_chronic",
        predictions=with_prob(Emphysema=0.79, Fibrosis=0.41),
        primary_finding="Emphysema",
        must_contain=REQUIRED_SECTIONS + ["Emphysema"],
    ),
    EvalCase(
        name="edema_acute",
        predictions=with_prob(Edema=0.85, Cardiomegaly=0.61, Effusion=0.55),
        primary_finding="Edema",
        must_contain=REQUIRED_SECTIONS + ["Edema"],
        urgent=True,
    ),
    EvalCase(
        name="effusion_isolated",
        predictions=with_prob(Effusion=0.84),
        primary_finding="Effusion",
        must_contain=REQUIRED_SECTIONS + ["Effusion"],
    ),
    EvalCase(
        name="nodule_followup",
        predictions=with_prob(Nodule=0.72),
        primary_finding="Nodule",
        must_contain=REQUIRED_SECTIONS + ["Nodule"],
        notes="Standard workup expected (size, follow-up CT).",
    ),
]
