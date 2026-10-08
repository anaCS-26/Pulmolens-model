import React from "react";
import { ArrowRight } from "lucide-react";
import { Button } from "../components/ui/Button";
import { Section } from "../components/ui/Section";

interface AboutProps {
    onBack: () => void;
    onStart: () => void;
}

const SECTIONS: { title: string; warn?: boolean; items: React.ReactNode[] }[] = [
    {
        title: "Aims",
        items: [
            "Support clinicians and students with concise, guideline-anchored chest X-ray summaries.",
            "Standardise first-line investigations and safety-net advice for common thoracic findings.",
            "Speed up learning through transparent, citable references.",
        ],
    },
    {
        title: "Initiatives",
        items: [
            "Integrate UK guidance (NICE, BTS) and institutional documents.",
            "Offer structured clinician reports and a one-click patient summary.",
            "Design for accessibility, privacy, and auditability.",
        ],
    },
    {
        title: "Warnings and disclaimers",
        warn: true,
        items: [
            <><span className="font-medium">Decision support only:</span> not a diagnosis, and not a substitute for clinical judgement.</>,
            <><span className="font-medium">Context matters:</span> history, examination, labs, and prior imaging all change management.</>,
            <><span className="font-medium">Urgent symptoms:</span> severe breathlessness, chest pain, haemoptysis, or hypoxia warrant urgent care.</>,
            <><span className="font-medium">Data protection:</span> use de-identified images only.</>,
            <><span className="font-medium">Model limits:</span> performance varies with device, positioning, and image quality.</>,
        ],
    },
];

export function About({ onBack, onStart }: AboutProps) {
    return (
        <div className="mx-auto max-w-[1200px] px-6">
            <header className="pb-14 pt-14 md:pb-20 md:pt-20">
                <button onClick={onBack} className="link text-sm text-ink-muted">← Overview</button>
                <h1 className="mt-8 max-w-[14em] font-serif text-[44px] font-normal leading-[1.02] tracking-[-0.02em] md:text-[64px]">
                    A transparent tool for learning to read chest films.
                </h1>
            </header>

            {SECTIONS.map((s, i) => (
                <Section key={s.title} index={String(i + 1).padStart(2, "0")} label={s.title}>
                    <ol className="max-w-[40em] space-y-3 text-[17px] leading-[1.55]">
                        {s.items.map((it, j) => (
                            <li key={j} className="grid grid-cols-[2rem_1fr]">
                                <span className={`font-mono text-[12px] leading-[2.2] ${s.warn ? "text-marker" : "text-ink-faint"}`}>{j + 1}.</span>
                                <span className="text-ink-soft">{it}</span>
                            </li>
                        ))}
                    </ol>
                </Section>
            ))}

            <Section label="Next">
                <div className="flex flex-wrap items-center gap-6">
                    <Button variant="primary" onClick={onStart} className="h-11 px-5">
                        Analyse an X-ray <ArrowRight className="h-4 w-4" />
                    </Button>
                    <button onClick={onBack} className="link text-sm">Back to the overview</button>
                </div>
            </Section>
        </div>
    );
}
