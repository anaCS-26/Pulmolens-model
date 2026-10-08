import React from "react";
import { ArrowLeft, ArrowRight } from "lucide-react";
import { Button } from "../components/ui/Button";
import { Section } from "../components/ui/Section";
import { cn } from "../utils/cn";

interface AboutProps {
    onBack: () => void;
    onStart: () => void;
}

const AIMS = [
    "Support clinicians and students with concise, guideline-anchored chest X-ray summaries.",
    "Standardise first-line investigations and safety-net advice for common thoracic findings.",
    "Speed up learning through transparent, citable references.",
];

const INITIATIVES = [
    "Integrate UK guidance (NICE, BTS) and institutional documents.",
    "Offer structured clinician reports and a one-click patient summary.",
    "Design for accessibility, privacy and auditability.",
];

const WARNINGS: { title: string; body: string; urgent?: boolean }[] = [
    { title: "Urgent symptoms", body: "Severe breathlessness, chest pain, haemoptysis or hypoxia warrant urgent care.", urgent: true },
    { title: "Decision support only", body: "Not a diagnosis, and not a substitute for clinical judgement." },
    { title: "Context matters", body: "History, examination, labs and prior imaging all change management." },
    { title: "Data protection", body: "Use de-identified images only." },
    { title: "Model limits", body: "Performance varies with device, positioning and image quality." },
];

export function About({ onBack, onStart }: AboutProps) {
    return (
        <div>
            <header className="mx-auto max-w-[1280px] px-4 pb-16 pt-12 sm:px-6 md:pb-24 md:pt-20 lg:px-10">
                <button onClick={onBack} className="flex items-center gap-1.5 text-[14px] text-fg-muted transition-colors hover:text-fg">
                    <ArrowLeft className="h-4 w-4" /> Overview
                </button>
                <h1 className="display mt-8 max-w-[14em] text-[42px] leading-[1] md:text-[66px] [font-stretch:118%]">
                    A transparent tool for learning to read chest films.
                </h1>
            </header>

            <Section kicker="Purpose" title="What it's for">
                <div className="mt-10 grid grid-cols-1 gap-10 md:grid-cols-2">
                    <List heading="Aims" items={AIMS} />
                    <List heading="Initiatives" items={INITIATIVES} />
                </div>
            </Section>

            <Section kicker="Warnings" title="Read these before you upload anything.">
                <div className="mt-10 grid grid-cols-1 gap-px overflow-hidden rounded-xl border border-line bg-line md:grid-cols-2">
                    {WARNINGS.map((w) => (
                        <div key={w.title} className={cn("bg-surface px-6 py-6", w.urgent && "bg-surface-raised md:col-span-2")}>
                            <h3 className="flex items-center gap-2.5 text-[17px] font-semibold">
                                {w.urgent && <span className="h-2 w-2 rounded-full bg-urgent shadow-[0_0_0_4px_rgb(var(--urgent)/0.18)]" aria-hidden />}
                                {w.title}
                            </h3>
                            <p className="mt-1.5 text-[15px] text-fg-muted">{w.body}</p>
                        </div>
                    ))}
                </div>
                <div className="mt-12 flex flex-wrap items-center gap-4">
                    <Button variant="primary" size="lg" onClick={onStart}>
                        Analyse an X-ray <ArrowRight className="h-4 w-4" />
                    </Button>
                    <Button variant="outline" size="lg" onClick={onBack}>Back to the overview</Button>
                </div>
            </Section>
        </div>
    );
}

function List({ heading, items }: { heading: string; items: string[] }) {
    return (
        <div>
            <h3 className="border-b border-line pb-3 text-[16px] font-semibold">{heading}</h3>
            <ul className="mt-1">
                {items.map((it) => (
                    <li key={it} className="border-b border-line-soft py-3.5 text-[16px] leading-relaxed text-fg-soft">{it}</li>
                ))}
            </ul>
        </div>
    );
}
