import React from "react";
import { motion } from "framer-motion";
import { ArrowRight } from "lucide-react";
import { Button } from "../components/ui/Button";
import { Section } from "../components/ui/Section";
import { EASE_OUT } from "../components/ui/motion";
import { FilmPanel } from "../components/viewer/FilmPanel";
import { FindingsTable } from "../components/results/FindingsList";
import { Prediction } from "../types";

interface LandingProps {
    onStart: () => void;
    onLearnMore: () => void;
}

const METHOD = [
    { title: "Consent and upload", body: "Confirm the image is de-identified, then drop, choose or paste a PNG or JPG up to 10 MB. Two sample films are bundled.", tech: "client-side checks" },
    { title: "Classify", body: "The model returns a score for each of fourteen thoracic findings. Each is judged against its own cutoff, not a blanket 50%.", tech: "POST /predict" },
    { title: "Localise", body: "An attention map for the highest-scoring finding is composited onto the film. Compare it with the original using a split view.", tech: "attention_overlay" },
    { title: "Summarise", body: "Retrieval over BTS, NICE, AHA and Fleischner guidance grounds a short synthesis. A citation is dropped if the source doesn't mention the finding.", tech: "POST /summarize, streamed" },
];

// Illustrative values for the specimen figure; labelled as such in its caption.
const SPECIMEN: Prediction[] = [
    { label: "Effusion", prob: 0.71, threshold: 0.57 },
    { label: "Infiltration", prob: 0.61, threshold: 0.57 },
    { label: "Atelectasis", prob: 0.48, threshold: 0.62 },
    { label: "Pleural_Thickening", prob: 0.33, threshold: 0.59 },
    { label: "Cardiomegaly", prob: 0.22, threshold: 0.58 },
    { label: "Nodule", prob: 0.12, threshold: 0.56 },
];

const LIMITS: [string, string][] = [
    ["Decision support only.", "It is not a diagnosis and not a substitute for clinical judgement."],
    ["Context matters.", "History, examination, labs and prior imaging all change management."],
    ["Urgent symptoms come first.", "Severe breathlessness, chest pain, haemoptysis or hypoxia warrant urgent care, whatever the scores say."],
    ["De-identified images only.", "Never upload a film that carries a name, date of birth or hospital number."],
    ["The model has limits.", "Performance varies with device, positioning and image quality."],
];

export function Landing({ onStart, onLearnMore }: LandingProps) {
    return (
        <div className="mx-auto max-w-[1200px] px-6">
            {/* Hero */}
            <motion.section
                initial={{ opacity: 0, y: 8 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.6, ease: EASE_OUT }}
                className="grid grid-cols-1 gap-x-10 gap-y-12 pb-16 pt-14 md:grid-cols-12 md:pb-24 md:pt-20"
            >
                <div className="md:col-span-7 md:pr-6">
                    <div className="label">Chest radiograph · decision-support prototype</div>
                    <h1 className="mt-6 font-serif text-[52px] font-normal leading-[0.98] tracking-[-0.025em] sm:text-[68px] lg:text-[84px]">
                        Chest <span className="whitespace-nowrap">X-ray</span> findings, with the working shown.
                    </h1>
                    <p className="mt-8 max-w-[34em] text-[17px] leading-[1.6] text-ink-soft">
                        Upload a de-identified chest radiograph. PulmoLens scores fourteen thoracic findings, marks where the
                        model was looking, and drafts a short synthesis that cites the guideline it drew on. It is a teaching and
                        portfolio project, not a diagnostic device.
                    </p>
                    <div className="mt-10 flex flex-wrap items-center gap-6">
                        <Button variant="primary" onClick={onStart} className="h-11 px-5">
                            Analyse an X-ray <ArrowRight className="h-4 w-4" />
                        </Button>
                        <button onClick={() => document.getElementById("limitations")?.scrollIntoView({ behavior: "smooth" })} className="link text-sm">
                            Read the limitations first
                        </button>
                    </div>
                    <ol className="mt-14 hidden gap-6 border-t border-rule pt-4 font-mono text-[12px] text-ink-muted sm:flex">
                        {[["01", "Method", "method"], ["02", "Output", "output"], ["03", "Limitations", "limitations"]].map(([n, t, id]) => (
                            <li key={id}>
                                <button onClick={() => document.getElementById(id)?.scrollIntoView({ behavior: "smooth" })} className="hover:text-ink">
                                    <span className="text-ink-faint">{n}</span> {t}
                                </button>
                            </li>
                        ))}
                    </ol>
                </div>

                <figure className="md:col-span-5">
                    <FilmPanel
                        className="aspect-square"
                        tl={"PulmoLens\nSample 01"}
                        tr={"1024 × 1024\n8-bit grey"}
                        bl="Not for diagnostic use"
                    >
                        <img src="/example1.png" alt="Sample chest radiograph" className="h-full w-full object-cover" draggable={false} />
                    </FilmPanel>
                    <figcaption className="mt-3 text-[13px] leading-snug text-ink-muted">
                        <span className="font-medium text-ink">Fig. 1</span> Sample 01, one of two bundled films you can analyse
                        without uploading anything.
                    </figcaption>
                </figure>
            </motion.section>

            <Section index="01" label="Method" id="method">
                <ol>
                    {METHOD.map((m, i) => (
                        <li key={m.title} className="grid grid-cols-[2.5rem_1fr] gap-x-4 border-b border-rule py-5 first:pt-0 md:grid-cols-[2.5rem_1fr_13rem]">
                            <span className="font-serif text-2xl leading-none text-ink-faint">{i + 1}</span>
                            <div>
                                <div className="font-medium">{m.title}</div>
                                <p className="mt-1 max-w-[52ch] text-[14.5px] leading-relaxed text-ink-muted">{m.body}</p>
                            </div>
                            <code className="col-start-2 mt-2 font-mono text-[12px] text-ink-muted md:col-start-3 md:mt-0 md:text-right">{m.tech}</code>
                        </li>
                    ))}
                </ol>
            </Section>

            <Section index="02" label="Output" id="output">
                <p className="max-w-[36em] font-serif text-[24px] leading-[1.35] tracking-[-0.005em]">
                    A report has three parts: a findings table, a short synthesis, and next-step guidance for anything flagged.
                </p>
                <div className="mt-10 grid grid-cols-1 gap-10 lg:grid-cols-[1.15fr_1fr]">
                    <figure>
                        <FindingsTable predictions={SPECIMEN} specimen />
                        <figcaption className="mt-3 text-[13px] leading-snug text-ink-muted">
                            <span className="font-medium text-ink">Fig. 2</span> Findings table, illustrative values. The vertical
                            tick is each finding's own cutoff; red marks a score above it.
                        </figcaption>
                    </figure>
                    <figure>
                        <blockquote className="border-l-2 border-ink pl-5 font-serif text-[18px] leading-[1.6]">
                            <p>
                                <span className="font-semibold">Radiographic signature.</span> Blunting of the right costophrenic
                                angle with a meniscus, consistent with a small pleural effusion.<sup className="ml-0.5 font-mono text-[10px] text-marker">1</sup>{" "}
                                Ultrasound will characterise it before any diagnostic tap.<sup className="ml-0.5 font-mono text-[10px] text-marker">2</sup>
                            </p>
                            <ol className="mt-4 space-y-0.5 font-sans text-[12.5px] leading-snug text-ink-muted">
                                <li><span className="font-mono text-marker">1</span> BTS Guideline for Pleural Disease</li>
                                <li><span className="font-mono text-marker">2</span> NICE Lung Cancer (differential context)</li>
                            </ol>
                        </blockquote>
                        <figcaption className="mt-3 text-[13px] leading-snug text-ink-muted">
                            <span className="font-medium text-ink">Fig. 3</span> Synthesis excerpt, illustrative. On a live report it
                            streams in as the model writes it.
                        </figcaption>
                    </figure>
                </div>
            </Section>

            <Section index="03" label="Limitations" id="limitations">
                <ol className="max-w-[40em] space-y-4 font-serif text-[20px] leading-[1.45]">
                    {LIMITS.map(([lead, rest], i) => (
                        <li key={lead} className="grid grid-cols-[2rem_1fr]">
                            <span className="font-mono text-[13px] leading-[2.2] text-ink-faint">{i + 1}.</span>
                            <span><span className="font-medium">{lead}</span> <span className="text-ink-soft">{rest}</span></span>
                        </li>
                    ))}
                </ol>
                <button onClick={onLearnMore} className="link mt-8 text-sm">Aims, initiatives and full disclaimers</button>
            </Section>

            <Section index="04" label="Try it">
                <div className="flex flex-col gap-6 md:flex-row md:items-end md:justify-between">
                    <p className="max-w-[30em] font-serif text-[24px] leading-[1.35]">
                        Both sample films run the full pipeline, so you can see a complete report without an image of your own.
                    </p>
                    <Button variant="primary" onClick={onStart} className="h-11 shrink-0 px-5">
                        Start an analysis <ArrowRight className="h-4 w-4" />
                    </Button>
                </div>
            </Section>
        </div>
    );
}
