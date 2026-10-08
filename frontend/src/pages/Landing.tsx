import React, { useState } from "react";
import { ArrowRight } from "lucide-react";
import { Button } from "../components/ui/Button";
import { Section } from "../components/ui/Section";
import { Tabs } from "../components/ui/Tabs";
import { FilmPanel, JET_GRADIENT } from "../components/viewer/FilmPanel";
import { CompareSlider } from "../components/viewer/CompareSlider";
import { OpacitySlider } from "../components/viewer/XrayViewer";
import { FindingsTable } from "../components/results/FindingsList";
import { PerformanceChart } from "../components/results/PerformanceChart";
import { MODEL_SUMMARY } from "../data/constants";
import { Prediction } from "../types";
import { cn } from "../utils/cn";

interface LandingProps {
    onStart: () => void;
    onLearnMore: () => void;
}

// Illustrative values for sample 01; labelled as such wherever they appear.
const SPECIMEN: Prediction[] = [
    { label: "Effusion", prob: 0.71, threshold: 0.5706 },
    { label: "Infiltration", prob: 0.61, threshold: 0.569 },
    { label: "Atelectasis", prob: 0.48, threshold: 0.6177 },
    { label: "Pleural_Thickening", prob: 0.33, threshold: 0.5883 },
    { label: "Cardiomegaly", prob: 0.22, threshold: 0.579 },
    { label: "Edema", prob: 0.18, threshold: 0.5551 },
    { label: "Consolidation", prob: 0.15, threshold: 0.5533 },
    { label: "Nodule", prob: 0.12, threshold: 0.5581 },
];

const PIPELINE = [
    { title: "Your film", body: "PNG or JPG up to 10 MB. De-identified only.", viz: "film" },
    { title: "Classifier", body: "DenseNet-121 with CBAM attention, at 512 × 512.", viz: "bars" },
    { title: "Per-finding cutoffs", body: "Each score is judged against its own threshold, calibrated for recall.", viz: "cutoffs" },
    { title: "Grad-CAM++ map", body: "Where the model looked for its highest-scoring finding.", viz: "map" },
    { title: "Cited summary", body: "Retrieved from BTS, NICE, AHA/ACC and Fleischner guidance, then streamed.", viz: "docs" },
] as const;

const LIMITS: { title: string; body: string; urgent?: boolean }[] = [
    { title: "Urgent symptoms come first", body: "Severe breathlessness, chest pain, haemoptysis or hypoxia warrant urgent care, whatever the scores say.", urgent: true },
    { title: "Decision support only", body: "It is not a diagnosis and not a substitute for clinical judgement." },
    { title: "Context changes management", body: "History, examination, labs and prior imaging all matter more than a single film." },
    { title: "De-identified images only", body: "Never upload a film that carries a name, date of birth or hospital number." },
    { title: "The model has limits", body: "Performance varies with device, positioning and image quality." },
];

const wrap = "mx-auto max-w-[1280px] px-4 sm:px-6 lg:px-10";

export function Landing({ onStart, onLearnMore }: LandingProps) {
    return (
        <div>
            {/* Hero */}
            <div className={cn(wrap, "grid grid-cols-1 items-center gap-12 pb-20 pt-12 md:pb-28 md:pt-20 lg:grid-cols-[5fr_7fr] lg:gap-16")}>
                <div>
                    <h1 className="display text-[46px] leading-[0.94] sm:text-[64px] lg:text-[84px] [font-stretch:118%]">
                        A second look at the chest&nbsp;X&#8209;ray
                    </h1>
                    <p className="mt-7 max-w-[31em] text-[17px] leading-relaxed text-fg-soft md:text-[18.5px]">
                        Upload a de-identified radiograph. PulmoLens scores fourteen thoracic findings, shows you where the model was
                        looking, and writes a short summary that cites the guideline behind it.
                    </p>
                    <div className="mt-9 flex flex-wrap gap-3">
                        <Button variant="primary" size="lg" onClick={onStart}>
                            Analyse an X-ray <ArrowRight className="h-4 w-4" />
                        </Button>
                        <Button variant="outline" size="lg" onClick={onStart}>Try a sample film</Button>
                    </div>
                    <dl className="mt-11 flex flex-wrap gap-x-9 gap-y-4 border-t border-line pt-5 text-[13.5px] text-fg-muted">
                        {[["14", "findings scored"], [MODEL_SUMMARY.meanAuc.toFixed(3), "mean AUC, held-out test"], ["4", "guideline bodies cited"]].map(([v, l]) => (
                            <div key={l}>
                                <dt className="sr-only">{l}</dt>
                                <dd className="num text-[23px] font-medium leading-tight text-fg [font-stretch:110%]">{v}</dd>
                                <dd>{l}</dd>
                            </div>
                        ))}
                    </dl>
                </div>
                <HeroFilm />
            </div>

            <Section
                id="how"
                kicker="How it works"
                title="One film in. Scores, a map and a cited summary out."
                sub="Everything runs on the live model. If the server can't be reached you see an error, never placeholder results."
            >
                <ol className="relative mt-14 grid grid-cols-1 gap-x-4 gap-y-10 sm:grid-cols-2 lg:grid-cols-5">
                    <span aria-hidden className="absolute inset-x-[1%] top-[5px] hidden h-px bg-line lg:block" />
                    <span aria-hidden className="travel absolute top-[1px] hidden h-[9px] w-[9px] rounded-full bg-accent shadow-[0_0_12px_2px_rgb(var(--accent)/0.5)] lg:block" />
                    {PIPELINE.map((p) => (
                        <li key={p.title} className="relative">
                            <span className="mb-5 block h-[11px] w-[11px] rounded-full border-[1.5px] border-fg-muted bg-surface" />
                            <h3 className="text-[17px] font-semibold [font-stretch:108%]">{p.title}</h3>
                            <p className="mt-1.5 text-[14px] leading-relaxed text-fg-muted">{p.body}</p>
                            <PipelineViz kind={p.viz} />
                        </li>
                    ))}
                </ol>
            </Section>

            <Section
                id="report"
                kicker="The report"
                title="Read it the way you'd read a film."
                sub="Impression first. Then every finding against its own cutoff, the attention map beside it, and a summary you can check against its sources."
            >
                <div className="mt-14 grid grid-cols-1 overflow-hidden rounded-xl border border-line bg-surface-raised lg:grid-cols-[minmax(0,1fr)_minmax(0,440px)]">
                    <div className="grid place-items-center bg-film p-5">
                        <FilmPanel className="aspect-square w-full max-w-[540px] !ring-0" tl={"example_cxr_1.png\n448 KB"} tr="Attention · Effusion" br={"Not for\ndiagnostic use"}>
                            <img src="/example1.png" alt="" className="absolute inset-0 h-full w-full object-cover" />
                            <img src="/sample-attention.png" alt="" className="absolute inset-0 h-full w-full object-cover opacity-75 mix-blend-screen" />
                        </FilmPanel>
                    </div>
                    <div className="flex min-w-0 flex-col border-t border-line lg:border-l lg:border-t-0">
                        <div className="border-b border-line px-6 py-5">
                            <div className="text-[12.5px] text-fg-faint">Impression</div>
                            <p className="mt-1.5 text-[20px] font-medium leading-snug">
                                Two findings above cutoff: <span className="font-semibold text-accent-ink">effusion</span> and{" "}
                                <span className="font-semibold text-accent-ink">infiltration</span>.
                            </p>
                        </div>
                        <div className="px-6 py-5">
                            <FindingsTable predictions={SPECIMEN} specimen />
                        </div>
                        <blockquote className="mt-auto border-t border-line px-6 py-5 text-[14.5px] leading-relaxed text-fg-soft">
                            <span className="font-semibold text-fg">Radiographic signature.</span> Blunting of the right costophrenic angle
                            with a meniscus, consistent with a small pleural effusion.<sup className="ml-0.5 font-medium text-accent-ink">1</sup> The
                            attention map sits over the right lower zone, where this finding is expected.
                            <footer className="mt-3 text-[12.5px] text-fg-muted">
                                <span className="font-medium text-accent-ink">1</span> BTS Guideline for Pleural Disease
                            </footer>
                        </blockquote>
                    </div>
                </div>
                <p className="mt-3 text-[13px] text-fg-muted">
                    Illustrative report on sample 01. The tick on each bar is that finding's cutoff; amber marks a score above it.
                </p>
            </Section>

            <Section
                id="model"
                kicker="Model"
                title="Tuned to miss less, at the cost of more false alarms."
                sub="Results on the held-out test set. Recall is high for common findings and precision is low across the board, which is why every score is shown next to its cutoff."
            >
                <dl className="mt-12 grid grid-cols-2 border-t border-line lg:grid-cols-4">
                    {[
                        [MODEL_SUMMARY.meanAuc.toFixed(3), "Mean AUC across fourteen findings"],
                        [MODEL_SUMMARY.microRecall.toFixed(2), "Micro-averaged recall"],
                        [MODEL_SUMMARY.microPrecision.toFixed(2), "Micro-averaged precision"],
                        [MODEL_SUMMARY.positives.toLocaleString(), "Positive labels in the test set"],
                    ].map(([v, l], i) => (
                        <div key={l} className={cn("pr-5 pt-5", i % 2 === 1 && "border-l border-line pl-5", i === 2 && "lg:border-l lg:pl-5")}>
                            <dt className="sr-only">{l}</dt>
                            <dd className="num text-[44px] font-light leading-none tracking-[-0.04em] md:text-[64px] [font-stretch:110%]">{v}</dd>
                            <dd className="mt-3 max-w-[14em] text-[14px] text-fg-muted">{l}</dd>
                        </div>
                    ))}
                </dl>
                <PerformanceChart className="mt-14" />
                <p className="mt-4 max-w-[52em] text-[14px] text-fg-muted">
                    Pneumonia recall is 0.20 on 164 cases, so the report warns that a low pneumonia score doesn't rule it out.
                </p>
            </Section>

            <Section id="limitations" kicker="Limits" title="Decision support for learning, not a diagnosis.">
                <div className="mt-12 grid grid-cols-1 gap-px overflow-hidden rounded-xl border border-line bg-line md:grid-cols-2">
                    {LIMITS.map((l) => (
                        <div key={l.title} className={cn("bg-surface px-6 py-6 md:px-7", l.urgent && "bg-surface-raised md:col-span-2")}>
                            <h3 className="flex items-center gap-2.5 text-[17px] font-semibold [font-stretch:106%]">
                                {l.urgent && <span className="h-2 w-2 rounded-full bg-urgent shadow-[0_0_0_4px_rgb(var(--urgent)/0.18)]" aria-hidden />}
                                {l.title}
                            </h3>
                            <p className="mt-1.5 max-w-[44em] text-[15px] text-fg-muted">{l.body}</p>
                        </div>
                    ))}
                </div>
                <button onClick={onLearnMore} className="link mt-6 text-[14px]">Aims, initiatives and full disclaimers</button>
            </Section>

            <section className="border-t border-line bg-surface-raised">
                <div className={cn(wrap, "flex flex-col gap-6 py-14 md:flex-row md:items-center md:justify-between md:py-20")}>
                    <p className="display max-w-[16em] text-[28px] leading-[1.08] md:text-[38px]">
                        Both sample films run the full pipeline, so you can see a complete report without an image of your own.
                    </p>
                    <Button variant="primary" size="lg" onClick={onStart} className="shrink-0">
                        Start an analysis <ArrowRight className="h-4 w-4" />
                    </Button>
                </div>
            </section>
        </div>
    );
}

type HeroMode = "attention" | "split" | "original";

/** Sample 01 with an illustrative attention map. The film develops, the map sweeps in, then the labels draw. */
function HeroFilm() {
    const [mode, setMode] = useState<HeroMode>("attention");
    const [opacity, setOpacity] = useState(0.75);

    const film = <img src="/example1.png" alt="Sample chest radiograph" draggable={false} className="develop absolute inset-0 h-full w-full object-cover" />;
    const map = (cls = "") => (
        <img src="/sample-attention.png" alt="" draggable={false} className={cn("absolute inset-0 h-full w-full object-cover mix-blend-screen", cls)} style={{ opacity }} />
    );

    return (
        <figure>
            <FilmPanel
                className="aspect-square w-full shadow-[0_40px_100px_-40px_rgba(0,0,0,0.55)]"
                tl={"PulmoLens\nSample 01\n512 × 512 input"}
                bl={mode === "original" ? "Original" : "Attention · Effusion"}
                br={"Not for\ndiagnostic use"}
            >
                {mode === "split" ? (
                    <CompareSlider
                        className="h-full w-full"
                        before={film}
                        after={<>{film}{map()}</>}
                    />
                ) : (
                    <>
                        {film}
                        {mode === "attention" && map("map-sweep")}
                    </>
                )}

                {mode === "attention" && (
                    <div aria-hidden className="pointer-events-none absolute inset-0 z-[5]">
                        <svg viewBox="0 0 100 100" preserveAspectRatio="none" className="absolute inset-0 h-full w-full overflow-visible">
                            <g fill="none" stroke="#dfe9ef" strokeWidth={1.2} vectorEffect="non-scaling-stroke">
                                <circle cx="30" cy="70" r="1.6" className="fade-in" style={{ animationDelay: "2.7s" }} vectorEffect="non-scaling-stroke" />
                                <path d="M31.6 69 L46 58 L54 58" className="draw-line" style={{ animationDelay: "2.8s" }} vectorEffect="non-scaling-stroke" />
                                <circle cx="38" cy="56" r="1.4" className="fade-in" style={{ animationDelay: "3.1s" }} vectorEffect="non-scaling-stroke" />
                                <path d="M39.4 55 L50 44 L58 44" className="draw-line" style={{ animationDelay: "3.2s" }} vectorEffect="non-scaling-stroke" />
                            </g>
                        </svg>
                        <Callout left="54.5%" top="58%" delay="3.4s" name="Effusion" score="71%" cutoff="57" />
                        <Callout left="58.5%" top="44%" delay="3.6s" name="Infiltration" score="61%" cutoff="57" />
                        <span className="fade-in absolute bottom-[14%] right-2.5 top-[14%] w-[5px] rounded-full opacity-90" style={{ background: JET_GRADIENT.replace("90deg", "0deg"), animationDelay: "2.4s" }} />
                    </div>
                )}
            </FilmPanel>
            <div className="mt-3.5 flex flex-wrap items-center justify-between gap-x-5 gap-y-3">
                <Tabs<HeroMode>
                    label="View"
                    value={mode}
                    onChange={setMode}
                    options={[
                        { value: "original", label: "Original" },
                        { value: "attention", label: "Attention" },
                        { value: "split", label: "Split" },
                    ]}
                />
                {mode !== "original" && <OpacitySlider value={opacity} onChange={setOpacity} className="max-w-[230px] flex-1" />}
            </div>
            <figcaption className="mt-2.5 text-[12.5px] text-fg-faint">
                Illustrative output on a bundled sample film. Switch to Split and drag to compare.
            </figcaption>
        </figure>
    );
}

function Callout({ left, top, delay, name, score, cutoff }: { left: string; top: string; delay: string; name: string; score: string; cutoff: string }) {
    return (
        <span
            className="fade-in absolute flex -translate-y-1/2 items-baseline gap-2 whitespace-nowrap rounded-md border border-white/20 bg-[#0b0f12]/85 px-2.5 py-1.5 text-[13px] text-[#e3e9ed] backdrop-blur"
            style={{ left, top, animationDelay: delay }}
        >
            <b className="font-semibold">{name}</b>
            <span className="num font-mono text-[12px] text-[#f5b24e] [font-stretch:80%]">{score}</span>
            <span className="hidden font-mono text-[11px] text-[#8a99a4] [font-stretch:80%] sm:inline">cutoff {cutoff}</span>
        </span>
    );
}

function PipelineViz({ kind }: { kind: (typeof PIPELINE)[number]["viz"] }) {
    const box = "relative mt-5 h-[92px] overflow-hidden rounded-lg border border-line bg-surface-raised";
    if (kind === "film" || kind === "map") {
        return (
            <div className={cn(box, "border-film bg-film")}>
                <img src={kind === "film" ? "/example2.png" : "/example1.png"} alt="" className="absolute inset-0 h-full w-full object-cover opacity-90" />
                {kind === "map" && <img src="/sample-attention.png" alt="" className="absolute inset-0 h-full w-full object-cover mix-blend-screen opacity-80" />}
            </div>
        );
    }
    if (kind === "docs") {
        return (
            <div className={cn(box, "flex flex-wrap content-center gap-1.5 p-2.5")}>
                {["BTS Pleural", "NICE CHF", "Fleischner 2017", "AHA/ACC HF", "BTS Nodules"].map((d) => (
                    <span key={d} className="rounded border border-line bg-surface-sunk px-1.5 py-0.5 text-[11.5px] text-fg-soft">{d}</span>
                ))}
            </div>
        );
    }
    return (
        <div className={cn(box, "flex flex-col justify-center gap-[7px] px-3.5")}>
            {SPECIMEN.slice(0, 6).map((p) => (
                <span
                    key={p.label}
                    className={cn("block h-[5px] rounded-full", kind === "cutoffs" && p.prob >= p.threshold ? "bg-accent" : "bg-bar")}
                    style={{ width: `${20 + p.prob * 80}%` }}
                />
            ))}
        </div>
    );
}
