import React, { useEffect, useState } from "react";
import { Check } from "lucide-react";
import { cn } from "../utils/cn";
import { FlowProgress } from "../components/layout/FlowProgress";
import { FilmPanel } from "../components/viewer/FilmPanel";

interface ProcessingProps {
    imageURL: string | null;
    fileName?: string;
}

const IS_DEMO = import.meta.env.VITE_DEMO_MODE === "true";

// The server returns everything in one response, so stage timing is paced
// locally; the last stage stays active until the response actually arrives.
const STAGES = [
    { label: "Uploading image", at: 0 },
    { label: "Preprocessing", at: 0.9 },
    { label: "Scoring 14 findings", at: 1.8 },
    { label: "Computing attention overlay", at: 3.0 },
];

const COLD_START_AFTER = 8;

export function Processing({ imageURL, fileName }: ProcessingProps) {
    const [t, setT] = useState(0);
    useEffect(() => {
        const start = performance.now();
        const id = setInterval(() => setT((performance.now() - start) / 1000), 100);
        return () => clearInterval(id);
    }, []);

    const activeIdx = STAGES.reduce((acc, s, i) => (t >= s.at ? i : acc), 0);

    return (
        <section className="mx-auto max-w-[1280px] px-4 pb-24 pt-10 sm:px-6 lg:px-10">
            <FlowProgress current={2} />
            <h1 className="display mt-7 text-[40px] leading-none md:text-[54px]">Analysing</h1>

            <div className="mt-10 grid grid-cols-1 gap-x-12 gap-y-10 lg:grid-cols-12">
                <FilmPanel
                    className="aspect-square w-full max-w-[620px] lg:col-span-7"
                    tl={fileName}
                    tr={`${t.toFixed(1)} s`}
                    bl={IS_DEMO ? "Mock mode" : "Running on server"}
                >
                    {imageURL && <img src={imageURL} alt="Uploaded chest X-ray" className="h-full w-full object-contain opacity-80" />}
                    <div aria-hidden className="scanline absolute inset-x-0 h-px bg-[rgb(242,162,58)] shadow-[0_0_12px_2px_rgba(242,162,58,0.55)]" />
                </FilmPanel>

                <div className="lg:col-span-5">
                    <div className="rounded-xl border border-line bg-surface-raised p-5 sm:p-6">
                        <h2 className="text-[16px] font-semibold">Pipeline</h2>
                        <ol className="mt-4">
                            {STAGES.map((s, i) => {
                                const done = i < activeIdx;
                                const active = i === activeIdx;
                                return (
                                    <li key={s.label} className={cn("grid grid-cols-[1.75rem_1fr_auto] items-center border-b border-line-soft py-3 text-[14.5px] last:border-0", !done && !active && "text-fg-faint")}>
                                        <span className="grid h-5 w-5 place-items-center">
                                            {done ? (
                                                <Check className="h-4 w-4 text-fg-muted" strokeWidth={2.25} />
                                            ) : active ? (
                                                <span className="relative flex h-2.5 w-2.5">
                                                    <span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-accent opacity-60" />
                                                    <span className="relative inline-flex h-2.5 w-2.5 rounded-full bg-accent" />
                                                </span>
                                            ) : (
                                                <span className="h-2 w-2 rounded-full border border-line" />
                                            )}
                                        </span>
                                        <span className={cn(active ? "font-medium text-fg" : done ? "text-fg-soft" : "")}>{s.label}</span>
                                        <span className={cn("num text-[13px]", active ? "text-accent-ink" : "text-fg-faint")}>
                                            {done ? "Done" : active ? "Running" : "Queued"}
                                        </span>
                                    </li>
                                );
                            })}
                        </ol>
                    </div>

                    {t > COLD_START_AFTER && !IS_DEMO && (
                        <p className="mt-5 rounded-lg border border-accent/40 bg-accent/[0.06] px-4 py-3 text-[14px] leading-relaxed text-fg-soft">
                            The server is warming up after a period of inactivity. Please stay on this screen while it spins up.
                        </p>
                    )}
                </div>
            </div>
        </section>
    );
}
