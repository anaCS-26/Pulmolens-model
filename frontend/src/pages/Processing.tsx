import React, { useEffect, useState } from "react";
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
        <section className="mx-auto max-w-[1200px] px-6 pb-24 pt-12">
            <FlowProgress current={2} />
            <h1 className="mt-6 font-serif text-[44px] font-normal leading-none tracking-[-0.02em] md:text-[56px]">Analysing</h1>

            <div className="mt-12 grid grid-cols-1 gap-x-12 gap-y-10 lg:grid-cols-12">
                <FilmPanel
                    className="aspect-[4/3] lg:col-span-7"
                    tl={fileName}
                    tr={`${t.toFixed(1)} s`}
                    bl={IS_DEMO ? "Mock mode" : "Running on server"}
                >
                    {imageURL && <img src={imageURL} alt="Uploaded chest X-ray" className="h-full w-full object-contain opacity-80" />}
                    <div aria-hidden className="scanline absolute inset-x-0 h-px bg-white/70 shadow-[0_0_10px_rgba(255,255,255,0.5)]" />
                </FilmPanel>

                <div className="lg:col-span-5">
                    <div className="border-t border-ink pt-5">
                        <div className="label">Pipeline</div>
                        <ol className="mt-4 font-mono text-[13px]">
                            {STAGES.map((s, i) => {
                                const done = i < activeIdx;
                                const active = i === activeIdx;
                                return (
                                    <li key={s.label} className={cn("grid grid-cols-[3.5rem_1fr_auto] border-b border-rule py-2.5", !done && !active && "text-ink-faint")}>
                                        <span className="num text-ink-faint">{i <= activeIdx ? `${s.at.toFixed(1)}s` : "—"}</span>
                                        <span className={cn(active && "text-ink")}>{s.label}</span>
                                        <span className={cn(done ? "text-ink-muted" : active ? "text-marker" : "")}>
                                            {done ? "done" : active ? <>running<span className="animate-caret-blink">_</span></> : "queued"}
                                        </span>
                                    </li>
                                );
                            })}
                        </ol>
                    </div>

                    {t > COLD_START_AFTER && !IS_DEMO && (
                        <p className="mt-6 border-l-2 border-marker pl-4 text-[14px] leading-relaxed text-ink-soft">
                            The server is warming up after a period of inactivity. Please stay on this screen while it spins up.
                        </p>
                    )}
                </div>
            </div>
        </section>
    );
}
