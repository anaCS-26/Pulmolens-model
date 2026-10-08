import React from "react";
import { CLINICIAN_COPY, GUIDELINE_TAGS } from "../../data/constants";
import { DEFAULT_GUIDANCE, GUIDANCE_BULLETS } from "../../data/guidance";
import { prettyLabel, pct } from "../../utils/format";

interface GuidedCardProps {
    label: string;
    prob: number;
}

export function GuidedCard({ label, prob }: GuidedCardProps) {
    const blurb = CLINICIAN_COPY[label] || "";
    const tags = (GUIDELINE_TAGS[label] || []).filter((t) => t !== "None");
    const bullets = GUIDANCE_BULLETS[label] || DEFAULT_GUIDANCE;

    return (
        <article className="border-b border-rule py-5 first:pt-0">
            <h3 className="flex items-baseline gap-3">
                <span className="font-serif text-[21px]">{prettyLabel(label)}</span>
                <span className="num font-mono text-[12px] text-marker">{pct(prob)}</span>
            </h3>
            <p className="mt-1 max-w-[62ch] text-[13.5px] leading-relaxed text-ink-muted">{blurb}</p>
            <ol className="mt-3 space-y-1.5 text-[14.5px]">
                {bullets.map((t, i) => (
                    <li key={i} className="grid grid-cols-[1.5rem_1fr]">
                        <span className="font-mono text-[12px] leading-[1.8] text-ink-faint">{i + 1}</span>
                        <span>{t}</span>
                    </li>
                ))}
            </ol>
            {tags.length > 0 && (
                <p className="mt-3 font-mono text-[11.5px] text-ink-muted">Sources: {tags.join("; ")}</p>
            )}
        </article>
    );
}
