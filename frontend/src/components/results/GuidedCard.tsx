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
        <article className="border-b border-line-soft py-5 first:pt-0 last:border-0">
            <h3 className="flex items-center gap-2.5">
                <span className="h-1.5 w-1.5 rounded-full bg-accent" aria-hidden />
                <span className="text-[16px] font-semibold">{prettyLabel(label)}</span>
                <span className="num font-mono text-[12px] text-accent-ink [font-stretch:80%]">{pct(prob)}</span>
            </h3>
            <p className="mt-1.5 max-w-[62ch] text-[13.5px] leading-relaxed text-fg-muted">{blurb}</p>
            <ul className="mt-3 space-y-1.5 text-[14.5px] text-fg-soft">
                {bullets.map((t, i) => (
                    <li key={i} className="grid grid-cols-[1rem_1fr]">
                        <span className="text-fg-faint">–</span>
                        <span>{t}</span>
                    </li>
                ))}
            </ul>
            {tags.length > 0 && <p className="mt-3 text-[12.5px] text-fg-muted">Sources: {tags.join("; ")}</p>}
        </article>
    );
}
