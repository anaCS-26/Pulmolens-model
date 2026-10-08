import React, { useEffect, useRef, useState } from "react";
import { ThinkingLoader } from "../ui/ThinkingLoader";

// Reveals `target` one character at a time. Adapts cadence when the
// streaming backlog grows so we never fall far behind the model.
function useTypewriter(target: string, baseCps = 70): string {
    const [shown, setShown] = useState("");
    const shownRef = useRef("");
    shownRef.current = shown;

    // If the upstream buffer was reset/replaced (no longer a prefix), restart.
    useEffect(() => {
        if (target.length === 0) {
            if (shownRef.current.length !== 0) setShown("");
            return;
        }
        if (!target.startsWith(shownRef.current)) {
            setShown("");
        }
    }, [target]);

    useEffect(() => {
        if (shown.length >= target.length) return;
        let raf = 0;
        let last = performance.now();
        const tick = (now: number) => {
            const dt = now - last;
            last = now;
            setShown((prev) => {
                if (prev.length >= target.length) return prev;
                const remaining = target.length - prev.length;
                // Catch-up: speed climbs with backlog so a 500-char burst drains in ~1s.
                const cps = baseCps + Math.max(0, remaining - 30) * 6;
                const advance = Math.max(1, Math.round((dt / 1000) * cps));
                return target.slice(0, Math.min(target.length, prev.length + advance));
            });
            raf = requestAnimationFrame(tick);
        };
        raf = requestAnimationFrame(tick);
        return () => cancelAnimationFrame(raf);
    }, [target, shown.length, baseCps]);

    return shown;
}

// Render `text` as individually-animated <span>s keyed by absolute offset
// `start`. React reuses spans across re-renders for offsets that haven't
// changed, so each char animates exactly once on first mount.
function AnimatedChars({ text, start, bold, tailStart }: { text: string; start: number; bold?: boolean; tailStart: number }) {
    const Wrap = bold ? 'strong' : 'span';
    const wrapClass = bold ? "font-semibold text-ink" : undefined;
    const splitAt = Math.max(0, Math.min(text.length, tailStart - start));
    const settled = text.slice(0, splitAt);
    const tail = text.slice(splitAt);
    return (
        <Wrap className={wrapClass}>
            {settled}
            {Array.from(tail).map((ch, k) => (
                <span key={start + splitAt + k} className="char-fade-up">
                    {ch === ' ' ? ' ' : ch}
                </span>
            ))}
        </Wrap>
    );
}

function MarkdownLite({ text, isStreaming }: { text: string; isStreaming?: boolean }) {
    const displayed = useTypewriter(text || "");
    const stillTyping = !!isStreaming || displayed.length < (text?.length || 0);
    // Only the last ANIMATED_TAIL chars get per-char animation while streaming;
    // older chars collapse back to plain text so we don't accumulate thousands
    // of inline-block spans (each carrying a finished filter/transform animation)
    // that would otherwise stall scroll-time compositing.
    const ANIMATED_TAIL = 80;
    const tailStart = stillTyping ? Math.max(0, displayed.length - ANIMATED_TAIL) : displayed.length;

    if (!displayed) return null;
    const lines = displayed.split("\n");

    // Track the running absolute character offset so each <span> key is
    // stable across re-renders so already-mounted chars don't re-animate.
    let offset = 0;

    return (
        <div className="space-y-1">
            {lines.map((l, i) => {
                const isLast = i === lines.length - 1;
                const lineStart = offset;
                offset += l.length + 1; // +1 for the newline we split on

                const trimmed = l.trim();
                const isBullet = trimmed.startsWith("*") && !trimmed.startsWith("**");
                const indentDelta = l.indexOf(trimmed);
                const content = isBullet ? trimmed.slice(1).trimStart() : trimmed;

                if (!content) return <div key={`gap-${lineStart}`} className="h-3" />;

                // Compute where `content` begins inside the original line so each
                // span carries a globally-unique, stable key.
                const contentStartInLine = isBullet
                    ? l.indexOf("*") + 1 + (l.slice(l.indexOf("*") + 1).length - l.slice(l.indexOf("*") + 1).trimStart().length)
                    : indentDelta;
                let cursor = lineStart + contentStartInLine;

                const parts = content.split(/(\*\*[^*]+\*\*)/g).filter(Boolean);
                const rendered = parts.map((p, j) => {
                    const isBold = /^\*\*[^*]+\*\*$/.test(p);
                    const inner = isBold ? p.slice(2, -2) : p;
                    const node = (
                        <AnimatedChars
                            key={`${lineStart}-${j}`}
                            text={inner}
                            start={cursor + (isBold ? 2 : 0)}
                            bold={isBold}
                            tailStart={tailStart}
                        />
                    );
                    cursor += p.length;
                    return node;
                });

                const caret = stillTyping && isLast ? (
                    <span className="inline-block w-[2px] h-[1em] bg-marker ml-0.5 align-[-0.15em] animate-caret-blink" />
                ) : null;

                if (isBullet) {
                    return (
                        <div key={`b-${lineStart}`} className="flex items-start gap-2.5 pl-1">
                            <span className="mt-0 shrink-0 font-sans text-ink-faint">–</span>
                            <div>{rendered}{caret}</div>
                        </div>
                    );
                }
                return <div key={`p-${lineStart}`} className="mb-1.5 last:mb-0">{rendered}{caret}</div>;
            })}
        </div>
    );
}

interface ClinicalReportProps {
    report: string | null;
    sources: string[];
    isSummarizing?: boolean;
    hasOverlay: boolean;
}

export function ClinicalReport({ report, sources, isSummarizing, hasOverlay }: ClinicalReportProps) {
    const hasText = typeof report === "string" && report.length > 0;

    if (isSummarizing && !hasText) {
        return (
            <div className="py-2">
                <ThinkingLoader />
                <p className="mt-3 max-w-[48ch] text-[13.5px] text-ink-muted">Retrieving guidance and drafting a synthesis. The findings above are ready to review meanwhile.</p>
            </div>
        );
    }

    if (!hasText) {
        return (
            <p className="text-[14.5px] text-ink-muted">
                No synthesis was generated for this image.{" "}
                {hasOverlay ? "The summarisation service didn't return a report." : "A synthesis needs the attention overlay, which wasn't returned."}
            </p>
        );
    }

    return (
        <div>
            <div className="max-w-[62ch] font-serif text-[18px] leading-[1.65] text-ink">
                <MarkdownLite text={report} isStreaming={isSummarizing} />
            </div>

            {sources.length > 0 && (
                <ol className="mt-6 space-y-1 border-t border-rule pt-3 text-[13px] leading-snug text-ink-muted">
                    {sources.map((s, idx) => (
                        <li key={idx} className="grid grid-cols-[1.5rem_1fr]">
                            <span className="font-mono text-marker">{idx + 1}</span>
                            <span>{s}</span>
                        </li>
                    ))}
                </ol>
            )}
        </div>
    );
}
