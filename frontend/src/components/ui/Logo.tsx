import React from "react";

/** Plain wordmark. The circled cross is a viewfinder reticle, not a medical cross. */
export function Logo({ onClick }: { onClick?: () => void }) {
    return (
        <button onClick={onClick} className="flex items-baseline gap-2" aria-label="PulmoLens home">
            <svg viewBox="0 0 16 16" className="h-[13px] w-[13px] translate-y-[1px] text-marker" fill="none" stroke="currentColor" strokeWidth="1.6" aria-hidden>
                <circle cx="8" cy="8" r="6.2" />
                <path d="M8 0.5v4M8 11.5v4M0.5 8h4M11.5 8h4" />
            </svg>
            <span className="font-serif text-[21px] font-medium leading-none tracking-[-0.01em] text-ink">PulmoLens</span>
        </button>
    );
}
