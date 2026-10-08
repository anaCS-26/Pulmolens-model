import React from "react";

/** Wordmark with the app mark: lungs and trachea on a film plate, with an amber attention hotspot. */
export function Logo({ onClick }: { onClick?: () => void }) {
    return (
        <button onClick={onClick} className="flex items-center gap-2.5" aria-label="PulmoLens home">
            <img src="/favicon.svg" alt="" aria-hidden className="h-[22px] w-[22px]" />
            <span className="text-[17px] font-semibold leading-none tracking-[-0.01em] text-fg [font-stretch:112%]">PulmoLens</span>
        </button>
    );
}
