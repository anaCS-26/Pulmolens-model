import React from "react";

/** Wordmark with a small film plate, graded the way a radiograph is. */
export function Logo({ onClick }: { onClick?: () => void }) {
    return (
        <button onClick={onClick} className="flex items-center gap-2.5" aria-label="PulmoLens home">
            <span
                aria-hidden
                className="h-[21px] w-[17px] rounded-[3px] bg-[linear-gradient(160deg,#cfd9df,#4b5860_72%)] shadow-[inset_0_0_0_1px_rgba(255,255,255,0.25)]"
            />
            <span className="text-[17px] font-semibold leading-none tracking-[-0.01em] text-fg [font-stretch:112%]">PulmoLens</span>
        </button>
    );
}
