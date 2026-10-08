import React from "react";
import { cn } from "../../utils/cn";

interface FilmPanelProps {
    children: React.ReactNode;
    tl?: React.ReactNode;
    tr?: React.ReactNode;
    bl?: React.ReactNode;
    br?: React.ReactNode;
    className?: string;
}

/** Black viewing panel with PACS-style annotations in the four corners. */
export function FilmPanel({ children, tl, tr, bl, br, className }: FilmPanelProps) {
    const corner = "pointer-events-none absolute z-10 film-text whitespace-pre-line";
    return (
        <div className={cn("relative overflow-hidden rounded-lg bg-film ring-1 ring-line", className)}>
            {children}
            {tl && <div className={cn(corner, "left-3.5 top-3")}>{tl}</div>}
            {tr && <div className={cn(corner, "right-3.5 top-3 text-right")}>{tr}</div>}
            {bl && <div className={cn(corner, "bottom-3 left-3.5")}>{bl}</div>}
            {br && <div className={cn(corner, "bottom-3 right-3.5 text-right")}>{br}</div>}
        </div>
    );
}

/** The jet scale the attention map is drawn in, low to high. */
export const JET_GRADIENT =
    "linear-gradient(90deg,#00007f,#0000ff 12%,#007fff 30%,#00ffff 38%,#7fff7f 50%,#ffff00 62%,#ff7f00 75%,#ff0000 88%,#7f0000)";

export function JetLegend({ className }: { className?: string }) {
    return (
        <span className={cn("inline-flex items-center gap-2 text-[12px] text-fg-muted", className)}>
            Less
            <span aria-hidden className="h-[5px] w-16 rounded-full" style={{ background: JET_GRADIENT }} />
            More influence
        </span>
    );
}
