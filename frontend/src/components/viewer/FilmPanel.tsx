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
        <div className={cn("relative overflow-hidden bg-film", className)}>
            {children}
            {tl && <div className={cn(corner, "left-3 top-2.5")}>{tl}</div>}
            {tr && <div className={cn(corner, "right-3 top-2.5 text-right")}>{tr}</div>}
            {bl && <div className={cn(corner, "bottom-2.5 left-3")}>{bl}</div>}
            {br && <div className={cn(corner, "bottom-2.5 right-3 text-right")}>{br}</div>}
        </div>
    );
}
