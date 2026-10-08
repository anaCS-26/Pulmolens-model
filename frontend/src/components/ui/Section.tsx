import React from "react";
import { cn } from "../../utils/cn";

interface SectionProps {
    label: string;
    index?: string;
    id?: string;
    children: React.ReactNode;
    className?: string;
}

/** A ruled row with a narrow label column on the left, like a report form. */
export function Section({ label, index, id, children, className }: SectionProps) {
    return (
        <section id={id} className={cn("grid grid-cols-1 gap-x-10 gap-y-5 border-t border-ink py-10 md:grid-cols-12 md:py-14", className)}>
            <div className="md:col-span-3">
                <div className="label flex gap-3 text-ink">
                    {index && <span className="text-ink-faint">{index}</span>}
                    <span>{label}</span>
                </div>
            </div>
            <div className="md:col-span-9">{children}</div>
        </section>
    );
}
