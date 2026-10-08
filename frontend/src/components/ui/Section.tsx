import React from "react";
import { cn } from "../../utils/cn";

interface SectionProps {
    kicker: string;
    title: React.ReactNode;
    sub?: React.ReactNode;
    id?: string;
    children?: React.ReactNode;
    className?: string;
}

/** A full-width band with an amber kicker, a display heading and an optional lede. */
export function Section({ kicker, title, sub, id, children, className }: SectionProps) {
    return (
        <section id={id} className={cn("scroll-mt-16 border-t border-line py-16 md:py-28", className)}>
            <div className="mx-auto max-w-[1280px] px-4 sm:px-6 lg:px-10">
                <p className="kicker">{kicker}</p>
                <h2 className="display mt-3 max-w-[18em] text-[32px] leading-[1.02] md:text-[50px]">{title}</h2>
                {sub && <p className="mt-5 max-w-[40em] text-[17px] text-fg-soft">{sub}</p>}
                {children}
            </div>
        </section>
    );
}
