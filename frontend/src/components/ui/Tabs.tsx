import React from "react";
import { cn } from "../../utils/cn";

interface TabsProps<T extends string> {
    value: T;
    onChange: (v: T) => void;
    options: { value: T; label: React.ReactNode; disabled?: boolean }[];
    className?: string;
    /** "surface" sits on the page; "film" sits on a black viewer. */
    tone?: "surface" | "film";
    label?: string;
}

/** Segmented control, as on a PACS toolbar. */
export function Tabs<T extends string>({ value, onChange, options, className, tone = "surface", label }: TabsProps<T>) {
    const film = tone === "film";
    return (
        <div
            role="group"
            aria-label={label}
            className={cn(
                "inline-flex rounded-lg border p-[3px]",
                film ? "border-white/15 bg-white/5" : "border-line bg-surface-raised",
                className
            )}
        >
            {options.map((o) => {
                const active = o.value === value;
                return (
                    <button
                        key={o.value}
                        aria-pressed={active}
                        disabled={o.disabled}
                        onClick={() => onChange(o.value)}
                        className={cn(
                            "rounded-md px-3 py-1.5 text-[13px] transition-colors disabled:cursor-not-allowed disabled:opacity-35",
                            film
                                ? active ? "bg-white/15 text-white" : "text-white/55 hover:text-white"
                                : active ? "bg-surface-sunk text-fg shadow-[inset_0_0_0_1px_rgb(var(--line))]" : "text-fg-muted hover:text-fg"
                        )}
                    >
                        {o.label}
                    </button>
                );
            })}
        </div>
    );
}
