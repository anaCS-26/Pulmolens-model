import React from "react";
import { cn } from "../../utils/cn";

interface TabsProps<T extends string> {
    value: T;
    onChange: (v: T) => void;
    options: { value: T; label: React.ReactNode; disabled?: boolean }[];
    className?: string;
    tone?: "paper" | "film";
}

/** Text tabs with an underline, as on a printed form. */
export function Tabs<T extends string>({ value, onChange, options, className, tone = "paper" }: TabsProps<T>) {
    return (
        <div role="tablist" className={cn("flex items-center gap-5", className)}>
            {options.map((o) => {
                const active = o.value === value;
                return (
                    <button
                        key={o.value}
                        role="tab"
                        aria-selected={active}
                        disabled={o.disabled}
                        onClick={() => onChange(o.value)}
                        className={cn(
                            "relative py-1 text-[13px] transition-colors disabled:opacity-35 disabled:cursor-not-allowed",
                            tone === "paper"
                                ? active ? "text-ink font-medium" : "text-ink-muted hover:text-ink"
                                : active ? "text-white font-medium" : "text-white/50 hover:text-white",
                            "after:absolute after:inset-x-0 after:-bottom-px after:h-px after:transition-opacity",
                            tone === "paper" ? "after:bg-ink" : "after:bg-white",
                            active ? "after:opacity-100" : "after:opacity-0"
                        )}
                    >
                        {o.label}
                    </button>
                );
            })}
        </div>
    );
}
