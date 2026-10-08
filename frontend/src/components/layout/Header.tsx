import React from "react";
import { cn } from "../../utils/cn";
import { Step } from "../../types";
import { Logo } from "../ui/Logo";
import { Theme, useTheme } from "../../theme";

interface HeaderProps {
    step: Step;
    setStep: (s: Step) => void;
    hasResults: boolean;
}

const IS_DEMO = import.meta.env.VITE_DEMO_MODE === "true";

export function Header({ step, setStep, hasResults }: HeaderProps) {
    // "Analyse" covers the whole upload -> processing hand-off.
    const activeId: Step = step === "processing" ? "upload" : step;
    const items: { id: Step; label: string; disabled?: boolean }[] = [
        { id: "landing", label: "Overview" },
        { id: "about", label: "About" },
        { id: "upload", label: "Analyse" },
        { id: "results", label: "Report", disabled: !hasResults },
    ];

    return (
        <header className="sticky top-0 z-40 border-b border-rule bg-paper/95 backdrop-blur-[2px] print:hidden">
            <div className="mx-auto flex h-14 max-w-[1200px] items-center gap-5 px-4 sm:gap-10 sm:px-6">
                <Logo onClick={() => setStep("landing")} />
                <nav className="flex items-center gap-4 self-stretch sm:gap-6" aria-label="Primary">
                    {items.map((it) => {
                        const active = activeId === it.id;
                        return (
                            <button
                                key={it.id}
                                disabled={it.disabled}
                                onClick={() => setStep(it.id)}
                                aria-current={active ? "page" : undefined}
                                className={cn(
                                    "relative h-full text-[13.5px] transition-colors",
                                    it.id !== "upload" && it.id !== "results" && "hidden sm:block",
                                    active ? "text-ink font-medium" : "text-ink-muted hover:text-ink",
                                    "disabled:cursor-not-allowed disabled:text-ink-faint/60",
                                    "after:absolute after:inset-x-0 after:bottom-[-1px] after:h-[2px] after:bg-ink",
                                    active ? "after:block" : "after:hidden"
                                )}
                            >
                                {it.label}
                            </button>
                        );
                    })}
                </nav>
                <div className="ml-auto flex shrink-0 items-center gap-6">
                    <div className="hidden items-center gap-3 lg:flex">
                        {IS_DEMO && <span className="label !text-marker">Mock mode</span>}
                        <span className="label">Research prototype · not for clinical use</span>
                    </div>
                    <ThemeToggle />
                </div>
            </div>
        </header>
    );
}

function ThemeToggle() {
    const [theme, setTheme] = useTheme();
    const opt = (t: Theme, label: string) => (
        <button
            onClick={() => setTheme(t)}
            aria-pressed={theme === t}
            className={cn("transition-colors", theme === t ? "text-ink underline underline-offset-[5px]" : "text-ink-faint hover:text-ink")}
        >
            {label}
        </button>
    );
    return (
        <div className="flex items-center gap-1.5 font-mono text-[11.5px]" role="group" aria-label="Colour theme">
            {opt("light", "Light")}
            <span className="text-ink-faint">/</span>
            {opt("dark", "Dark")}
        </div>
    );
}
