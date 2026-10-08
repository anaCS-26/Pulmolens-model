import React from "react";
import { cn } from "../../utils/cn";
import { Step } from "../../types";
import { Logo } from "../ui/Logo";

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
            <div className="mx-auto flex h-14 max-w-[1200px] items-center gap-10 px-6">
                <Logo onClick={() => setStep("landing")} />
                <nav className="flex items-center gap-6 self-stretch" aria-label="Primary">
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
                <div className="ml-auto hidden items-center gap-3 md:flex">
                    {IS_DEMO && <span className="label !text-marker">Mock mode</span>}
                    <span className="label">Research prototype · not for clinical use</span>
                </div>
            </div>
        </header>
    );
}
