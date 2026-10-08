import React from "react";
import { Moon, Sun } from "lucide-react";
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
        <header className="sticky top-0 z-40 border-b border-line bg-surface/85 backdrop-blur-md print:hidden">
            <div className="mx-auto flex h-[60px] max-w-[1280px] items-center gap-6 px-4 sm:gap-10 sm:px-6 lg:px-10">
                <Logo onClick={() => setStep("landing")} />
                <nav className="flex items-center gap-5 self-stretch sm:gap-7" aria-label="Primary">
                    {items.map((it) => {
                        const active = activeId === it.id;
                        return (
                            <button
                                key={it.id}
                                disabled={it.disabled}
                                onClick={() => setStep(it.id)}
                                aria-current={active ? "page" : undefined}
                                className={cn(
                                    "relative h-full text-[14px] transition-colors",
                                    it.id !== "upload" && it.id !== "results" && "hidden sm:block",
                                    active ? "text-fg" : "text-fg-muted hover:text-fg",
                                    "disabled:cursor-not-allowed disabled:text-fg-faint/60",
                                    "after:absolute after:inset-x-0 after:bottom-[-1px] after:h-[2px] after:rounded-full after:bg-accent",
                                    active ? "after:block" : "after:hidden"
                                )}
                            >
                                {it.label}
                            </button>
                        );
                    })}
                </nav>
                <div className="ml-auto flex shrink-0 items-center gap-3">
                    {IS_DEMO && <span className="rounded-full border border-accent/50 px-2.5 py-1 text-[12px] text-accent-ink">Mock mode</span>}
                    <span className="hidden rounded-full border border-line px-2.5 py-1 text-[12.5px] text-fg-muted lg:inline">
                        Research prototype · not for clinical use
                    </span>
                    <ThemeToggle />
                </div>
            </div>
        </header>
    );
}

function ThemeToggle() {
    const [theme, setTheme] = useTheme();
    const opt = (t: Theme, Icon: typeof Sun, label: string) => (
        <button
            onClick={() => setTheme(t)}
            aria-pressed={theme === t}
            aria-label={label}
            title={label}
            className={cn(
                "grid h-[26px] w-[30px] place-items-center rounded-full transition-colors",
                theme === t ? "bg-surface-sunk text-fg" : "text-fg-muted hover:text-fg"
            )}
        >
            <Icon className="h-[15px] w-[15px]" strokeWidth={1.75} />
        </button>
    );
    return (
        <div className="flex rounded-full border border-line p-[2px]" role="group" aria-label="Colour theme">
            {opt("light", Sun, "Light theme")}
            {opt("dark", Moon, "Dark theme")}
        </div>
    );
}
