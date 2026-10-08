import { useEffect, useState } from "react";

export type Theme = "light" | "dark";

// Kept in sync with the inline script in index.html, which applies the theme
// before first paint so a dark-mode visitor never sees a flash of white.
const KEY = "pulmolens.theme";
const META_COLOR: Record<Theme, string> = { light: "#f1f4f6", dark: "#0b0f12" };

function apply(t: Theme) {
    document.documentElement.classList.toggle("dark", t === "dark");
    document.querySelector('meta[name="theme-color"]')?.setAttribute("content", META_COLOR[t]);
}

function stored(): Theme | null {
    try {
        const v = localStorage.getItem(KEY);
        return v === "light" || v === "dark" ? v : null;
    } catch {
        return null;
    }
}

/** Follows the OS setting until the user picks a theme, then remembers the choice. */
export function useTheme(): [Theme, (t: Theme) => void] {
    const [theme, setThemeState] = useState<Theme>(() =>
        document.documentElement.classList.contains("dark") ? "dark" : "light"
    );

    useEffect(() => {
        if (stored()) return;
        const mq = window.matchMedia("(prefers-color-scheme: dark)");
        const onChange = (e: MediaQueryListEvent) => {
            if (stored()) return;
            const t: Theme = e.matches ? "dark" : "light";
            apply(t);
            setThemeState(t);
        };
        mq.addEventListener("change", onChange);
        return () => mq.removeEventListener("change", onChange);
    }, []);

    const setTheme = (t: Theme) => {
        try { localStorage.setItem(KEY, t); } catch { /* private mode: still switch for this page */ }
        apply(t);
        setThemeState(t);
    };

    return [theme, setTheme];
}
