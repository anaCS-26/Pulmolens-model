/** @type {import('tailwindcss').Config} */

// Colours are RGB triplets in CSS variables (see index.css) so one `dark`
// class on <html> re-themes everything and opacity modifiers still work.
const v = (name) => `rgb(var(--${name}) / <alpha-value>)`;

export default {
  darkMode: 'class',
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      fontFamily: {
        // Mona Sans for everything; Martian Mono only for annotations burned onto film.
        sans: ['"Mona Sans"', 'ui-sans-serif', 'system-ui', '-apple-system', 'Segoe UI', 'Roboto', 'sans-serif'],
        mono: ['"Martian Mono"', 'ui-monospace', 'SFMono-Regular', 'Menlo', 'Consolas', 'monospace'],
      },
      colors: {
        surface: { DEFAULT: v('surface'), raised: v('surface-raised'), sunk: v('surface-sunk') },
        fg: { DEFAULT: v('fg'), soft: v('fg-soft'), muted: v('fg-muted'), faint: v('fg-faint') },
        line: { DEFAULT: v('line'), soft: v('line-soft') },
        // High-contrast fill for primary buttons: ink on light, lightbox white on dark.
        solid: { DEFAULT: v('solid'), fg: v('solid-fg') },
        bar: v('bar'),
        // Amber is the secondary colour, and the colour of a score above its cutoff.
        accent: { DEFAULT: v('accent'), ink: v('accent-ink') },
        // Red is reserved for urgent clinical safety messages.
        urgent: v('urgent'),
        // Film panels stay black in both themes.
        film: '#000000',
      },
      borderRadius: {
        DEFAULT: '6px',
      },
    },
  },
  plugins: [],
};
