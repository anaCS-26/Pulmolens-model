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
        // Plex for the instrument, Newsreader for anything meant to be read.
        sans: ['"IBM Plex Sans"', 'ui-sans-serif', 'system-ui', 'Segoe UI', 'Roboto', 'Arial', 'sans-serif'],
        serif: ['Newsreader', 'ui-serif', 'Georgia', 'serif'],
        mono: ['"IBM Plex Mono"', 'ui-monospace', 'SFMono-Regular', 'Menlo', 'Consolas', 'monospace'],
      },
      colors: {
        paper: { DEFAULT: v('paper'), raised: v('paper-raised'), sunk: v('paper-sunk') },
        ink: { DEFAULT: v('ink'), soft: v('ink-soft'), muted: v('ink-muted'), faint: v('ink-faint') },
        rule: { DEFAULT: v('rule'), strong: v('rule-strong') },
        // The red of a radiologist's grease pencil: used only for flags and warnings.
        marker: { DEFAULT: v('marker'), dark: v('marker-dark'), soft: v('marker-soft') },
        // Film panels stay black in both themes.
        film: '#0c0c0b',
      },
      borderRadius: {
        DEFAULT: '3px',
      },
    },
  },
  plugins: [],
};
