/** @type {import('tailwindcss').Config} */
export default {
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
        paper: { DEFAULT: '#f2efe8', raised: '#f8f6f1', sunk: '#e8e4da' },
        ink: { DEFAULT: '#1b1a17', soft: '#3a3832', muted: '#6c685e', faint: '#9b968a' },
        rule: { DEFAULT: '#d5d0c4', strong: '#b9b3a5' },
        // The red of a radiologist's grease pencil: used only for flags and warnings.
        marker: { DEFAULT: '#c2401c', dark: '#9e3215', soft: '#f1ddd3' },
        film: '#0c0c0b',
      },
      borderRadius: {
        DEFAULT: '3px',
      },
    },
  },
  plugins: [],
};
