import React, { useState, useEffect } from 'react';

const MEDICAL_VERBS = [
  "Auscultating",
  "Palpating",
  "Stabilizing",
  "Triaging",
  "Diagnosing",
  "Correlating",
  "Synthesizing",
  "Examining",
  "Scanning",
  "Probing",
  "Consulting",
  "Analyzing",
  "Localizing",
  "Differentiating",
];

// Compact EKG: flat → tight QRS (Q-dip, R-spike, S-dip) → flat. The spike
// sits in the middle of the trace so the bright sweep crosses it cleanly.
const EKG_PATH = "M0 9 H22 l1 1 l1 -7 l2 14 l1 -7 l1 -1 H64";

const CharacterAnimator: React.FC<{ text: string; trailingDots?: number }> = ({
  text,
  trailingDots = 3,
}) => {
  const chars = [...text.split(''), ...Array(trailingDots).fill('.')];
  return (
    <span className="inline-flex whitespace-nowrap" aria-label={text + '...'}>
      {chars.map((char, index) => (
        <span
          key={`${text}-${index}`}
          className="char-fade-up"
          style={{ animationDelay: `${index * 35}ms` }}
        >
          {char === ' ' ? ' ' : char}
        </span>
      ))}
    </span>
  );
};

export const ThinkingLoader: React.FC = () => {
  const [verbIndex, setVerbIndex] = useState(0);

  useEffect(() => {
    const id = setInterval(() => {
      setVerbIndex((prev) => (prev + 1) % MEDICAL_VERBS.length);
    }, 2800);
    return () => clearInterval(id);
  }, []);

  return (
    <div className="inline-flex items-center gap-3">
      <svg
        viewBox="0 0 64 18"
        className="h-4 w-[52px] shrink-0 overflow-visible"
        aria-hidden="true"
      >
        <path
          d={EKG_PATH}
          fill="none"
          stroke="currentColor"
          strokeWidth="1.2"
          strokeLinecap="round"
          strokeLinejoin="round"
          className="text-rule-strong"
        />
        <path
          d={EKG_PATH}
          pathLength={100}
          fill="none"
          stroke="#c2401c"
          strokeWidth="1.6"
          strokeLinecap="round"
          strokeLinejoin="round"
          className="ekg-sweep"
        />
      </svg>

      <div className="flex items-baseline gap-2 font-mono text-[12px] text-ink-muted">
        <span className="text-ink">Drafting synthesis</span>
        <span
          key={verbIndex}
          aria-live="polite"
        >
          <CharacterAnimator text={MEDICAL_VERBS[verbIndex].toLowerCase()} />
        </span>
      </div>
    </div>
  );
};
