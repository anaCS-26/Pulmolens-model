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
    <div className="inline-flex items-center gap-3 py-1.5 px-3 bg-white/80 backdrop-blur-xl rounded-lg border border-slate-200 shadow-sm animate-in fade-in duration-500">
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
          className="text-slate-200"
        />
        <path
          d={EKG_PATH}
          pathLength={100}
          fill="none"
          stroke="rgb(79 70 229)"
          strokeWidth="1.8"
          strokeLinecap="round"
          strokeLinejoin="round"
          className="ekg-sweep"
        />
      </svg>

      <h3
        key={verbIndex}
        className="text-sm font-medium text-slate-800 tracking-tight leading-none"
      >
        <CharacterAnimator text={MEDICAL_VERBS[verbIndex]} />
      </h3>

      <style dangerouslySetInnerHTML={{ __html: `
        @keyframes ekg-sweep {
          0%   { stroke-dashoffset: 14; }
          100% { stroke-dashoffset: -100; }
        }
        .ekg-sweep {
          stroke-dasharray: 14 100;
          stroke-dashoffset: 14;
          filter: drop-shadow(0 0 2.5px rgba(99, 102, 241, 0.85));
          animation: ekg-sweep 2.4s cubic-bezier(0.45, 0, 0.55, 1) infinite;
        }
      `}} />
    </div>
  );
};
