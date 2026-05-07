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

// EKG path: flat → small P bump → flat → QRS spike → flat → T bump → flat.
// viewBox is 120x24; line baseline y=12.
const EKG_PATH =
  "M0 12 H30 l2 -2 l2 2 H50 l2 6 l2 -14 l2 14 l2 -6 H78 l3 -3 l3 3 H120";

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
          className="inline-block animate-char-fade-in"
          style={{
            animationDelay: `${index * 35}ms`,
            animationFillMode: 'both',
          }}
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
    const verbInterval = setInterval(() => {
      setVerbIndex((prev) => (prev + 1) % MEDICAL_VERBS.length);
    }, 2800);
    return () => clearInterval(verbInterval);
  }, []);

  return (
    <div className="inline-flex items-center gap-3 py-2 px-3.5 bg-white/70 backdrop-blur-xl rounded-lg border border-slate-200/80 shadow-sm mx-auto animate-in fade-in duration-500">
      {/* EKG trace */}
      <svg
        viewBox="0 0 120 24"
        className="h-5 w-[60px] shrink-0 overflow-visible"
        aria-hidden="true"
      >
        {/* Muted baseline trace */}
        <path
          d={EKG_PATH}
          fill="none"
          stroke="currentColor"
          strokeWidth="1"
          strokeLinecap="round"
          strokeLinejoin="round"
          className="text-slate-200"
        />
        {/* Animated bright sweep — short dash travels along the path */}
        <path
          d={EKG_PATH}
          pathLength={100}
          fill="none"
          stroke="rgb(79 70 229)"
          strokeWidth="1.6"
          strokeLinecap="round"
          strokeLinejoin="round"
          className="ekg-sweep"
        />
      </svg>

      <div className="flex flex-col leading-tight">
        <h3
          key={verbIndex}
          className="text-[13px] font-semibold text-slate-800 tracking-tight"
        >
          <CharacterAnimator text={MEDICAL_VERBS[verbIndex]} />
        </h3>
        <p className="text-[10px] font-medium text-slate-400 tracking-wide mt-0.5">
          synthesizing analysis
        </p>
      </div>

      <style dangerouslySetInnerHTML={{ __html: `
        @keyframes char-fade-in {
          0% { opacity: 0; transform: translateY(6px); filter: blur(2px); }
          100% { opacity: 1; transform: translateY(0); filter: blur(0); }
        }
        .animate-char-fade-in {
          animation: char-fade-in 0.5s cubic-bezier(0.2, 0.8, 0.2, 1);
        }
        @keyframes ekg-sweep {
          0%   { stroke-dashoffset: 22; }
          100% { stroke-dashoffset: -100; }
        }
        .ekg-sweep {
          stroke-dasharray: 22 100;
          stroke-dashoffset: 22;
          filter: drop-shadow(0 0 2.5px rgba(99, 102, 241, 0.8));
          animation: ekg-sweep 2.4s cubic-bezier(0.45, 0, 0.55, 1) infinite;
        }
      `}} />
    </div>
  );
};
