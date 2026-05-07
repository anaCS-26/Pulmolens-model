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
  "Radiating",
  "Grounding",
  "Consulting",
  "Developing",
  "Fixating",
  "Analyzing",
  "Localizing",
  "Differentiating"
];

const SPINNER_FRAMES = ['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏'];

export const ThinkingLoader: React.FC = () => {
  const [verbIndex, setVerbIndex] = useState(0);
  const [spinnerIndex, setSpinnerIndex] = useState(0);

  useEffect(() => {
    // Cycle the spinner very fast (mimic terminal refresh)
    const spinnerInterval = setInterval(() => {
      setSpinnerIndex((prev) => (prev + 1) % SPINNER_FRAMES.length);
    }, 80);

    // Cycle verbs every 1.5 seconds
    const verbInterval = setInterval(() => {
      setVerbIndex((prev) => (prev + 1) % MEDICAL_VERBS.length);
    }, 1500);

    return () => {
      clearInterval(spinnerInterval);
      clearInterval(verbInterval);
    };
  }, []);

  return (
    <div className="flex flex-col items-center justify-center py-12 px-6 bg-white/5 backdrop-blur-md rounded-2xl border border-white/10 shadow-xl animate-in fade-in duration-700">
      <div className="flex items-center space-x-4 mb-4">
        <span className="text-4xl font-mono text-cyan-400 min-w-[1.5rem] text-center">
          {SPINNER_FRAMES[spinnerIndex]}
        </span>
        <h3 className="text-2xl font-semibold bg-gradient-to-r from-white via-cyan-100 to-white/70 bg-clip-text text-transparent">
          {MEDICAL_VERBS[verbIndex]}...
        </h3>
      </div>
      
      <p className="text-cyan-200/60 font-mono text-sm tracking-widest uppercase">
        Gemma 4 Reasoning in Progress
      </p>

      {/* Decorative pulse line */}
      <div className="mt-8 w-48 h-1 bg-white/5 rounded-full overflow-hidden relative">
        <div className="absolute inset-0 bg-gradient-to-r from-transparent via-cyan-400 to-transparent w-24 animate-shimmer" />
      </div>

      <style dangerouslySetInnerHTML={{ __html: `
        @keyframes shimmer {
          0% { transform: translateX(-100%); }
          100% { transform: translateX(200%); }
        }
        .animate-shimmer {
          animation: shimmer 2s infinite ease-in-out;
        }
      `}} />
    </div>
  );
};
