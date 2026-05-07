import React, { useState, useEffect } from 'react';
import { Activity } from 'lucide-react';

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
  "Differentiating"
];

const CharacterAnimator: React.FC<{ text: string }> = ({ text }) => {
  return (
    <span className="inline-flex overflow-hidden">
      {text.split('').map((char, index) => (
        <span
          key={`${text}-${index}`}
          className="inline-block animate-char-fade-in"
          style={{ 
            animationDelay: `${index * 30}ms`,
            animationFillMode: 'both'
          }}
        >
          {char === ' ' ? '\u00A0' : char}
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
    <div className="flex items-center gap-4 py-3 px-5 bg-white/70 backdrop-blur-xl rounded-xl border border-slate-200 shadow-md max-w-[280px] mx-auto animate-in fade-in duration-500">
      <div className="relative flex items-center justify-center shrink-0 w-8 h-8">
        {/* Base muted icon */}
        <Activity className="h-6 w-6 text-slate-200" />
        
        {/* Animated scanning line overlay */}
        <div className="absolute inset-0 flex items-center justify-center overflow-hidden pointer-events-none">
          <div className="relative h-6 w-6">
            <Activity className="h-6 w-6 text-indigo-600" />
            {/* The scanning "glow line" */}
            <div className="absolute inset-0 bg-gradient-to-r from-transparent via-white/80 to-transparent w-full animate-scan-line skew-x-12 mix-blend-overlay" />
          </div>
        </div>
      </div>
      
      <div className="flex flex-col min-w-0">
        <h3 className="text-sm font-bold text-slate-900 leading-none flex items-center">
          <CharacterAnimator text={MEDICAL_VERBS[verbIndex]} />
          <span className="text-indigo-500 ml-0.5 animate-pulse">...</span>
        </h3>
        <p className="text-[9px] font-bold text-slate-400 uppercase tracking-tighter mt-1">
          Synthesizing Analysis
        </p>
      </div>

      <style dangerouslySetInnerHTML={{ __html: `
        @keyframes char-fade-in {
          0% { opacity: 0; transform: translateY(10px); filter: blur(2px); }
          100% { opacity: 1; transform: translateY(0); filter: blur(0); }
        }
        @keyframes scan-line {
          0% { transform: translateX(-150%); }
          100% { transform: translateX(150%); }
        }
        .animate-char-fade-in {
          animation: char-fade-in 0.6s cubic-bezier(0.2, 0.8, 0.2, 1);
        }
        .animate-scan-line {
          animation: scan-line 2s infinite ease-in-out;
        }
      `}} />
    </div>
  );
};
