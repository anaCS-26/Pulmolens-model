import React, { useState, useEffect } from 'react';
import { Activity } from 'lucide-react';

const MEDICAL_VERBS = [
  "Auscultating...",
  "Palpating...",
  "Stabilizing...",
  "Triaging...",
  "Diagnosing...",
  "Correlating...",
  "Synthesizing...",
  "Examining...",
  "Scanning...",
  "Probing...",
  "Consulting...",
  "Analyzing...",
  "Localizing...",
  "Differentiating..."
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
    <div className="flex items-center gap-5 py-4 px-6 bg-white/80 backdrop-blur-2xl rounded-2xl border border-slate-200 shadow-xl max-w-[320px] mx-auto animate-in fade-in zoom-in-95 duration-500 ring-1 ring-slate-900/5">
      <div className="relative shrink-0 w-10 h-10 flex items-center justify-center">
        {/* The "Inactive" Muted Base Line */}
        <Activity className="absolute h-8 w-8 text-slate-200/60" strokeWidth={2.5} />
        
        {/* The "Active" Purple Scanning Pulse Line */}
        <div className="absolute inset-0 flex items-center justify-center overflow-hidden">
          <div 
            className="absolute h-8 w-8 text-indigo-600 animate-ekg-sweep"
            style={{ 
                maskImage: 'linear-gradient(to right, transparent 0%, black 50%, transparent 100%)',
                WebkitMaskImage: 'linear-gradient(to right, transparent 0%, black 50%, transparent 100%)',
                maskSize: '200% 100%',
                WebkitMaskSize: '200% 100%',
            }}
          >
            <Activity className="h-8 w-8" strokeWidth={3} />
          </div>
        </div>
        
        {/* Subtle glow dot that follows the sweep (optional for polish) */}
        <div className="absolute inset-0 overflow-hidden pointer-events-none">
            <div className="h-full w-2 bg-indigo-400/40 blur-md animate-ekg-glow" />
        </div>
      </div>
      
      <div className="flex flex-col min-w-0 py-1">
        <h3 className="text-xl font-bold text-slate-900 leading-none h-7 flex items-center">
          <CharacterAnimator text={MEDICAL_VERBS[verbIndex]} />
        </h3>
        <p className="text-[10px] font-extrabold text-slate-400 uppercase tracking-widest mt-1.5 opacity-80">
          Synthesizing Analysis
        </p>
      </div>

      <style dangerouslySetInnerHTML={{ __html: `
        @keyframes char-fade-in {
          0% { opacity: 0; transform: translateY(12px); filter: blur(2px); }
          100% { opacity: 1; transform: translateY(0); filter: blur(0); }
        }
        @keyframes ekg-sweep {
          0% { transform: translateX(-100%); }
          100% { transform: translateX(100%); }
        }
        @keyframes ekg-glow {
            0% { transform: translateX(-200%); }
            100% { transform: translateX(400%); }
        }
        .animate-char-fade-in {
          animation: char-fade-in 0.7s cubic-bezier(0.16, 1, 0.3, 1);
        }
        .animate-ekg-sweep {
          animation: ekg-sweep 2s infinite linear;
        }
        .animate-ekg-glow {
            animation: ekg-glow 2s infinite linear;
        }
      `}} />
    </div>
  );
};
