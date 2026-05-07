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
    }, 3000); // Slightly longer for the fade-in to be appreciated

    return () => clearInterval(verbInterval);
  }, []);

  return (
    <div className="flex items-center gap-6 py-5 px-8 bg-white/60 backdrop-blur-xl rounded-2xl border border-slate-200 shadow-lg animate-in fade-in duration-700 max-w-sm mx-auto ring-1 ring-slate-900/5">
      <div className="relative flex items-center justify-center shrink-0">
        {/* Heartbeat Shimmer Glow */}
        <div className="relative h-8 w-8">
           {/* Static Base Icon */}
           <Activity className="absolute inset-0 h-8 w-8 text-slate-200" />
           
           {/* Glowing Animated Icon Overlay */}
           <div className="absolute inset-0 h-8 w-8 overflow-hidden pointer-events-none">
              <div className="h-full w-[200%] flex animate-glow-sweep">
                 <div className="w-1/2 flex items-center justify-center">
                    <Activity className="h-8 w-8 text-indigo-500 filter drop-shadow-[0_0_8px_rgba(99,102,241,0.6)]" />
                 </div>
                 <div className="w-1/2" />
              </div>
           </div>
        </div>
        
        {/* Subtle background pulse */}
        <div className="absolute inset-0 bg-indigo-500/10 rounded-full blur-2xl animate-pulse" />
      </div>
      
      <div className="flex flex-col min-w-0">
        <h3 className="text-lg font-bold text-slate-900 leading-tight">
          <CharacterAnimator text={MEDICAL_VERBS[verbIndex]} />
          <span className="text-indigo-500 inline-block animate-pulse ml-0.5">...</span>
        </h3>
        <p className="text-[10px] font-extrabold text-slate-400 uppercase tracking-widest mt-1">
          Synthesizing Report
        </p>
      </div>

      <style dangerouslySetInnerHTML={{ __html: `
        @keyframes char-fade-in {
          0% { 
            opacity: 0; 
            transform: translateY(16px); 
            filter: blur(4px);
          }
          100% { 
            opacity: 1; 
            transform: translateY(0); 
            filter: blur(0);
          }
        }
        @keyframes glow-sweep {
          0% { transform: translateX(-100%); }
          100% { transform: translateX(50%); }
        }
        .animate-char-fade-in {
          animation: char-fade-in 0.8s cubic-bezier(0.16, 1, 0.3, 1);
        }
        .animate-glow-sweep {
          animation: glow-sweep 2.5s infinite linear;
        }
      `}} />
    </div>
  );
};
