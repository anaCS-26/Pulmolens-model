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

export const ThinkingLoader: React.FC = () => {
  const [verbIndex, setVerbIndex] = useState(0);

  useEffect(() => {
    // Cycle verbs every 2 seconds for a calmer pace
    const verbInterval = setInterval(() => {
      setVerbIndex((prev) => (prev + 1) % MEDICAL_VERBS.length);
    }, 2000);

    return () => clearInterval(verbInterval);
  }, []);

  return (
    <div className="flex items-center gap-4 py-6 px-6 bg-slate-50 rounded-2xl border border-slate-200 shadow-sm animate-in fade-in duration-500">
      <div className="relative flex items-center justify-center">
        <Activity className="h-6 w-6 text-indigo-600 animate-pulse-fast" />
        {/* Subtle outer glow that pulses */}
        <div className="absolute inset-0 bg-indigo-400/20 rounded-full blur-md animate-ping-slow" />
      </div>
      
      <div className="flex flex-col">
        <h3 className="text-base font-medium text-slate-900">
          {MEDICAL_VERBS[verbIndex]}...
        </h3>
        <p className="text-xs text-slate-500 font-medium">
          Generating clinical summary
        </p>
      </div>

      <style dangerouslySetInnerHTML={{ __html: `
        @keyframes pulse-fast {
          0%, 100% { transform: scale(1); opacity: 1; }
          50% { transform: scale(1.15); opacity: 0.8; }
        }
        @keyframes ping-slow {
          0% { transform: scale(1); opacity: 0.4; }
          100% { transform: scale(1.6); opacity: 0; }
        }
        .animate-pulse-fast {
          animation: pulse-fast 0.8s infinite ease-in-out;
        }
        .animate-ping-slow {
          animation: ping-slow 2s infinite cubic-bezier(0, 0, 0.2, 1);
        }
      `}} />
    </div>
  );
};
