import React from "react";
import { AlertOctagon, ChevronRight, ChevronLeft } from "lucide-react";
import { cn } from "../utils/cn";

interface ConsentProps {
    agreed: boolean;
    setAgreed: (v: boolean) => void;
    onContinue: () => void;
    onBack: () => void;
}

export function Consent({
    agreed,
    setAgreed,
    onContinue,
    onBack,
}: ConsentProps) {
    return (
        <section className="mt-8">
            <div className="p-6 rounded-3xl bg-white shadow-sm border border-slate-200">
                <h2 className="text-2xl font-semibold tracking-tight">Consent and data handling</h2>

                <div className="mt-4 rounded-xl bg-amber-50 border border-amber-200 p-4 text-sm text-amber-900">
                    <div className="flex items-center gap-2 font-semibold mb-2">
                        <AlertOctagon className="h-4 w-4" />
                        Important privacy notice
                    </div>
                    <p className="mb-2">
                        <strong>Please do not upload any image containing sensitive or patient-identifiable information.</strong>
                    </p>
                    <p>
                        PulmoLens is a demonstration tool. Uploaded images are processed to generate output and help us understand how the prototype is used.
                        They are <strong>not</strong> used to train the model, but they are stored for analysis.
                        By continuing, you accept the risks of uploading data to a public demo environment.
                    </p>
                </div>

                <p className="text-slate-600 mt-4 text-sm">
                    Please confirm you understand the terms above before continuing.
                </p>
                <label className="mt-4 flex items-start gap-3 text-sm text-slate-700 cursor-pointer p-3 rounded-xl hover:bg-slate-50 border border-transparent hover:border-slate-200 transition-colors">
                    <input type="checkbox" className="mt-1 h-4 w-4" checked={agreed} onChange={(e) => setAgreed(e.target.checked)} />
                    <span>I have read the privacy notice, will only upload de-identified test images, and accept the risks.</span>
                </label>
                <div className="mt-6 flex items-center gap-3">
                    <button
                        disabled={!agreed}
                        onClick={onContinue}
                        className={cn(
                            "inline-flex items-center gap-2 rounded-2xl px-5 py-3 shadow",
                            agreed ? "bg-slate-900 text-white hover:shadow-md" : "bg-slate-200 text-slate-500 cursor-not-allowed"
                        )}
                    >
                        Continue to upload <ChevronRight className="h-4 w-4" />
                    </button>
                    <button onClick={onBack} className="inline-flex items-center gap-2 px-4 py-2 rounded-xl border border-slate-300 bg-white">
                        <ChevronLeft className="h-4 w-4" /> Back
                    </button>
                </div>
            </div>
        </section>
    );
}
