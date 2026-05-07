import React, { useRef, useState } from "react";
import { Upload, ChevronLeft } from "lucide-react";
import { cn } from "../utils/cn";

interface UploadPanelProps {
    onFile: (f: File) => void;
    onBack: () => void;
}

export function UploadPanel({ onFile, onBack }: UploadPanelProps) {
    const inputRef = useRef<HTMLInputElement | null>(null);
    const [hover, setHover] = useState(false);

    return (
        <section className="mt-8 grid grid-cols-1 lg:grid-cols-3 gap-8">
            <div className="lg:col-span-2 p-6 rounded-3xl bg-white shadow-sm border border-slate-200">
                <h2 className="text-2xl font-semibold tracking-tight">Upload a chest X-ray</h2>
                <p className="text-slate-600 mt-2 text-sm">Accepted formats: JPG, JPEG, PNG.</p>

                <div
                    onDragOver={(e) => { e.preventDefault(); setHover(true); }}
                    onDragLeave={() => setHover(false)}
                    onDrop={(e) => { e.preventDefault(); setHover(false); const f = e.dataTransfer.files?.[0]; if (f) onFile(f); }}
                    className={cn(
                        "mt-6 h-56 rounded-2xl border-2 border-dashed flex items-center justify-center",
                        hover ? "border-slate-900 bg-slate-50" : "border-slate-300"
                    )}
                >
                    <div className="text-center">
                        <Upload className="mx-auto h-10 w-10 text-slate-400" />
                        <div className="mt-2 font-medium">Drag and drop your image</div>
                        <div className="text-sm text-slate-500">or</div>
                        <button onClick={() => inputRef.current?.click()} className="mt-2 inline-flex items-center gap-2 rounded-xl bg-slate-900 text-white px-4 py-2">
                            Browse files
                        </button>
                        <input ref={inputRef} type="file" accept=".jpg,.jpeg,.png" className="hidden" onChange={(e) => { const f = e.target.files?.[0]; if (f) onFile(f); }} />
                    </div>
                </div>

                <div className="mt-6">
                    <button onClick={onBack} className="inline-flex items-center gap-2 px-4 py-2 rounded-xl border border-slate-300 bg-white">
                        <ChevronLeft className="h-4 w-4" /> Back
                    </button>
                </div>
            </div>

            <div className="space-y-6">
                <div className="p-6 rounded-3xl bg-white shadow-sm border border-slate-200">
                    <h3 className="font-semibold">Try an example</h3>
                    <p className="text-sm text-slate-600 mt-1">New here? Click an image below to run the model instantly.</p>
                    <div className="mt-4 grid grid-cols-2 gap-3">
                        {[1, 2].map((i) => (
                            <button
                                key={i}
                                onClick={async () => {
                                    try {
                                        const res = await fetch(`/example${i}.png`);
                                        const blob = await res.blob();
                                        const file = new File([blob], `example_cxr_${i}.png`, { type: "image/png" });
                                        onFile(file);
                                    } catch (e) {
                                        console.error("Failed to load example", e);
                                    }
                                }}
                                className="group relative aspect-square rounded-xl overflow-hidden border border-slate-200 hover:border-slate-900 transition-all"
                            >
                                <img src={`/example${i}.png`} alt={`Example ${i}`} className="absolute inset-0 h-full w-full object-cover group-hover:scale-105 transition-transform" />
                                <div className="absolute inset-0 bg-black/0 group-hover:bg-black/10 transition-colors" />
                            </button>
                        ))}
                    </div>
                </div>

                <div className="p-6 rounded-3xl bg-white shadow-sm border border-slate-200">
                    <h3 className="font-semibold">What happens next?</h3>
                    <ol className="mt-2 list-decimal list-inside text-sm text-slate-600 space-y-1">
                        <li>Your image is sent to the backend.</li>
                        <li>The server runs the model and returns pathology probabilities.</li>
                        <li>Only the model output is displayed. If the server is unavailable, you will see an error message.</li>
                    </ol>
                </div>
            </div>
        </section>
    );
}
