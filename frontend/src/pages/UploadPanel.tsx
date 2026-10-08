import React, { useCallback, useEffect, useRef, useState } from "react";
import { AnimatePresence, motion } from "framer-motion";
import { cn } from "../utils/cn";
import { Button } from "../components/ui/Button";
import { EASE_OUT } from "../components/ui/motion";
import { FlowProgress } from "../components/layout/FlowProgress";
import { FilmPanel } from "../components/viewer/FilmPanel";

interface UploadPanelProps {
    agreed: boolean;
    onAgree: (v: boolean) => void;
    onFile: (f: File) => void;
}

// Mirrors the backend's default MAX_UPLOAD_BYTES so oversize files fail fast, locally.
const MAX_BYTES = 10 * 1024 * 1024;
const ACCEPTED = ["image/png", "image/jpeg", "image/jpg"];

function validate(f: File): string | null {
    if (!ACCEPTED.includes(f.type)) return `“${f.name}” isn't a PNG or JPG.`;
    if (f.size > MAX_BYTES) return `“${f.name}” is ${(f.size / 1024 / 1024).toFixed(1)} MB. The limit is 10 MB.`;
    return null;
}

export function UploadPanel({ agreed, onAgree, onFile }: UploadPanelProps) {
    const [error, setError] = useState<string | null>(null);
    const [windowDrag, setWindowDrag] = useState(false);

    const submit = useCallback(
        (f: File | undefined | null) => {
            if (!f) return;
            const err = validate(f);
            if (err) { setError(err); return; }
            setError(null);
            onFile(f);
        },
        [onFile]
    );

    const loadSample = async (i: number) => {
        try {
            const res = await fetch(`/example${i}.png`);
            const blob = await res.blob();
            onFile(new File([blob], `example_cxr_${i}.png`, { type: "image/png" }));
        } catch (e) {
            console.error("Failed to load example", e);
            setError("Couldn't load the sample image.");
        }
    };

    // Window-wide drag-and-drop and clipboard paste, once consent is given.
    useEffect(() => {
        if (!agreed) return;
        let depth = 0;
        const hasFiles = (e: DragEvent) => Array.from(e.dataTransfer?.types || []).includes("Files");
        const onEnter = (e: DragEvent) => { if (!hasFiles(e)) return; depth++; setWindowDrag(true); };
        const onLeave = () => { depth = Math.max(0, depth - 1); if (depth === 0) setWindowDrag(false); };
        const onOver = (e: DragEvent) => { if (hasFiles(e)) e.preventDefault(); };
        const onDrop = (e: DragEvent) => {
            if (!hasFiles(e)) return;
            e.preventDefault();
            depth = 0;
            setWindowDrag(false);
            submit(e.dataTransfer?.files?.[0]);
        };
        const onPaste = (e: ClipboardEvent) => {
            const item = Array.from(e.clipboardData?.items || []).find((it) => it.kind === "file" && it.type.startsWith("image/"));
            const f = item?.getAsFile();
            if (f) submit(new File([f], f.name || "pasted_image.png", { type: f.type }));
        };
        window.addEventListener("dragenter", onEnter);
        window.addEventListener("dragleave", onLeave);
        window.addEventListener("dragover", onOver);
        window.addEventListener("drop", onDrop);
        window.addEventListener("paste", onPaste);
        return () => {
            window.removeEventListener("dragenter", onEnter);
            window.removeEventListener("dragleave", onLeave);
            window.removeEventListener("dragover", onOver);
            window.removeEventListener("drop", onDrop);
            window.removeEventListener("paste", onPaste);
        };
    }, [agreed, submit]);

    return (
        <section className="mx-auto max-w-[1200px] px-6 pb-24 pt-12">
            <FlowProgress current={agreed ? 1 : 0} />
            <h1 className="mt-6 font-serif text-[44px] font-normal leading-none tracking-[-0.02em] md:text-[56px]">New analysis</h1>

            <div className="mt-12 grid grid-cols-1 gap-x-12 gap-y-14 lg:grid-cols-12">
                <div className="lg:col-span-8">
                    {/* Step 1 */}
                    <div className="border-t border-ink pt-5">
                        <div className="flex items-baseline justify-between gap-4">
                            <h2 className="flex items-baseline gap-3 text-[17px] font-medium">
                                <span className="font-mono text-[12px] text-ink-faint">1</span> Consent and data handling
                            </h2>
                            {agreed && (
                                <button onClick={() => onAgree(false)} className="link text-[13px] text-ink-muted">Review</button>
                            )}
                        </div>
                        <AnimatePresence initial={false} mode="wait">
                            {agreed ? (
                                <motion.p key="done" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }} className="mt-2 pl-[22px] text-[14px] text-ink-muted">
                                    ✓ Accepted for this browser session.
                                </motion.p>
                            ) : (
                                <ConsentForm key="form" onAgree={() => onAgree(true)} />
                            )}
                        </AnimatePresence>
                    </div>

                    {/* Step 2 */}
                    <div className={cn("mt-12 border-t pt-5 transition-colors", agreed ? "border-ink" : "border-rule")}>
                        <h2 className={cn("flex items-baseline gap-3 text-[17px] font-medium", !agreed && "text-ink-faint")}>
                            <span className="font-mono text-[12px] text-ink-faint">2</span> Image
                        </h2>
                        {agreed ? (
                            <motion.div initial={{ opacity: 0, y: 6 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.4, ease: EASE_OUT }}>
                                <Dropzone onFile={submit} />
                                {error && (
                                    <p role="alert" className="mt-3 border-l-2 border-marker pl-3 text-[14px] text-marker-dark">{error}</p>
                                )}
                            </motion.div>
                        ) : (
                            <p className="mt-2 pl-[22px] text-[14px] text-ink-faint">Available once you've accepted the notice above, or use a sample film.</p>
                        )}
                    </div>
                </div>

                <aside className="lg:col-span-4">
                    <div className="border-t border-ink pt-5">
                        <h2 className="text-[17px] font-medium">Sample films</h2>
                        <p className="mt-1 text-[14px] text-ink-muted">No image to hand? Run the full pipeline on a bundled film.</p>
                        <div className="mt-5 grid grid-cols-2 gap-3">
                            {[1, 2].map((i) => (
                                <button key={i} onClick={() => loadSample(i)} className="group text-left" aria-label={`Analyse sample ${i}`}>
                                    <FilmPanel className="aspect-square" tl={`Sample 0${i}`}>
                                        <img src={`/example${i}.png`} alt="" className="h-full w-full object-cover opacity-90 transition-opacity group-hover:opacity-100" />
                                    </FilmPanel>
                                    <span className="mt-2 inline-block text-[13px] text-ink-muted transition-colors group-hover:text-ink">
                                        Analyse sample {i} →
                                    </span>
                                </button>
                            ))}
                        </div>
                    </div>

                    <div className="mt-10 border-t border-rule pt-5">
                        <h2 className="text-[15px] font-medium">What happens next</h2>
                        <ol className="mt-3 space-y-2.5 text-[14px] leading-relaxed text-ink-muted">
                            {[
                                "Your image is sent to the backend.",
                                "The server runs the model and returns pathology probabilities and an attention overlay.",
                                "Only live model output is displayed. If the server is unavailable you'll see an error, never mock results.",
                            ].map((t, i) => (
                                <li key={i} className="grid grid-cols-[1.25rem_1fr]">
                                    <span className="font-mono text-[12px] leading-[1.9] text-ink-faint">{i + 1}</span>
                                    <span>{t}</span>
                                </li>
                            ))}
                        </ol>
                    </div>
                </aside>
            </div>

            {/* full-window drop target */}
            <AnimatePresence>
                {windowDrag && (
                    <motion.div
                        initial={{ opacity: 0 }}
                        animate={{ opacity: 1 }}
                        exit={{ opacity: 0 }}
                        transition={{ duration: 0.15 }}
                        className="pointer-events-none fixed inset-0 z-50 bg-paper/90 p-6"
                    >
                        <div className="grid h-full w-full place-items-center border-2 border-dashed border-ink">
                            <div className="text-center">
                                <div className="font-serif text-5xl">Drop to analyse</div>
                                <div className="mt-3 font-mono text-[12px] text-ink-muted">PNG or JPG · up to 10 MB</div>
                            </div>
                        </div>
                    </motion.div>
                )}
            </AnimatePresence>
        </section>
    );
}

function ConsentForm({ onAgree }: { onAgree: () => void }) {
    const [checked, setChecked] = useState(false);
    return (
        <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }} transition={{ duration: 0.2 }} className="mt-5 pl-[22px]">
            <div className="border-l-2 border-marker bg-paper-raised px-5 py-4">
                <p className="font-medium">Please do not upload any image containing sensitive or patient-identifiable information.</p>
                <p className="mt-2 text-[14px] leading-relaxed text-ink-soft">
                    PulmoLens is a demonstration tool. Uploaded images are processed to generate output and help us understand how
                    the prototype is used. They are <strong className="font-semibold">not</strong> used to train the model, but they
                    are stored for analysis. By continuing, you accept the risks of uploading data to a public demo environment.
                </p>
            </div>

            <label className="mt-5 flex cursor-pointer items-start gap-3 text-[14.5px]">
                <input
                    type="checkbox"
                    checked={checked}
                    onChange={(e) => setChecked(e.target.checked)}
                    className="mt-[3px] h-4 w-4 shrink-0 cursor-pointer accent-ink"
                />
                <span>I have read the privacy notice, will only upload de-identified test images, and accept the risks.</span>
            </label>

            <Button variant="primary" className="mt-5" disabled={!checked} onClick={onAgree}>
                Continue
            </Button>
        </motion.div>
    );
}

function Dropzone({ onFile }: { onFile: (f: File | undefined) => void }) {
    const inputRef = useRef<HTMLInputElement | null>(null);
    const [hover, setHover] = useState(false);

    return (
        <div
            onDragOver={(e) => { e.preventDefault(); setHover(true); }}
            onDragLeave={() => setHover(false)}
            onDrop={() => setHover(false) /* the window-level listener submits the file */}
            onClick={() => inputRef.current?.click()}
            onKeyDown={(e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); inputRef.current?.click(); } }}
            role="button"
            tabIndex={0}
            aria-label="Upload a chest X-ray"
            className={cn(
                "mt-5 ml-[22px] flex h-64 cursor-pointer flex-col items-center justify-center border border-dashed text-center transition-colors",
                hover ? "border-ink bg-paper-raised" : "border-ink/35 hover:border-ink hover:bg-paper-raised"
            )}
        >
            <div className="font-serif text-[28px] leading-tight">Drop a radiograph here</div>
            <div className="mt-2 text-[14px] text-ink-muted">
                or <span className="text-ink underline decoration-ink/30 underline-offset-[3px]">choose a file</span>, or paste one with{" "}
                <kbd className="font-mono text-[12px]">Ctrl V</kbd>
            </div>
            <div className="mt-5 font-mono text-[11.5px] text-ink-faint">PNG or JPG · up to 10 MB</div>
            <input
                ref={inputRef}
                type="file"
                accept=".jpg,.jpeg,.png"
                className="hidden"
                onClick={(e) => e.stopPropagation()}
                onChange={(e) => { onFile(e.target.files?.[0]); e.target.value = ""; }}
            />
        </div>
    );
}
