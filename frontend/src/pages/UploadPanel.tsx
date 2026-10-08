import React, { useCallback, useEffect, useRef, useState } from "react";
import { AnimatePresence, motion } from "framer-motion";
import { ArrowRight, Check, Upload } from "lucide-react";
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
        <section className="mx-auto max-w-[1280px] px-4 pb-24 pt-10 sm:px-6 lg:px-10">
            <FlowProgress current={agreed ? 1 : 0} />
            <h1 className="display mt-7 text-[40px] leading-none md:text-[54px]">New analysis</h1>

            <div className="mt-10 grid grid-cols-1 gap-x-10 gap-y-10 lg:grid-cols-12">
                <div className="space-y-5 lg:col-span-8">
                    {/* Step 1 */}
                    <div className="rounded-xl border border-line bg-surface-raised p-5 sm:p-6">
                        <div className="flex items-center justify-between gap-4">
                            <StepTitle n={1} done={agreed}>Consent and data handling</StepTitle>
                            {agreed && (
                                <button onClick={() => onAgree(false)} className="link text-[13px] text-fg-muted">Review</button>
                            )}
                        </div>
                        <AnimatePresence initial={false} mode="wait">
                            {agreed ? (
                                <motion.p key="done" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }} className="mt-2 pl-9 text-[14px] text-fg-muted">
                                    Accepted for this browser session.
                                </motion.p>
                            ) : (
                                <ConsentForm key="form" onAgree={() => onAgree(true)} />
                            )}
                        </AnimatePresence>
                    </div>

                    {/* Step 2 */}
                    <div className={cn("rounded-xl border p-5 transition-colors sm:p-6", agreed ? "border-line bg-surface-raised" : "border-line-soft bg-transparent")}>
                        <StepTitle n={2} muted={!agreed}>Image</StepTitle>
                        {agreed ? (
                            <motion.div initial={{ opacity: 0, y: 6 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.4, ease: EASE_OUT }}>
                                <Dropzone onFile={submit} />
                                {error && (
                                    <p role="alert" className="mt-3 rounded-lg border border-urgent/40 bg-urgent/[0.05] px-3.5 py-2.5 text-[14px] text-urgent">{error}</p>
                                )}
                            </motion.div>
                        ) : (
                            <p className="mt-2 pl-9 text-[14px] text-fg-faint">Available once you've accepted the notice above, or use a sample film.</p>
                        )}
                    </div>
                </div>

                <aside className="space-y-8 lg:col-span-4">
                    <div>
                        <h2 className="text-[16px] font-semibold">Sample films</h2>
                        <p className="mt-1 text-[14px] text-fg-muted">No image to hand? Run the full pipeline on a bundled film.</p>
                        <div className="mt-4 grid grid-cols-2 gap-3">
                            {[1, 2].map((i) => (
                                <button key={i} onClick={() => loadSample(i)} className="group text-left" aria-label={`Analyse sample ${i}`}>
                                    <FilmPanel className="aspect-square transition-shadow group-hover:ring-accent" tl={`Sample 0${i}`}>
                                        <img src={`/example${i}.png`} alt="" className="h-full w-full object-cover opacity-85 transition-opacity group-hover:opacity-100" />
                                    </FilmPanel>
                                    <span className="mt-2 flex items-center gap-1.5 text-[13.5px] text-fg-muted transition-colors group-hover:text-fg">
                                        Analyse sample {i} <ArrowRight className="h-3.5 w-3.5 transition-transform group-hover:translate-x-0.5" />
                                    </span>
                                </button>
                            ))}
                        </div>
                    </div>

                    <div className="border-t border-line pt-6">
                        <h2 className="text-[16px] font-semibold">What happens next</h2>
                        <ol className="mt-3 space-y-3 text-[14px] leading-relaxed text-fg-muted">
                            {[
                                "Your image is sent to the model server.",
                                "The server returns a score for each of fourteen findings and an attention map.",
                                "Only live model output is shown. If the server is unavailable you'll see an error, never mock results.",
                            ].map((t, i) => (
                                <li key={i} className="grid grid-cols-[1.5rem_1fr]">
                                    <span className="num text-fg-faint">{i + 1}</span>
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
                        className="pointer-events-none fixed inset-0 z-50 bg-surface/90 p-6 backdrop-blur-sm"
                    >
                        <div className="grid h-full w-full place-items-center rounded-2xl border-2 border-dashed border-accent">
                            <div className="text-center">
                                <div className="display text-[44px]">Drop to analyse</div>
                                <div className="mt-2 text-[14px] text-fg-muted">PNG or JPG · up to 10 MB</div>
                            </div>
                        </div>
                    </motion.div>
                )}
            </AnimatePresence>
        </section>
    );
}

function StepTitle({ n, done, muted, children }: { n: number; done?: boolean; muted?: boolean; children: React.ReactNode }) {
    return (
        <h2 className={cn("flex items-center gap-3 text-[17px] font-semibold", muted && "text-fg-faint")}>
            <span
                className={cn(
                    "grid h-6 w-6 place-items-center rounded-full border text-[12px] num",
                    done ? "border-line bg-surface-sunk text-fg-muted" : muted ? "border-line" : "border-accent bg-accent text-black"
                )}
            >
                {done ? <Check className="h-3.5 w-3.5" strokeWidth={2.5} /> : n}
            </span>
            {children}
        </h2>
    );
}

function ConsentForm({ onAgree }: { onAgree: () => void }) {
    const [checked, setChecked] = useState(false);
    return (
        <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }} transition={{ duration: 0.2 }} className="mt-5 sm:pl-9">
            <div className="rounded-lg border border-accent/40 bg-accent/[0.06] px-5 py-4">
                <p className="font-semibold">Please do not upload any image containing sensitive or patient-identifiable information.</p>
                <p className="mt-2 text-[14px] leading-relaxed text-fg-soft">
                    PulmoLens is a demonstration tool. Uploaded images are processed to generate output and help us understand how
                    the prototype is used. They are <strong className="font-semibold">not</strong> used to train the model, but they
                    are stored for analysis. By continuing, you accept the risks of uploading data to a public demo environment.
                </p>
            </div>

            <label className="mt-5 flex cursor-pointer items-start gap-3 text-[14.5px] text-fg-soft">
                <input
                    type="checkbox"
                    checked={checked}
                    onChange={(e) => setChecked(e.target.checked)}
                    className="mt-[3px] h-4 w-4 shrink-0 cursor-pointer accent-[rgb(var(--accent))]"
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
                "mt-5 flex h-64 cursor-pointer flex-col items-center justify-center rounded-xl border-[1.5px] border-dashed text-center transition-colors sm:ml-9",
                hover ? "border-accent bg-accent/[0.05]" : "border-line hover:border-fg-muted hover:bg-surface-sunk/60"
            )}
        >
            <Upload className="h-6 w-6 text-fg-muted" strokeWidth={1.5} />
            <div className="mt-3 text-[20px] font-semibold [font-stretch:108%]">Drop a radiograph here</div>
            <div className="mt-1.5 text-[14px] text-fg-muted">
                or <span className="text-fg underline decoration-accent underline-offset-[3px]">choose a file</span>, or paste one with{" "}
                <kbd className="rounded border border-line bg-surface-sunk px-1.5 py-0.5 font-sans text-[12px]">Ctrl V</kbd>
            </div>
            <div className="mt-4 text-[12.5px] text-fg-faint">PNG or JPG · up to 10 MB</div>
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
