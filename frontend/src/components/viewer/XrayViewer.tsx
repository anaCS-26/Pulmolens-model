import React from "react";
import { Tabs } from "../ui/Tabs";
import { CompareSlider } from "./CompareSlider";
import { FilmPanel } from "./FilmPanel";
import { prettyLabel } from "../../utils/format";

export type ViewMode = "overlay" | "compare" | "original";

interface XrayViewerProps {
    imageURL: string | null;
    overlay: string | null;
    overlayFinding?: string;
    fileName: string;
    mode: ViewMode;
    setMode: (m: ViewMode) => void;
    opacity: number;
    setOpacity: (v: number) => void;
    onExpand: () => void;
}

export function OpacitySlider({ value, onChange, className, tone = "paper" }: { value: number; onChange: (v: number) => void; className?: string; tone?: "paper" | "film" }) {
    return (
        <label className={`flex items-center gap-3 ${tone === "film" ? "text-white" : "text-ink"} ${className || ""}`}>
            <span className={`font-mono text-[11px] uppercase tracking-[0.06em] ${tone === "film" ? "text-white/60" : "text-ink-muted"}`}>Opacity</span>
            <input
                type="range"
                min={0}
                max={1}
                step={0.05}
                value={value}
                onChange={(e) => onChange(parseFloat(e.target.value))}
                className="range w-full min-w-[90px]"
                style={{ ["--fill" as string]: `${value * 100}%` }}
                aria-label="Overlay opacity"
            />
            <span className={`num w-9 text-right font-mono text-[12px] ${tone === "film" ? "text-white/70" : "text-ink-muted"}`}>{Math.round(value * 100)}%</span>
        </label>
    );
}

/**
 * The attention overlay is a composite (X-ray with heat baked in), so
 * "opacity" crossfades composite over original rather than tinting a mask.
 */
export function XrayViewer({ imageURL, overlay, overlayFinding, fileName, mode, setMode, opacity, setOpacity, onExpand }: XrayViewerProps) {
    const effective: ViewMode = overlay ? mode : "original";
    const img = (src: string, alt: string, style?: React.CSSProperties) => (
        <img src={src} alt={alt} draggable={false} className="absolute inset-0 h-full w-full object-contain transition-opacity duration-200" style={style} />
    );
    const finding = overlayFinding ? prettyLabel(overlayFinding) : "top finding";

    return (
        <figure>
            <FilmPanel
                className="aspect-[4/3] w-full"
                tl={fileName}
                tr={overlay && effective !== "original" ? `Attention · ${finding}` : undefined}
                bl={effective === "compare" ? undefined : effective === "overlay" ? `Overlay ${Math.round(opacity * 100)}%` : "Original"}
            >
                {!imageURL ? null : effective === "compare" && overlay ? (
                    <CompareSlider
                        className="h-full w-full"
                        before={img(imageURL, "Original chest X-ray")}
                        after={img(overlay, "Attention overlay")}
                    />
                ) : (
                    <button onDoubleClick={onExpand} className="absolute inset-0 cursor-zoom-in" aria-label="Open fullscreen viewer">
                        {img(imageURL, "Uploaded chest X-ray")}
                        {overlay && effective === "overlay" && img(overlay, "Attention overlay", { opacity })}
                    </button>
                )}
            </FilmPanel>

            <div className="flex flex-wrap items-center justify-between gap-x-6 gap-y-3 border-b border-rule py-3">
                <Tabs<ViewMode>
                    value={effective}
                    onChange={setMode}
                    options={[
                        { value: "overlay", label: "Overlay", disabled: !overlay },
                        { value: "compare", label: "Compare", disabled: !overlay },
                        { value: "original", label: "Original" },
                    ]}
                />
                <div className="flex flex-1 items-center justify-end gap-6">
                    {overlay && effective === "overlay" && <OpacitySlider value={opacity} onChange={setOpacity} className="max-w-[220px] flex-1" />}
                    <button onClick={onExpand} className="link text-[13px]" title="Fullscreen (F)">Fullscreen</button>
                </div>
            </div>

            <figcaption className="mt-3 text-[13px] leading-snug text-ink-muted">
                {overlay ? (
                    <>
                        <span className="font-medium text-ink">Attention map for {finding.toLowerCase()}.</span> Warm regions moved the
                        score most. It is not a lesion outline.
                    </>
                ) : (
                    "No attention overlay was returned for this image, so only the original is shown."
                )}
            </figcaption>
        </figure>
    );
}
