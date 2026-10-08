import React, { useRef, useState } from "react";
import { Maximize2 } from "lucide-react";
import { Tabs } from "../ui/Tabs";
import { CompareSlider } from "./CompareSlider";
import { FilmPanel, JetLegend } from "./FilmPanel";
import { prettyLabel } from "../../utils/format";
import { cn } from "../../utils/cn";

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
    /** "film" draws the toolbar and caption for a black stage. */
    tone?: "surface" | "film";
    className?: string;
}

export function OpacitySlider({ value, onChange, className, tone = "surface" }: { value: number; onChange: (v: number) => void; className?: string; tone?: "surface" | "film" }) {
    const film = tone === "film";
    return (
        <label className={cn("flex items-center gap-3 text-[13px]", film ? "text-white/60" : "text-fg-muted", className)}>
            <span>Overlay</span>
            <input
                type="range"
                min={0}
                max={1}
                step={0.05}
                value={value}
                onChange={(e) => onChange(parseFloat(e.target.value))}
                className={cn("range w-full min-w-[90px]", film ? "text-white" : "text-accent")}
                style={{ ["--fill" as string]: `${value * 100}%` }}
                aria-label="Overlay opacity"
            />
            <span className={cn("num w-9 text-right", film ? "text-white/80" : "text-fg-soft")}>{Math.round(value * 100)}%</span>
        </label>
    );
}

/** Maps a pointer position to a pixel on an object-contain image, or null if it is outside the image. */
function pixelAt(img: HTMLImageElement | null, clientX: number, clientY: number) {
    if (!img || !img.naturalWidth) return null;
    const r = img.getBoundingClientRect();
    const scale = Math.min(r.width / img.naturalWidth, r.height / img.naturalHeight);
    const w = img.naturalWidth * scale, h = img.naturalHeight * scale;
    const x = (clientX - r.left - (r.width - w) / 2) / scale;
    const y = (clientY - r.top - (r.height - h) / 2) / scale;
    if (x < 0 || y < 0 || x >= img.naturalWidth || y >= img.naturalHeight) return null;
    return { x: Math.floor(x), y: Math.floor(y), px: clientX - r.left, py: clientY - r.top };
}

/**
 * The attention overlay is a composite (X-ray with heat baked in), so
 * "opacity" crossfades composite over original rather than tinting a mask.
 */
export function XrayViewer({ imageURL, overlay, overlayFinding, fileName, mode, setMode, opacity, setOpacity, onExpand, tone = "surface", className }: XrayViewerProps) {
    const film = tone === "film";
    const effective: ViewMode = overlay ? mode : "original";
    const imgRef = useRef<HTMLImageElement>(null);
    const [cursor, setCursor] = useState<ReturnType<typeof pixelAt>>(null);
    const finding = overlayFinding ? prettyLabel(overlayFinding) : "top finding";

    const img = (src: string, alt: string, style?: React.CSSProperties, ref?: React.Ref<HTMLImageElement>) => (
        <img ref={ref} src={src} alt={alt} draggable={false} className="absolute inset-0 h-full w-full object-contain transition-opacity duration-200" style={style} />
    );

    return (
        <figure className={className}>
            <FilmPanel
                className={cn("aspect-square w-full", film && "!ring-white/10")}
                tl={fileName}
                tr={overlay && effective !== "original" ? `Attention · ${finding}` : undefined}
                bl={effective === "compare" ? undefined : effective === "overlay" ? `Overlay ${Math.round(opacity * 100)}%` : "Original"}
                br={effective === "compare" ? undefined : cursor ? `X ${cursor.x}  Y ${cursor.y}` : "Not for\ndiagnostic use"}
            >
                {!imageURL ? null : effective === "compare" && overlay ? (
                    <CompareSlider
                        className="h-full w-full"
                        before={img(imageURL, "Original chest X-ray")}
                        after={img(overlay, "Attention overlay")}
                    />
                ) : (
                    <button
                        onDoubleClick={onExpand}
                        onPointerMove={(e) => setCursor(pixelAt(imgRef.current, e.clientX, e.clientY))}
                        onPointerLeave={() => setCursor(null)}
                        className="absolute inset-0 cursor-crosshair"
                        aria-label="Open fullscreen viewer"
                    >
                        {img(imageURL, "Uploaded chest X-ray", undefined, imgRef)}
                        {overlay && effective === "overlay" && img(overlay, "Attention overlay", { opacity })}
                        {cursor && (
                            <span aria-hidden className="pointer-events-none absolute inset-0">
                                <span className="absolute inset-x-0 h-px bg-white/30" style={{ top: cursor.py }} />
                                <span className="absolute inset-y-0 w-px bg-white/30" style={{ left: cursor.px }} />
                            </span>
                        )}
                    </button>
                )}
            </FilmPanel>

            <div className="flex flex-wrap items-center justify-between gap-x-5 gap-y-3 py-3">
                <Tabs<ViewMode>
                    tone={tone}
                    label="View"
                    value={effective}
                    onChange={setMode}
                    options={[
                        { value: "overlay", label: "Attention", disabled: !overlay },
                        { value: "compare", label: "Split", disabled: !overlay },
                        { value: "original", label: "Original" },
                    ]}
                />
                <div className="flex flex-1 items-center justify-end gap-5">
                    {overlay && effective === "overlay" && <OpacitySlider tone={tone} value={opacity} onChange={setOpacity} className="max-w-[230px] flex-1" />}
                    <button onClick={onExpand} className={cn("flex items-center gap-1.5 text-[13px] transition-colors", film ? "text-white/60 hover:text-white" : "text-fg-muted hover:text-fg")} title="Fullscreen (F)">
                        <Maximize2 className="h-3.5 w-3.5" /> Fullscreen
                    </button>
                </div>
            </div>

            <figcaption className={cn("flex flex-wrap items-center justify-between gap-3 text-[13px] leading-snug", film ? "text-white/55" : "text-fg-muted")}>
                {overlay ? (
                    <>
                        <span>
                            <span className={film ? "text-white/90" : "text-fg"}>Attention map for {finding.toLowerCase()}.</span> Warm regions moved the score most. It is
                            not a lesion outline.
                        </span>
                        <JetLegend className={film ? "!text-white/55" : undefined} />
                    </>
                ) : (
                    "No attention overlay was returned for this image, so only the original is shown."
                )}
            </figcaption>
        </figure>
    );
}
