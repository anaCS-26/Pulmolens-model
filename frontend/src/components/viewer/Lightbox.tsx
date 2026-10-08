import React, { useCallback, useEffect, useRef, useState } from "react";
import { createPortal } from "react-dom";
import { motion } from "framer-motion";
import { cn } from "../../utils/cn";
import { OpacitySlider } from "./XrayViewer";

interface LightboxProps {
    title: string;
    imageURL: string;
    overlay: string | null;
    opacity: number;
    setOpacity: (v: number) => void;
    onClose: () => void;
}

const MIN = 1;
const MAX = 6;

/** Fullscreen viewer: scroll to zoom toward the cursor, drag to pan. */
export function Lightbox({ title, imageURL, overlay, opacity, setOpacity, onClose }: LightboxProps) {
    const stageRef = useRef<HTMLDivElement>(null);
    const [view, setView] = useState({ s: 1, x: 0, y: 0 });
    const [showOverlay, setShowOverlay] = useState(true);
    const drag = useRef<{ x: number; y: number; ox: number; oy: number } | null>(null);
    const [dragging, setDragging] = useState(false);

    const zoomAt = useCallback((factor: number, clientX?: number, clientY?: number) => {
        setView((v) => {
            const s2 = Math.min(MAX, Math.max(MIN, v.s * factor));
            if (s2 === 1) return { s: 1, x: 0, y: 0 };
            const r = stageRef.current?.getBoundingClientRect();
            // Keep the point under the cursor fixed while scaling.
            const px = r && clientX !== undefined ? clientX - (r.left + r.width / 2) : 0;
            const py = r && clientY !== undefined ? clientY - (r.top + r.height / 2) : 0;
            const k = s2 / v.s;
            return { s: s2, x: px - (px - v.x) * k, y: py - (py - v.y) * k };
        });
    }, []);

    const reset = () => setView({ s: 1, x: 0, y: 0 });

    useEffect(() => {
        const prev = document.body.style.overflow;
        document.body.style.overflow = "hidden";
        const onKey = (e: KeyboardEvent) => {
            if (e.key === "Escape") onClose();
            else if (e.key === "+" || e.key === "=") zoomAt(1.25);
            else if (e.key === "-") zoomAt(0.8);
            else if (e.key === "0") reset();
            else if (e.key.toLowerCase() === "o" && overlay) setShowOverlay((v) => !v);
        };
        window.addEventListener("keydown", onKey);
        return () => {
            document.body.style.overflow = prev;
            window.removeEventListener("keydown", onKey);
        };
    }, [onClose, zoomAt, overlay]);

    // Non-passive wheel listener so the page doesn't scroll underneath.
    useEffect(() => {
        const el = stageRef.current;
        if (!el) return;
        const onWheel = (e: WheelEvent) => {
            e.preventDefault();
            zoomAt(e.deltaY < 0 ? 1.12 : 1 / 1.12, e.clientX, e.clientY);
        };
        el.addEventListener("wheel", onWheel, { passive: false });
        return () => el.removeEventListener("wheel", onWheel);
    }, [zoomAt]);

    const ctl = "font-mono text-[12px] text-white/60 transition-colors hover:text-white disabled:opacity-30 disabled:hover:text-white/60";

    return createPortal(
        <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            transition={{ duration: 0.15 }}
            className="fixed inset-0 z-[60] flex flex-col bg-film text-white print:hidden"
            role="dialog"
            aria-modal="true"
            aria-label="Fullscreen X-ray viewer"
        >
            <div className="flex items-center justify-between gap-4 border-b border-white/15 px-5 py-3">
                <div className="film-text min-w-0 truncate !text-[11px]">{title}</div>
                <div className="flex items-center gap-5">
                    <button onClick={() => zoomAt(0.8)} disabled={view.s <= MIN} className={ctl} aria-label="Zoom out">−</button>
                    <span className="num w-12 text-center font-mono text-[12px] text-white">{Math.round(view.s * 100)}%</span>
                    <button onClick={() => zoomAt(1.25)} disabled={view.s >= MAX} className={ctl} aria-label="Zoom in">+</button>
                    <button onClick={reset} className={ctl} title="Reset (0)">Reset</button>
                    {overlay && (
                        <button onClick={() => setShowOverlay((v) => !v)} className={cn(ctl, showOverlay && "text-white")} aria-pressed={showOverlay} title="Toggle overlay (O)">
                            Overlay {showOverlay ? "on" : "off"}
                        </button>
                    )}
                    <button onClick={onClose} className={cn(ctl, "text-white")} aria-label="Close viewer">Close ✕</button>
                </div>
            </div>

            <div
                ref={stageRef}
                className={cn("relative flex-1 overflow-hidden touch-none", view.s > 1 ? (dragging ? "cursor-grabbing" : "cursor-grab") : "cursor-zoom-in")}
                onDoubleClick={(e) => (view.s > 1 ? reset() : zoomAt(2.5, e.clientX, e.clientY))}
                onPointerDown={(e) => {
                    if (view.s <= 1) return;
                    (e.currentTarget as HTMLElement).setPointerCapture(e.pointerId);
                    drag.current = { x: e.clientX, y: e.clientY, ox: view.x, oy: view.y };
                    setDragging(true);
                }}
                onPointerMove={(e) => {
                    const d = drag.current;
                    if (!d) return;
                    setView((v) => ({ ...v, x: d.ox + e.clientX - d.x, y: d.oy + e.clientY - d.y }));
                }}
                onPointerUp={() => { drag.current = null; setDragging(false); }}
            >
                <div className="absolute inset-0 p-4">
                    <div
                        className={cn("relative h-full w-full", !dragging && "transition-transform duration-150 ease-out")}
                        style={{ transform: `translate(${view.x}px, ${view.y}px) scale(${view.s})` }}
                    >
                        <img src={imageURL} alt="Chest X-ray" draggable={false} className="absolute inset-0 h-full w-full object-contain" />
                        {overlay && showOverlay && (
                            <img src={overlay} alt="Attention overlay" draggable={false} className="absolute inset-0 h-full w-full object-contain transition-opacity duration-200" style={{ opacity }} />
                        )}
                    </div>
                </div>
            </div>

            <div className="flex flex-wrap items-center justify-between gap-4 border-t border-white/15 px-5 py-3">
                <div className="hidden gap-5 font-mono text-[11px] text-white/40 sm:flex">
                    <span>Scroll: zoom</span><span>Drag: pan</span><span>Double-click: 2.5×</span>{overlay && <span>O: overlay</span>}<span>Esc: close</span>
                </div>
                {overlay && showOverlay && <OpacitySlider tone="film" value={opacity} onChange={setOpacity} className="w-full max-w-xs" />}
            </div>
        </motion.div>,
        document.body
    );
}
