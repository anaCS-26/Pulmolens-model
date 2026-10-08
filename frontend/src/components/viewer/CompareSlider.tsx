import React, { useCallback, useEffect, useRef, useState } from "react";
import { animate } from "framer-motion";
import { cn } from "../../utils/cn";

interface CompareSliderProps {
    before: React.ReactNode;
    after: React.ReactNode;
    beforeLabel?: string;
    afterLabel?: string;
    /** Sweep back and forth on its own until the user grabs it. */
    autoplay?: boolean;
    className?: string;
}

/**
 * Two layers stacked exactly on top of each other; a draggable divider reveals
 * `before` on the left and `after` on the right.
 */
export function CompareSlider({ before, after, beforeLabel = "Original", afterLabel = "Overlay", autoplay, className }: CompareSliderProps) {
    const ref = useRef<HTMLDivElement>(null);
    const [pos, setPos] = useState(50);
    const [dragging, setDragging] = useState(false);
    const [touched, setTouched] = useState(false);

    useEffect(() => {
        if (!autoplay || touched) return;
        const controls = animate(50, [50, 22, 78, 50], {
            duration: 6,
            ease: "easeInOut",
            repeat: Infinity,
            onUpdate: setPos,
        });
        return () => controls.stop();
    }, [autoplay, touched]);

    const setFromClientX = useCallback((clientX: number) => {
        const r = ref.current?.getBoundingClientRect();
        if (!r) return;
        setPos(Math.min(100, Math.max(0, ((clientX - r.left) / r.width) * 100)));
    }, []);

    const onPointerDown = (e: React.PointerEvent) => {
        setTouched(true);
        setDragging(true);
        (e.currentTarget as HTMLElement).setPointerCapture(e.pointerId);
        setFromClientX(e.clientX);
    };

    const onKey = (e: React.KeyboardEvent) => {
        const step = e.shiftKey ? 10 : 2;
        if (e.key === "ArrowLeft") { setTouched(true); setPos((p) => Math.max(0, p - step)); e.preventDefault(); }
        if (e.key === "ArrowRight") { setTouched(true); setPos((p) => Math.min(100, p + step)); e.preventDefault(); }
    };

    return (
        <div
            ref={ref}
            className={cn("relative select-none touch-none overflow-hidden", dragging ? "cursor-grabbing" : "cursor-ew-resize", className)}
            onPointerDown={onPointerDown}
            onPointerMove={(e) => dragging && setFromClientX(e.clientX)}
            onPointerUp={() => setDragging(false)}
            onPointerCancel={() => setDragging(false)}
        >
            <div className="absolute inset-0">{after}</div>
            <div className="absolute inset-0" style={{ clipPath: `inset(0 ${100 - pos}% 0 0)` }}>
                {before}
            </div>

            <span className="film-text pointer-events-none absolute bottom-2.5 left-3" style={{ opacity: pos > 14 ? 1 : 0, transition: "opacity 200ms" }}>
                {beforeLabel}
            </span>
            <span className="film-text pointer-events-none absolute bottom-2.5 right-3" style={{ opacity: pos < 86 ? 1 : 0, transition: "opacity 200ms" }}>
                {afterLabel}
            </span>

            <div className="pointer-events-none absolute inset-y-0" style={{ left: `${pos}%` }}>
                <div className="absolute inset-y-0 -translate-x-1/2 w-px bg-white" />
                <div
                    role="slider"
                    tabIndex={0}
                    aria-label="Compare original and overlay"
                    aria-valuemin={0}
                    aria-valuemax={100}
                    aria-valuenow={Math.round(pos)}
                    onKeyDown={onKey}
                    className={cn(
                        "pointer-events-auto absolute top-1/2 grid h-7 w-3.5 -translate-x-1/2 -translate-y-1/2 place-items-center rounded-[2px]",
                        "bg-white ring-1 ring-black/40 transition-[height]",
                        dragging && "h-9"
                    )}
                >
                    <span className="h-3 w-px bg-black/50" />
                </div>
            </div>
        </div>
    );
}
