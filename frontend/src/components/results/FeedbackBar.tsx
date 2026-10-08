import React, { useState } from "react";
import { cn } from "../../utils/cn";

type Rating = "good" | "bad";

interface FeedbackBarProps {
    onSubmit: (rating: Rating) => Promise<void>;
    disabled?: boolean;
}

export function FeedbackBar({ onSubmit, disabled }: FeedbackBarProps) {
    const [state, setState] = useState<"idle" | "sending" | "sent" | "error">("idle");
    const [rating, setRating] = useState<Rating | null>(null);

    const send = async (r: Rating) => {
        if (state === "sending" || state === "sent") return;
        setRating(r);
        setState("sending");
        try {
            await onSubmit(r);
            setState("sent");
        } catch (e) {
            console.error("Feedback failed", e);
            setState("error");
        }
    };

    if (state === "sent") {
        return <p className="text-[14px] text-fg-soft">Thanks. Your feedback was sent to the team.</p>;
    }

    const opt = (r: Rating, label: string) => (
        <button
            onClick={() => send(r)}
            disabled={disabled || state === "sending"}
            className={cn("rounded-lg border border-line px-3 py-1.5 text-[13.5px] transition-colors hover:bg-surface-sunk disabled:opacity-40", state === "sending" && rating === r && "text-fg-muted")}
        >
            {state === "sending" && rating === r ? "Sending…" : label}
        </button>
    );

    return (
        <div>
            <div className="flex flex-wrap items-center gap-x-3 gap-y-2">
                <span className="mr-2 text-[14px] font-medium">Was this analysis helpful?</span>
                {opt("good", "Yes, helpful")}
                {opt("bad", "No, not accurate")}
            </div>
            <p className={cn("mt-2 text-[12.5px]", state === "error" ? "text-urgent" : "text-fg-muted")}>
                {state === "error" ? "Couldn't send feedback. Please try again." : "Sending feedback shares this image and its scores with the team."}
            </p>
        </div>
    );
}
