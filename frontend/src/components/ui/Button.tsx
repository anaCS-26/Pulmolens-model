import React from "react";
import { cn } from "../../utils/cn";

type Variant = "primary" | "outline" | "text";
type Size = "sm" | "md" | "lg";

interface ButtonProps extends React.ButtonHTMLAttributes<HTMLButtonElement> {
    variant?: Variant;
    size?: Size;
}

const VARIANTS: Record<Variant, string> = {
    primary: "bg-solid text-solid-fg hover:opacity-90",
    outline: "border border-line text-fg hover:bg-surface-sunk",
    text: "text-fg underline decoration-fg/25 underline-offset-[3px] hover:decoration-accent",
};

const SIZES: Record<Size, string> = {
    sm: "h-[34px] px-3.5 text-[13.5px] gap-1.5",
    md: "h-10 px-4 text-[14.5px] gap-2",
    lg: "h-11 px-5 text-[15px] gap-2.5",
};

export const Button = React.forwardRef<HTMLButtonElement, ButtonProps>(function Button(
    { variant = "outline", size = "md", className, ...rest },
    ref
) {
    return (
        <button
            ref={ref}
            className={cn(
                "inline-flex items-center justify-center rounded-lg font-medium whitespace-nowrap select-none",
                "transition-[background-color,opacity,transform] active:translate-y-px",
                "disabled:cursor-not-allowed disabled:opacity-35 disabled:active:translate-y-0",
                SIZES[size],
                VARIANTS[variant],
                variant === "text" && "!h-auto !px-0",
                className
            )}
            {...rest}
        />
    );
});
