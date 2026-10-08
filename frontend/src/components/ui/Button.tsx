import React from "react";
import { cn } from "../../utils/cn";

type Variant = "primary" | "outline" | "text";
type Size = "sm" | "md";

interface ButtonProps extends React.ButtonHTMLAttributes<HTMLButtonElement> {
    variant?: Variant;
    size?: Size;
}

const VARIANTS: Record<Variant, string> = {
    primary: "bg-ink text-paper hover:bg-black",
    outline: "border border-ink/25 text-ink hover:border-ink",
    text: "text-ink underline decoration-ink/30 underline-offset-[3px] hover:decoration-ink",
};

const SIZES: Record<Size, string> = {
    sm: "h-8 px-3 text-[13px] gap-1.5",
    md: "h-10 px-4 text-sm gap-2",
};

export const Button = React.forwardRef<HTMLButtonElement, ButtonProps>(function Button(
    { variant = "outline", size = "md", className, ...rest },
    ref
) {
    return (
        <button
            ref={ref}
            className={cn(
                "inline-flex items-center justify-center rounded font-medium whitespace-nowrap select-none transition-colors",
                "disabled:cursor-not-allowed disabled:opacity-35",
                SIZES[size],
                VARIANTS[variant],
                variant === "text" && "!px-0",
                className
            )}
            {...rest}
        />
    );
});
