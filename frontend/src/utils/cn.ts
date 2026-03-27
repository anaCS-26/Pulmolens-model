export function cn(...args: (string | false | undefined)[]) {
    return args.filter(Boolean).join(" ");
}
