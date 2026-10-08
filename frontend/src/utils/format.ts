/** "Pleural_Thickening" -> "Pleural thickening" */
export function prettyLabel(label: string): string {
  const s = label.replace(/_/g, " ").toLowerCase();
  return s.charAt(0).toUpperCase() + s.slice(1);
}

export function pct(p: number, digits = 0): string {
  return `${(p * 100).toFixed(digits)}%`;
}

export function capitalize(s: string): string {
  return s.charAt(0).toUpperCase() + s.slice(1);
}
