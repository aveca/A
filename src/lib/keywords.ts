const STOP_WORDS = new Set([
  "le", "la", "les", "un", "une", "des", "du", "de", "the", "a", "an",
  "et", "and", "of", "for", "with", "au", "aux", "en", "in", "on",
  "new", "bio", "max", "mini", "g", "kg", "ml", "cl", "l", "mg",
]);

export function tokenize(text: string): string[] {
  const raw = text
    .toLowerCase()
    .normalize("NFD")
    .replace(/[\u0300-\u036f]/g, "")
    .replace(/[^a-z0-9\s]/g, " ")
    .split(/\s+/)
    .filter(Boolean);

  const out = raw.filter((w) => w.length >= 3 && !STOP_WORDS.has(w));
  return Array.from(new Set(out));
}

export function overlap(a: string[], b: string[]): number {
  const setA = new Set(a);
  const setB = new Set(b);
  let shared = 0;
  setB.forEach((w) => {
    if (setA.has(w)) shared++;
  });
  return shared;
}

export function jaccard(a: string[], b: string[]): number {
  if (a.length === 0 || b.length === 0) return 0;
  const setA = new Set(a);
  const setB = new Set(b);
  let inter = 0;
  setB.forEach((w) => {
    if (setA.has(w)) inter++;
  });
  const union = setA.size + setB.size - inter;
  return union === 0 ? 0 : inter / union;
}
