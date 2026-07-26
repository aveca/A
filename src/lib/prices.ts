import type { ExtractedPrice } from "../types";

const CURRENCY_PATTERNS: { currency: string; regex: RegExp }[] = [
  { currency: "EUR", regex: /(\d+(?:[.,]\d{1,2})?)\s*€/g },
  { currency: "EUR", regex: /€\s*(\d+(?:[.,]\d{1,2})?)/g },
  { currency: "USD", regex: /(\d+(?:[.,]\d{1,2})?)\s*\$/g },
  { currency: "USD", regex: /\$\s*(\d+(?:[.,]\d{1,2})?)/g },
  { currency: "GBP", regex: /(\d+(?:[.,]\d{1,2})?)\s*£/g },
  { currency: "XOF", regex: /(\d[\d\s.,]*)\s*(?:FCFA|CFA|XOF)/gi },
  { currency: "XOF", regex: /(?:FCFA|CFA|XOF)\s*(\d[\d\s.,]*)/gi },
  { currency: "XAF", regex: /(\d[\d\s.,]*)\s*(?:FCFA|CFA|XAF)/gi },
];

export function extractPrices(text: string): ExtractedPrice[] {
  const found: ExtractedPrice[] = [];
  const seen = new Set<string>();

  for (const { currency, regex } of CURRENCY_PATTERNS) {
    const re = new RegExp(regex.source, regex.flags);
    let m: RegExpExecArray | null;
    while ((m = re.exec(text)) !== null) {
      const raw = m[0];
      if (seen.has(raw)) continue;
      seen.add(raw);
      const numStr = m[1].replace(/\s/g, "").replace(/\.(?=\d{3})/g, "").replace(",", ".");
      const amount = parseFloat(numStr);
      if (!isNaN(amount) && amount > 0 && amount < 100000) {
        found.push({ amount, currency, raw });
      }
    }
  }

  found.sort((a, b) => a.amount - b.amount);
  return found;
}
