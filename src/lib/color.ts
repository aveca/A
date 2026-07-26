import type { ColorFingerprint } from "../types";

export function computeColorFingerprint(
  ctx: CanvasRenderingContext2D,
  sx: number,
  sy: number,
  sw: number,
  sh: number
): ColorFingerprint {
  sw = Math.max(1, Math.floor(sw));
  sh = Math.max(1, Math.floor(sh));
  const step = Math.max(2, Math.floor(Math.min(sw, sh) / 60));
  const data = ctx.getImageData(sx, sy, sw, sh).data;

  let rs = 0, gs = 0, bs = 0, count = 0;
  const hist = new Array(16).fill(0);

  for (let y = 0; y < sh; y += step) {
    for (let x = 0; x < sw; x += step) {
      const i = (y * sw + x) * 4;
      const r = data[i], g = data[i + 1], b = data[i + 2];
      rs += r; gs += g; bs += b; count++;
      const bucket = Math.floor(((r + g + b) / 3) / 16);
      hist[Math.min(15, bucket)]++;
    }
  }

  if (count === 0) {
    return { avg: [0, 0, 0], hist: new Array(16).fill(0) };
  }

  const total = hist.reduce((a, c) => a + c, 0) || 1;
  const histNorm = hist.map((h) => h / total);

  return {
    avg: [Math.round(rs / count), Math.round(gs / count), Math.round(bs / count)],
    hist: histNorm.map((v) => Number(v.toFixed(4))),
  };
}

export function colorDistance(a: ColorFingerprint, b: ColorFingerprint): number {
  const dr = a.avg[0] - b.avg[0];
  const dg = a.avg[1] - b.avg[1];
  const db = a.avg[2] - b.avg[2];
  const euclid = Math.sqrt(dr * dr + dg * dg + db * db) / 441;

  let chi = 0;
  for (let i = 0; i < 16; i++) {
    const d = a.hist[i] - b.hist[i];
    chi += (d * d) / (a.hist[i] + b.hist[i] + 1e-6);
  }
  chi = Math.sqrt(chi) / 4;

  return Math.min(1, euclid * 0.5 + chi * 0.5);
}

export function guessShape(
  ctx: CanvasRenderingContext2D,
  sx: number,
  sy: number,
  sw: number,
  sh: number
): DetectionShape {
  const { avg } = computeColorFingerprint(ctx, sx, sy, sw, sh);
  const ratio = sw / Math.max(1, sh);

  if (ratio > 0.55 && ratio < 0.85) return "bottle";
  if (ratio > 0.9 && ratio < 1.1) return "can";
  if (ratio < 0.5) return "bottle";
  if (ratio > 1.6) return "bag";
  if (ratio > 1.1 && ratio < 1.6) return "box";
  if (avg[0] > 200 && avg[1] > 200 && avg[2] > 200) return "box";
  return "round";
}

export type DetectionShape = "box" | "bottle" | "can" | "round" | "bag" | "unknown";

export function guessCategory(text: string, shape: DetectionShape): string {
  const t = text.toLowerCase();
  if (/yaourt|lait|fromage|beurre|creme|yogourt|cheese|milk/.test(t)) return "dairy";
  if (/coca|pepsi|fanta|sprite|eau|water|jus|juice|soda|biere|beer|vin|wine/.test(t)) return "beverage";
  if (/gateau|biscuit|cookie|cake|chocolat|chocolate|bonbon|candy|lu| Petit beurre|pretzel/.test(t)) return "snack";
  if (/pates|pasta|riz|rice|farine|flour|sucre|sugar|huile|oil|sauce/.test(t)) return "grocery";
  if (/savon|soap|shampoing|dentifrice|gel|douche/.test(t)) return "hygiene";
  if (/pain|bread|viennoiserie|croissant/.test(t)) return "bakery";
  if (/legume|vegetable|tomate|salade|fruit|pomme|banane/.test(t)) return "produce";
  if (shape === "bottle") return "beverage";
  if (shape === "can") return "beverage";
  return "unknown";
}
