import { supabase } from "./supabase";
import type {
  ColorFingerprint,
  Product,
  RecognitionMatch,
  DetectionResult,
} from "../types";
import { tokenize, jaccard, overlap } from "./keywords";
import { colorDistance, guessCategory } from "./color";

let productCache: Product[] | null = null;
let cacheLoading: Promise<Product[]> | null = null;

export async function loadProductCache(): Promise<Product[]> {
  if (productCache) return productCache;
  if (cacheLoading) return cacheLoading;
  cacheLoading = (async () => {
    const { data, error } = await supabase
      .from("products")
      .select("*")
      .order("updated_at", { ascending: false });
    if (error) throw error;
    productCache = (data || []) as Product[];
    return productCache;
  })();
  return cacheLoading;
}

export function invalidateCache(): void {
  productCache = null;
  cacheLoading = null;
}

function bestName(detection: DetectionResult): { name: string; text: string } {
  const lines = [...detection.textLines].sort((a, b) => b.confidence - a.confidence);
  let name = "";
  let text = detection.textLines.map((l) => l.text).join(" ").trim();

  for (const l of lines) {
    const clean = l.text.replace(/[^a-zA-ZÀ-ÿ0-9\s'-]/g, " ").replace(/\s+/g, " ").trim();
    const words = clean.split(/\s+/).filter((w) => w.length >= 3);
    if (words.length >= 2 && clean.length <= 40) {
      const titleCase = words.map((w) => w[0].toUpperCase() + w.slice(1).toLowerCase()).join(" ");
      name = titleCase;
      break;
    }
  }

  if (!name) {
    const long = lines.find((l) => l.text.replace(/[^a-zA-ZÀ-ÿ0-9\s]/g, "").trim().length >= 3);
    if (long) {
      const clean = long.text.replace(/[^a-zA-ZÀ-ÿ0-9\s'-]/g, " ").replace(/\s+/g, " ").trim();
      name = clean.split(/\s+/).slice(0, 3).join(" ");
    }
  }

  if (!name) name = detection.barcodes[0]?.text || "Produit inconnu";
  if (!text) text = detection.textLines.map((l) => l.text).join(" ").trim();
  return { name, text };
}

export async function findKnownProduct(
  detection: DetectionResult
): Promise<RecognitionMatch | null> {
  const all = await loadProductCache();
  if (all.length === 0) return null;

  const barcode = detection.barcodes[0]?.text;
  if (barcode) {
    const match = all.find((p) => p.barcode === barcode);
    if (match) {
      return { product: match, method: "barcode", confidence: 1 };
    }
  }

  const text = detection.textLines.map((l) => l.text).join(" ");
  const tokens = tokenize(text);
  if (tokens.length > 0) {
    let best: { p: Product; score: number } | null = null;
    for (const p of all) {
      if (!p.keywords || p.keywords.length === 0) continue;
      const j = jaccard(tokens, p.keywords);
      const o = overlap(tokens, p.keywords);
      const score = j * 0.6 + Math.min(1, o / 3) * 0.4;
      if (score > 0.45 && (!best || score > best.score)) {
        best = { p, score };
      }
    }
    if (best) {
      return { product: best.p, method: "ocr", confidence: best.score };
    }
  }

  if (detection.colorFingerprint) {
    let best: { p: Product; score: number } | null = null;
    for (const p of all) {
      if (!p.color_fingerprint) continue;
      const dist = colorDistance(detection.colorFingerprint, p.color_fingerprint as ColorFingerprint);
      const score = 1 - dist;
      if (score > 0.7 && (!best || score > best.score)) {
        best = { p, score };
      }
    }
    if (best) {
      return { product: best.p, method: "shape", confidence: best.score };
    }
  }

  return null;
}

export async function saveProduct(
  detection: DetectionResult,
  thumbnail: string | null,
  nameOverride?: string
): Promise<Product> {
  const { name, text } = bestName(detection);
  const finalName = (nameOverride && nameOverride.trim()) || name;
  const price = detection.prices[0]?.amount ?? null;
  const currency = detection.prices[0]?.currency ?? null;
  const barcode = detection.barcodes[0]?.text || null;
  const keywords = tokenize(`${finalName} ${text}`);
  const category = guessCategory(`${finalName} ${text}`, detection.dominantShape);

  if (barcode) {
    const { data: existing } = await supabase
      .from("products")
      .select("*")
      .eq("barcode", barcode)
      .maybeSingle();
    if (existing) {
      const { data } = await supabase
        .from("products")
        .update({
          name: finalName,
          price: price ?? existing.price,
          currency: currency ?? existing.currency,
          detected_text: text || existing.detected_text,
          keywords: keywords.length ? keywords : existing.keywords,
          color_fingerprint: detection.colorFingerprint || existing.color_fingerprint,
          thumbnail: thumbnail || existing.thumbnail,
          category: category || existing.category,
          times_scanned: (existing.times_scanned || 0) + 1,
          updated_at: new Date().toISOString(),
        })
        .eq("id", existing.id)
        .select()
        .single();
      invalidateCache();
      return data as Product;
    }
  }

  const { data, error } = await supabase
    .from("products")
    .insert({
      name: finalName,
      barcode,
      price,
      currency,
      detected_text: text || null,
      keywords: keywords.length ? keywords : null,
      color_fingerprint: detection.colorFingerprint,
      thumbnail,
      category,
      times_scanned: 1,
    })
    .select()
    .single();
  if (error) throw error;
  invalidateCache();
  return data as Product;
}

export async function listProducts(): Promise<Product[]> {
  return loadProductCache();
}

export async function deleteProduct(id: string): Promise<void> {
  const { error } = await supabase.from("products").delete().eq("id", id);
  if (error) throw error;
  invalidateCache();
}

export async function renameProduct(id: string, name: string): Promise<void> {
  const keywords = tokenize(name);
  const { error } = await supabase
    .from("products")
    .update({ name, keywords, updated_at: new Date().toISOString() })
    .eq("id", id);
  if (error) throw error;
  invalidateCache();
}


export { listProducts, deleteProduct, renameProduct }