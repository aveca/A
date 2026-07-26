export type DetectionMethod = "barcode" | "ocr" | "shape" | "memory";

export interface DetectedBarcode {
  format: string;
  text: string;
  corners: { x: number; y: number }[];
}

export interface DetectedTextLine {
  text: string;
  confidence: number;
  bbox: { x0: number; y0: number; x1: number; y1: number };
}

export interface ExtractedPrice {
  amount: number;
  currency: string;
  raw: string;
}

export interface ColorFingerprint {
  avg: [number, number, number];
  hist: number[];
}

export interface DetectionResult {
  method: DetectionMethod;
  barcodes: DetectedBarcode[];
  textLines: DetectedTextLine[];
  prices: ExtractedPrice[];
  colorFingerprint: ColorFingerprint | null;
  dominantShape: "box" | "bottle" | "can" | "round" | "bag" | "unknown";
  timestamp: number;
}

export interface Product {
  id?: string;
  name: string;
  barcode: string | null;
  price: number | null;
  currency: string | null;
  detected_text: string | null;
  keywords: string[] | null;
  color_fingerprint: ColorFingerprint | null;
  thumbnail: string | null;
  category: string | null;
  times_scanned: number;
  created_at?: string;
  updated_at?: string;
}

export interface RecognitionMatch {
  product: Product;
  method: DetectionMethod;
  confidence: number;
}

export interface ScanFrame {
  detection: DetectionResult;
  match: RecognitionMatch | null;
  thumbnail: string | null;
}
