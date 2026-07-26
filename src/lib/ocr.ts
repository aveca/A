import Tesseract from "tesseract.js";
import type { DetectedTextLine } from "../types";

let workerPromise: Promise<Tesseract.Worker> | null = null;

async function getWorker(): Promise<Tesseract.Worker> {
  if (!workerPromise) {
    workerPromise = (async () => {
      const worker = await Tesseract.createWorker(["eng", "fra"], 1, {
        logger: () => {},
      });
      await worker.setParameters({
        tessedit_pageseg_mode: Tesseract.PSM.SPARSE_TEXT,
        preserve_interword_spaces: "1",
      });
      return worker;
    })();
  }
  return workerPromise;
}

export interface CropRegion {
  sx: number;
  sy: number;
  sw: number;
  sh: number;
}

export function preprocessForOCR(
  source: HTMLCanvasElement,
  region: CropRegion
): HTMLCanvasElement {
  const { sx, sy, sw, sh } = region;
  const safeW = Math.max(1, sw);
  const safeH = Math.max(1, sh);
  const scale = 2;
  const out = document.createElement("canvas");
  out.width = safeW * scale;
  out.height = safeH * scale;
  const octx = out.getContext("2d", { willReadFrequently: true })!;
  octx.imageSmoothingEnabled = true;
  octx.imageSmoothingQuality = "high";
  octx.drawImage(source, sx, sy, safeW, safeH, 0, 0, out.width, out.height);

  const imgData = octx.getImageData(0, 0, out.width, out.height);
  const d = imgData.data;
  const total = out.width * out.height;

  const hist = new Array(256).fill(0);
  for (let i = 0; i < d.length; i += 4) {
    const gray = (0.299 * d[i] + 0.587 * d[i + 1] + 0.114 * d[i + 2]) | 0;
    d[i + 3] = gray;
    hist[gray]++;
  }

  let sum = 0;
  for (let i = 0; i < 256; i++) sum += i * hist[i];
  let sumB = 0, wB = 0, maxVar = -1, threshold = 127;
  for (let i = 0; i < 256; i++) {
    wB += hist[i];
    if (wB === 0) continue;
    const wF = total - wB;
    if (wF === 0) break;
    sumB += i * hist[i];
    const mB = sumB / wB;
    const mF = (sum - sumB) / wF;
    const between = wB * wF * (mB - mF) * (mB - mF);
    if (between > maxVar) {
      maxVar = between;
      threshold = i;
    }
  }

  for (let i = 0; i < d.length; i += 4) {
    const gray = d[i + 3];
    const v = gray > threshold ? 255 : 0;
    d[i] = v;
    d[i + 1] = v;
    d[i + 2] = v;
    d[i + 3] = 255;
  }
  octx.putImageData(imgData, 0, 0);
  return out;
}

export async function recognizeText(
  canvas: HTMLCanvasElement
): Promise<DetectedTextLine[]> {
  try {
    const worker = await getWorker();
    const { data } = await worker.recognize(canvas);
    const lines: DetectedTextLine[] = (data.lines || [])
      .filter((l: any) => l.text && l.text.trim().length >= 2 && l.confidence >= 45)
      .map((l: any) => ({
        text: l.text.trim(),
        confidence: Math.round(l.confidence),
        bbox: {
          x0: l.bbox?.x0 ?? 0,
          y0: l.bbox?.y0 ?? 0,
          x1: l.bbox?.x1 ?? canvas.width,
          y1: l.bbox?.y1 ?? canvas.height,
        },
      }));
    return lines;
  } catch (e) {
    console.warn("OCR failed", e);
    return [];
  }
}

export async function terminateOCR(): Promise<void> {
  if (workerPromise) {
    try {
      const worker = await workerPromise;
      await worker.terminate();
    } catch {
      // ignore
    }
    workerPromise = null;
  }
}
