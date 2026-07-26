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

export async function recognizeText(
  canvas: HTMLCanvasElement
): Promise<DetectedTextLine[]> {
  try {
    const worker = await getWorker();
    const { data } = await worker.recognize(canvas);
    const lines: DetectedTextLine[] = (data.lines || [])
      .filter((l: any) => l.text && l.text.trim().length >= 2)
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
