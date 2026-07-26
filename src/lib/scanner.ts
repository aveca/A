import { BarcodeScanner } from "./barcode";
import { recognizeText, preprocessForOCR, type CropRegion } from "./ocr";
import { extractPrices } from "./prices";
import { computeColorFingerprint, guessShape } from "./color";
import { findKnownProduct, saveProduct } from "./productMemory";
import type {
  DetectedBarcode,
  DetectionResult,
  ScanFrame,
  RecognitionMatch,
} from "../types";

const OCR_INTERVAL_MS = 1200;
const COLOR_INTERVAL_MS = 600;
const SCAN_REGION_RATIO = { x: 0.1, y: 0.2, w: 0.8, h: 0.6 };

export interface ScannerCallbacks {
  onFrame?: (frame: ScanFrame) => void;
  onDetection?: (detection: DetectionResult) => void;
  onMatch?: (match: RecognitionMatch, detection: DetectionResult) => void;
  onStatus?: (status: ScannerStatus) => void;
  onError?: (msg: string) => void;
}

export interface ScannerStatus {
  cameraReady: boolean;
  ocrRunning: boolean;
  lastOcrAt: number;
  lastBarcodeAt: number;
  fps: number;
}

export class Scanner {
  private video: HTMLVideoElement;
  private canvas: HTMLCanvasElement;
  private ctx: CanvasRenderingContext2D;
  private barcode = new BarcodeScanner();
  private cb: ScannerCallbacks;
  private rafId: number | null = null;
  private running = false;
  private lastOcrAt = 0;
  private lastColorAt = 0;
  private lastBarcodeAt = 0;
  private frameTimes: number[] = [];
  private knownBarcodes = new Set<string>();
  private stream: MediaStream | null = null;
  private ocrBusy = false;

  public status: ScannerStatus = {
    cameraReady: false,
    ocrRunning: false,
    lastOcrAt: 0,
    lastBarcodeAt: 0,
    fps: 0,
  };

  constructor(video: HTMLVideoElement, cb: ScannerCallbacks) {
    this.video = video;
    this.canvas = document.createElement("canvas");
    this.ctx = this.canvas.getContext("2d", { willReadFrequently: true })!;
    this.cb = cb;
  }

  async start(): Promise<void> {
    if (this.running) return;
    this.running = true;

    try {
      this.stream = await navigator.mediaDevices.getUserMedia({
        video: {
          facingMode: { ideal: "environment" },
          width: { ideal: 1920 },
          height: { ideal: 1080 },
        },
        audio: false,
      });
      this.video.srcObject = this.stream;
      await this.video.play();
      this.status.cameraReady = true;
      this.emitStatus();
    } catch (e) {
      this.running = false;
      this.cb.onError?.(`Caméra inaccessible: ${(e as Error).message}`);
      return;
    }

    this.barcode.decodeFromVideo(this.video, (b) => this.onBarcode(b));
    this.loop();
  }

  stop(): void {
    this.running = false;
    if (this.rafId) cancelAnimationFrame(this.rafId);
    this.rafId = null;
    this.barcode.stop();
    if (this.stream) {
      this.stream.getTracks().forEach((t) => t.stop());
      this.stream = null;
    }
    this.status.cameraReady = false;
    this.status.ocrRunning = false;
    this.emitStatus();
  }

  private emitStatus() {
    this.cb.onStatus?.({ ...this.status });
  }

  private getRegion(): CropRegion {
    const w = this.canvas.width || this.video.videoWidth || 1280;
    const h = this.canvas.height || this.video.videoHeight || 720;
    return {
      sx: Math.round(w * SCAN_REGION_RATIO.x),
      sy: Math.round(h * SCAN_REGION_RATIO.y),
      sw: Math.round(w * SCAN_REGION_RATIO.w),
      sh: Math.round(h * SCAN_REGION_RATIO.h),
    };
  }

  private onBarcode(b: DetectedBarcode) {
    this.lastBarcodeAt = performance.now();
    this.status.lastBarcodeAt = this.lastBarcodeAt;
    if (this.knownBarcodes.has(b.text)) return;
    this.knownBarcodes.add(b.text);
    setTimeout(() => this.knownBarcodes.delete(b.text), 4000);
    this.runFullDetection("barcode", [b]);
  }

  private async loop() {
    if (!this.running) return;
    const now = performance.now();
    this.frameTimes.push(now);
    if (this.frameTimes.length > 30) this.frameTimes.shift();
    if (this.frameTimes.length > 5) {
      this.status.fps = Math.round(
        (1000 * (this.frameTimes.length - 1)) /
          (this.frameTimes[this.frameTimes.length - 1] - this.frameTimes[0])
      );
    }

    if (this.video.readyState >= 2 && this.video.videoWidth > 0) {
      const w = this.video.videoWidth;
      const h = this.video.videoHeight;
      if (this.canvas.width !== w) this.canvas.width = w;
      if (this.canvas.height !== h) this.canvas.height = h;
      this.ctx.drawImage(this.video, 0, 0, w, h);

      if (now - this.lastColorAt > COLOR_INTERVAL_MS) {
        this.lastColorAt = now;
        const region = this.getRegion();
        const detection = this.buildDetection("shape", [], [], true, region);
        const match = await findKnownProduct(detection);
        if (match) {
          this.cb.onMatch?.(match, detection);
        }
        this.cb.onFrame?.({ detection, match, thumbnail: null });
      }

      if (now - this.lastOcrAt > OCR_INTERVAL_MS && !this.ocrBusy) {
        this.lastOcrAt = now;
        this.ocrBusy = true;
        this.status.ocrRunning = true;
        this.emitStatus();
        this.runFullDetection("ocr").finally(() => {
          this.ocrBusy = false;
          this.status.ocrRunning = false;
          this.emitStatus();
        });
      }
    }

    this.rafId = requestAnimationFrame(() => this.loop());
  }

  private buildDetection(
    method: DetectionResult["method"],
    barcodes: DetectedBarcode[],
    textLines: DetectionResult["textLines"],
    includeColor: boolean,
    region: CropRegion
  ): DetectionResult {
    const text = textLines.map((l) => l.text).join(" ");
    const prices = extractPrices(text);
    let colorFingerprint = null;
    let dominantShape: DetectionResult["dominantShape"] = "unknown";
    if (includeColor) {
      colorFingerprint = computeColorFingerprint(
        this.ctx,
        region.sx,
        region.sy,
        region.sw,
        region.sh
      );
      dominantShape = guessShape(
        this.ctx,
        region.sx,
        region.sy,
        region.sw,
        region.sh
      );
    }
    return {
      method,
      barcodes,
      textLines,
      prices,
      colorFingerprint,
      dominantShape,
      timestamp: Date.now(),
    };
  }

  private async runFullDetection(
    method: DetectionResult["method"],
    presetBarcodes: DetectedBarcode[] = [],
    presetText: DetectionResult["textLines"] = []
  ) {
    if (this.video.readyState < 2) return;
    const w = this.canvas.width;
    const h = this.canvas.height;
    this.ctx.drawImage(this.video, 0, 0, w, h);
    const region = this.getRegion();

    let barcodes = presetBarcodes;
    if (barcodes.length === 0) {
      const b = await this.barcode.decodeFromCanvas(this.canvas);
      if (b) barcodes = [b];
    }

    let textLines = presetText;
    if (textLines.length === 0) {
      const ocrCanvas = preprocessForOCR(this.canvas, region);
      textLines = await recognizeText(ocrCanvas);
    }

    const detection = this.buildDetection(method, barcodes, textLines, true, region);
    this.cb.onDetection?.(detection);

    const match = await findKnownProduct(detection);
    if (match) {
      this.cb.onMatch?.(match, detection);
    }

    const thumb = this.makeThumbnail();
    this.cb.onFrame?.({ detection, match, thumbnail: thumb });
  }

  private makeThumbnail(): string | null {
    try {
      const region = this.getRegion();
      const tw = 160;
      const th = Math.round((region.sh / region.sw) * tw);
      const tc = document.createElement("canvas");
      tc.width = tw;
      tc.height = th;
      tc.getContext("2d")!.drawImage(
        this.canvas,
        region.sx,
        region.sy,
        region.sw,
        region.sh,
        0,
        0,
        tw,
        th
      );
      return tc.toDataURL("image/jpeg", 0.6);
    } catch {
      return null;
    }
  }

  async saveCurrent(nameOverride?: string): Promise<void> {
    const w = this.canvas.width;
    const h = this.canvas.height;
    if (w === 0 || h === 0) return;
    this.ctx.drawImage(this.video, 0, 0, w, h);
    const region = this.getRegion();
    const thumb = this.makeThumbnail();
    let barcodes: DetectedBarcode[] = [];
    const b = await this.barcode.decodeFromCanvas(this.canvas);
    if (b) barcodes = [b];
    const ocrCanvas = preprocessForOCR(this.canvas, region);
    const textLines = await recognizeText(ocrCanvas);
    const detection = this.buildDetection("memory", barcodes, textLines, true, region);
    await saveProduct(detection, thumb, nameOverride);
  }
}
