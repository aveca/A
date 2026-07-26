import { BrowserMultiFormatReader, IScannerControls } from "@zxing/browser";
import { DecodeHintType, BarcodeFormat } from "@zxing/library";
import type { DetectedBarcode } from "../types";

export class BarcodeScanner {
  private reader: BrowserMultiFormatReader;
  private controls: IScannerControls | null = null;

  constructor() {
    const hints = new Map();
    hints.set(DecodeHintType.POSSIBLE_FORMATS, [
      BarcodeFormat.EAN_13,
      BarcodeFormat.EAN_8,
      BarcodeFormat.UPC_A,
      BarcodeFormat.UPC_E,
      BarcodeFormat.CODE_128,
      BarcodeFormat.CODE_39,
      BarcodeFormat.QR_CODE,
      BarcodeFormat.DATA_MATRIX,
      BarcodeFormat.ITF,
    ]);
    hints.set(DecodeHintType.TRY_HARDER, true);
    this.reader = new BrowserMultiFormatReader(hints, {
      delayBetweenScanAttempts: 120,
    });
  }

  async decodeFromVideo(
    video: HTMLVideoElement,
    onDetected: (b: DetectedBarcode) => void
  ): Promise<void> {
    this.stop();
    this.controls = await this.reader.decodeFromVideoDevice(
      undefined,
      video,
      (result, _err, controls) => {
        if (result) {
          const fmt = result.getBarcodeFormat().toString();
          const text = result.getText();
          const pts = result.getResultPoints() || [];
          onDetected({
            format: fmt,
            text,
            corners: pts.map((p) => ({ x: p.getX(), y: p.getY() })),
          });
        }
      }
    );
  }

  async decodeFromCanvas(canvas: HTMLCanvasElement): Promise<DetectedBarcode | null> {
    try {
      const result = await this.reader.decodeFromCanvas(canvas);
      if (!result) return null;
      const fmt = result.getBarcodeFormat().toString();
      const pts = result.getResultPoints() || [];
      return {
        format: fmt,
        text: result.getText(),
        corners: pts.map((p) => ({ x: p.getX(), y: p.getY() })),
      };
    } catch {
      return null;
    }
  }

  stop() {
    if (this.controls) {
      this.controls.stop();
      this.controls = null;
    }
  }
}
