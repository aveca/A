import { useCallback, useEffect, useRef, useState } from "react";
import { Scanner, type ScannerStatus } from "../lib/scanner";
import type { DetectionResult, RecognitionMatch, ScanFrame } from "../types";

export interface ScannerState {
  active: boolean;
  starting: boolean;
  status: ScannerStatus | null;
  error: string | null;
  detection: DetectionResult | null;
  match: RecognitionMatch | null;
  lastFrame: ScanFrame | null;
}

export function useScanner() {
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const scannerRef = useRef<Scanner | null>(null);
  const [state, setState] = useState<ScannerState>({
    active: false,
    starting: false,
    status: null,
    error: null,
    detection: null,
    match: null,
    lastFrame: null,
  });

  const start = useCallback(async () => {
    if (!videoRef.current || scannerRef.current) return;
    setState((s) => ({ ...s, starting: true, error: null }));
    const scanner = new Scanner(videoRef.current, {
      onStatus: (status) => setState((s) => ({ ...s, status })),
      onDetection: (detection) => setState((s) => ({ ...s, detection })),
      onMatch: (match) => setState((s) => ({ ...s, match })),
      onFrame: (lastFrame) => setState((s) => ({ ...s, lastFrame })),
      onError: (error) => setState((s) => ({ ...s, error, starting: false })),
    });
    scannerRef.current = scanner;
    await scanner.start();
    setState((s) => ({ ...s, active: true, starting: false }));
  }, []);

  const stop = useCallback(() => {
    scannerRef.current?.stop();
    scannerRef.current = null;
    setState({
      active: false,
      starting: false,
      status: null,
      error: null,
      detection: null,
      match: null,
      lastFrame: null,
    });
  }, []);

  const saveCurrent = useCallback(async (name?: string) => {
    if (!scannerRef.current) return;
    await scannerRef.current.saveCurrent(name);
  }, []);

  useEffect(() => {
    return () => {
      scannerRef.current?.stop();
      scannerRef.current = null;
    };
  }, []);

  return { videoRef, state, start, stop, saveCurrent };
}
