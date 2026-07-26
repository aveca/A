import { useEffect, useMemo, useState } from "react";
import {
  ScanLine,
  Camera,
  CameraOff,
  Save,
  Trash2,
  Download,
  Type,
  Barcode,
  Tag,
  Package,
  Library,
  Check,
  Pencil,
} from "lucide-react";
import { useScanner } from "./hooks/useScanner";
import { listProducts, deleteProduct, renameProduct } from "./lib/productMemory";
import { exportJSON, exportCSV, exportHTML } from "./lib/export";
import type { Product } from "./types";

export default function App() {
  const { videoRef, state, start, stop, saveCurrent } = useScanner();
  const [products, setProducts] = useState<Product[]>([]);
  const [showOcr, setShowOcr] = useState(true);
  const [showBarcodes, setShowBarcodes] = useState(true);
  const [saving, setSaving] = useState(false);
  const [saveModal, setSaveModal] = useState(false);
  const [saveName, setSaveName] = useState("");
  const [renamingId, setRenamingId] = useState<string | null>(null);
  const [renameValue, setRenameValue] = useState("");
  const [toast, setToast] = useState<string | null>(null);

  const refresh = async () => {
    try {
      const list = await listProducts();
      setProducts(list);
    } catch (e) {
      console.warn("listProducts failed", e);
    }
  };

  useEffect(() => {
    refresh();
  }, []);

  useEffect(() => {
    if (!toast) return;
    const t = setTimeout(() => setToast(null), 2200);
    return () => clearTimeout(t);
  }, [toast]);

  const detection = state.detection;
  const match = state.match;

  const handleStart = async () => {
    await start();
  };
  const handleStop = () => stop();

  const openSave = () => {
    setSaveName(match?.product?.name || guessName(detection) || "");
    setSaveModal(true);
  };

  const confirmSave = async () => {
    setSaving(true);
    try {
      await saveCurrent(saveName);
      await refresh();
      setSaveModal(false);
      setToast("Produit enregistré dans la mémoire");
    } catch (e) {
      setToast("Erreur lors de l'enregistrement");
      console.warn(e);
    } finally {
      setSaving(false);
    }
  };

  const handleDelete = async (id: string) => {
    try {
      await deleteProduct(id);
      await refresh();
      setToast("Produit supprimé");
    } catch (e) {
      setToast("Erreur de suppression");
      console.warn(e);
    }
  };

  const startRename = (p: Product) => {
    setRenamingId(p.id!);
    setRenameValue(p.name);
  };
  const confirmRename = async () => {
    if (!renamingId || !renameValue.trim()) return;
    try {
      await renameProduct(renamingId, renameValue.trim());
      await refresh();
      setRenamingId(null);
      setToast("Produit renommé");
    } catch (e) {
      setToast("Erreur de renommage");
      console.warn(e);
    }
  };

  const stats = useMemo(() => {
    const totalScans = products.reduce((s, p) => s + (p.times_scanned || 0), 0);
    const categories = new Set(products.map((p) => p.category).filter(Boolean)).size;
    return { count: products.length, totalScans, categories };
  }, [products]);

  return (
    <div className="app">
      <header className="header">
        <div className="brand">
          <div className="brand-logo">
            <ScanLine size={20} />
          </div>
          <div>
            <div className="brand-title">ScanGo</div>
            <div className="brand-sub">Reconnaissance produit autonome</div>
          </div>
        </div>
        <div className="header-actions">
          {products.length > 0 && (
            <>
              <button className="btn btn-sm" onClick={() => exportJSON(products)}>
                <Download size={14} /> JSON
              </button>
              <button className="btn btn-sm" onClick={() => exportCSV(products)}>
                <Download size={14} /> CSV
              </button>
              <button className="btn btn-sm" onClick={() => exportHTML(products)}>
                <Download size={14} /> HTML
              </button>
            </>
          )}
        </div>
      </header>

      <div className="scanner-wrap">
        <video ref={videoRef} className="scanner-video" playsInline muted />
        {!state.active && !state.starting && (
          <div className="scanner-idle">
            <div className="ico">
              <Camera size={28} />
            </div>
            <div style={{ fontWeight: 600, marginBottom: 4 }}>
              Caméra prête à démarrer
            </div>
            <div style={{ fontSize: 13 }}>
              Lance la caméra et montre un produit. Le système lit le code-barres,
              le texte sur l'emballage et le prix automatiquement.
            </div>
          </div>
        )}
        {state.starting && (
          <div className="scanner-idle">
            <div className="ico">
              <Camera size={28} />
            </div>
            <div>Démarrage de la caméra…</div>
          </div>
        )}
        {state.active && (
          <>
            <div className="scan-overlay">
              <div className="scan-frame" />
              <div className="scan-laser" />
            </div>
            <div className="status-pill">
              <span className={`pill ${state.active ? "live" : ""}`}>
                <span className="dot" /> LIVE · {state.status?.fps ?? 0} fps
              </span>
              {state.status?.ocrRunning && (
                <span className="pill working">
                  <span className="dot" /> OCR
                </span>
              )}
              {detection && detection.barcodes.length > 0 && (
                <span className="pill working">
                  <span className="dot" /> CODE {detection.barcodes[0].format}
                </span>
              )}
            </div>
            {match && match.product && (
              <div className="match-banner">
                {match.product.thumbnail && (
                  <img className="match-thumb" src={match.product.thumbnail} alt="" />
                )}
                <div className="match-info">
                  <div className="match-name">{match.product.name}</div>
                  <div className="match-meta">
                    {match.product.price != null && (
                      <span>
                        {match.product.price} {match.product.currency || ""}
                      </span>
                    )}
                    {match.product.barcode && <span>{match.product.barcode}</span>}
                    {match.product.category && <span>{match.product.category}</span>}
                  </div>
                </div>
                <div className="match-method">
                  {match.method === "barcode" && "Code-barres"}
                  {match.method === "ocr" && "Texte"}
                  {match.method === "shape" && "Visuel"}
                  {match.method === "memory" && "Mémoire"}
                  <div style={{ fontSize: 10, color: "var(--text-muted)", fontWeight: 400 }}>
                    {Math.round(match.confidence * 100)}%
                  </div>
                </div>
              </div>
            )}
          </>
        )}
      </div>

      {state.error && (
        <div style={{ color: "var(--error)", fontSize: 13, padding: "8px 4px" }}>
          {state.error}
        </div>
      )}

      <div className="controls">
        {!state.active ? (
          <button className="btn btn-primary" onClick={handleStart} disabled={state.starting}>
            <Camera size={16} /> Démarrer la caméra
          </button>
        ) : (
          <button className="btn btn-danger" onClick={handleStop}>
            <CameraOff size={16} /> Arrêter
          </button>
        )}
        <button className="btn btn-primary" onClick={openSave} disabled={!state.active}>
          <Save size={16} /> Enregistrer le produit
        </button>
        <div className="toggle-group" style={{ marginLeft: "auto" }}>
          <button
            className={`toggle ${showOcr ? "active" : ""}`}
            onClick={() => setShowOcr((v) => !v)}
          >
            <Type size={14} /> Texte <span className="switch" />
          </button>
          <button
            className={`toggle ${showBarcodes ? "active" : ""}`}
            onClick={() => setShowBarcodes((v) => !v)}
          >
            <Barcode size={14} /> Codes <span className="switch" />
          </button>
        </div>
      </div>

      <div className="grid">
        <div className="panel">
          <div className="panel-head">
            <div className="panel-title">
              <Type className="ico" size={16} /> Texte détecté
            </div>
            <span className="panel-count">
              {detection?.textLines?.length ?? 0} lignes
            </span>
          </div>
          <div className="list">
            {showOcr && detection && detection.textLines.length > 0 ? (
              detection.textLines.map((l, i) => (
                <div className="list-item" key={i}>
                  <span className="text">{l.text}</span>
                  <span className="conf">{l.confidence}%</span>
                </div>
              ))
            ) : (
              <div className="empty">
                {state.active ? "Aucun texte détecté pour l'instant" : "Caméra arrêtée"}
              </div>
            )}
          </div>
        </div>

        <div className="panel">
          <div className="panel-head">
            <div className="panel-title">
              <Barcode className="ico" size={16} /> Codes-barres
            </div>
            <span className="panel-count">
              {detection?.barcodes?.length ?? 0} trouvés
            </span>
          </div>
          <div className="list">
            {showBarcodes && detection && detection.barcodes.length > 0 ? (
              detection.barcodes.map((b, i) => (
                <div className="list-item" key={i}>
                  <span className="text">{b.text}</span>
                  <span className="tag">{b.format}</span>
                </div>
              ))
            ) : (
              <div className="empty">
                {state.active ? "Aucun code-barres visible" : "Caméra arrêtée"}
              </div>
            )}
          </div>
        </div>

        <div className="panel">
          <div className="panel-head">
            <div className="panel-title">
              <Tag className="ico" size={16} /> Prix extraits
            </div>
            <span className="panel-count">
              {detection?.prices?.length ?? 0} trouvés
            </span>
          </div>
          <div className="list">
            {detection && detection.prices.length > 0 ? (
              detection.prices.map((p, i) => (
                <div className="list-item" key={i}>
                  <span className="text">{p.raw}</span>
                  <span className="tag price">
                    {p.amount} {p.currency}
                  </span>
                </div>
              ))
            ) : (
              <div className="empty">
                {state.active ? "Aucun prix détecté" : "Caméra arrêtée"}
              </div>
            )}
          </div>
        </div>

        <div className="panel">
          <div className="panel-head">
            <div className="panel-title">
              <Package className="ico" size={16} /> Analyse visuelle
            </div>
          </div>
          <div className="list">
            {detection && (
              <>
                <div className="list-item">
                  <span className="text">Forme dominante</span>
                  <span className="tag">{detection.dominantShape}</span>
                </div>
                {detection.colorFingerprint && (
                  <div className="list-item">
                    <span className="text">Couleur moyenne (RGB)</span>
                    <span className="conf">
                      {detection.colorFingerprint.avg.join(", ")}
                    </span>
                  </div>
                )}
              </>
            )}
            {!detection && <div className="empty">Caméra arrêtée</div>}
          </div>
        </div>
      </div>

      <div className="panel">
        <div className="panel-head">
          <div className="panel-title">
            <Library size={16} /> Bibliothèque produits
          </div>
          <span className="panel-count">
            {stats.count} produits · {stats.totalScans} scans · {stats.categories} catégories
          </span>
        </div>
        {products.length === 0 ? (
          <div className="empty">
            Aucun produit enregistré. Scanne et enregistre un produit pour qu'il soit
            reconnu automatiquement la prochaine fois.
          </div>
        ) : (
          <div className="library">
            {products.map((p) => (
              <div className="product-card" key={p.id}>
                {p.thumbnail ? (
                  <img src={p.thumbnail} alt={p.name} />
                ) : (
                  <div
                    style={{
                      height: 110,
                      background: "var(--bg)",
                      display: "grid",
                      placeItems: "center",
                      color: "var(--text-dim)",
                    }}
                  >
                    <Package size={28} />
                  </div>
                )}
                <div className="body">
                  {renamingId === p.id ? (
                    <>
                      <input
                        autoFocus
                        value={renameValue}
                        onChange={(e) => setRenameValue(e.target.value)}
                        style={{
                          width: "100%",
                          padding: "4px 6px",
                          borderRadius: 4,
                          border: "1px solid var(--border)",
                          background: "var(--bg)",
                          color: "var(--text)",
                          fontSize: 13,
                        }}
                      />
                      <div style={{ display: "flex", gap: 6, marginTop: 4 }}>
                        <button className="btn btn-sm btn-primary" onClick={confirmRename}>
                          <Check size={12} /> OK
                        </button>
                        <button
                          className="btn btn-sm"
                          onClick={() => setRenamingId(null)}
                        >
                          Annuler
                        </button>
                      </div>
                    </>
                  ) : (
                    <>
                      <div className="name">{p.name}</div>
                      <div className="meta">
                        {p.price != null && (
                          <span>
                            {p.price} {p.currency || ""}
                          </span>
                        )}
                        {p.category && p.category !== "unknown" && <span>{p.category}</span>}
                      </div>
                      <div className="footer">
                        <span className="scans">scanné {p.times_scanned}×</span>
                        <div style={{ display: "flex", gap: 4 }}>
                          <button
                            className="btn btn-sm btn-ghost"
                            onClick={() => startRename(p)}
                            title="Renommer"
                          >
                            <Pencil size={12} />
                          </button>
                          <button
                            className="btn btn-sm btn-ghost btn-danger"
                            onClick={() => handleDelete(p.id!)}
                            title="Supprimer"
                          >
                            <Trash2 size={12} />
                          </button>
                        </div>
                      </div>
                    </>
                  )}
                </div>
              </div>
            ))}
          </div>
        )}
      </div>

      <div className="footer">
        ScanGo · fonctionne dans le navigateur, hors-ligne après chargement ·
        mémoire des produits dans ta base de données
      </div>

      {saveModal && (
        <div className="modal-backdrop" onClick={() => setSaveModal(false)}>
          <div className="modal" onClick={(e) => e.stopPropagation()}>
            <h3>Enregistrer le produit</h3>
            <p>
              Le système va capturer l'image actuelle, lire le code-barres et le texte,
              et sauvegarder le produit. Il sera reconnu automatiquement la prochaine fois.
            </p>
            <input
              autoFocus
              placeholder="Nom du produit (ex: Lu Petit Beurre)"
              value={saveName}
              onChange={(e) => setSaveName(e.target.value)}
              onKeyDown={(e) => e.key === "Enter" && confirmSave()}
            />
            <div className="modal-actions">
              <button className="btn" onClick={() => setSaveModal(false)}>
                Annuler
              </button>
              <button
                className="btn btn-primary"
                onClick={confirmSave}
                disabled={saving || !saveName.trim()}
              >
                {saving ? "Enregistrement…" : "Enregistrer"}
              </button>
            </div>
          </div>
        </div>
      )}

      {toast && (
        <div
          style={{
            position: "fixed",
            bottom: 24,
            left: "50%",
            transform: "translateX(-50%)",
            background: "var(--primary)",
            color: "white",
            padding: "10px 18px",
            borderRadius: 999,
            fontSize: 13,
            fontWeight: 600,
            boxShadow: "var(--shadow)",
            zIndex: 100,
          }}
        >
          {toast}
        </div>
      )}
    </div>
  );
}

function guessName(detection: { textLines: { text: string }[] } | null): string {
  if (!detection) return "";
  const lines = [...detection.textLines].sort((a, b) => b.text.length - a.text.length);
  for (const l of lines) {
    const clean = l.text.replace(/[^a-zA-ZÀ-ÿ0-9\s'-]/g, " ").replace(/\s+/g, " ").trim();
    const words = clean.split(/\s+/).filter((w) => w.length >= 3);
    if (words.length >= 2 && clean.length <= 40) {
      return words.map((w) => w[0].toUpperCase() + w.slice(1).toLowerCase()).join(" ");
    }
  }
  return "";
}
