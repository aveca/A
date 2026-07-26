import type { Product } from "../types";

export function exportJSON(products: Product[]): void {
  download(
    JSON.stringify(products, null, 2),
    `produits-${stamp()}.json`,
    "application/json"
  );
}

export function exportCSV(products: Product[]): void {
  const headers = ["nom", "code-barres", "prix", "devise", "categorie", "scans", "texte", "cree_le"];
  const rows = products.map((p) => [
    esc(p.name),
    esc(p.barcode || ""),
    p.price ?? "",
    esc(p.currency || ""),
    esc(p.category || ""),
    p.times_scanned,
    esc((p.detected_text || "").replace(/\s+/g, " ").slice(0, 200)),
    esc(p.created_at || ""),
  ]);
  const csv = [headers.join(","), ...rows.map((r) => r.join(","))].join("\n");
  download(csv, `produits-${stamp()}.csv`, "text/csv");
}

export function exportHTML(products: Product[]): void {
  const rows = products
    .map(
      (p) => `
    <tr>
      <td>${p.thumbnail ? `<img src="${p.thumbnail}" />` : ""}</td>
      <td>${escapeHtml(p.name)}</td>
      <td>${escapeHtml(p.barcode || "")}</td>
      <td>${p.price ?? ""} ${escapeHtml(p.currency || "")}</td>
      <td>${escapeHtml(p.category || "")}</td>
      <td>${p.times_scanned}</td>
      <td>${escapeHtml((p.detected_text || "").slice(0, 120))}</td>
    </tr>`
    )
    .join("");
  const html = `<!doctype html><html lang="fr"><head><meta charset="utf-8"><title>Produits scannés</title>
  <style>
    body{font-family:system-ui,sans-serif;margin:24px;color:#0b0f14}
    h1{font-size:20px}
    table{border-collapse:collapse;width:100%;margin-top:16px;font-size:14px}
    th,td{border:1px solid #e2e8f0;padding:8px;text-align:left;vertical-align:top}
    th{background:#f1f5f9}
    img{width:48px;height:48px;object-fit:cover;border-radius:6px}
  </style></head><body><h1>Produits scannés (${products.length})</h1>
  <table><thead><tr><th></th><th>Nom</th><th>Code-barres</th><th>Prix</th><th>Catégorie</th><th>Scans</th><th>Texte détecté</th></tr></thead>
  <tbody>${rows}</tbody></table></body></html>`;
  download(html, `produits-${stamp()}.html`, "text/html");
}

function esc(v: string | number): string {
  const s = String(v);
  if (/[",\n]/.test(s)) return `"${s.replace(/"/g, '""')}"`;
  return s;
}

function escapeHtml(s: string): string {
  return s.replace(/[&<>"']/g, (c) =>
    ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]!)
  );
}

function stamp(): string {
  return new Date().toISOString().slice(0, 19).replace(/[:T]/g, "-");
}

function download(content: string, filename: string, type: string): void {
  const blob = new Blob([content], { type });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  a.click();
  URL.revokeObjectURL(url);
}
