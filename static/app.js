const video = document.getElementById('video');
const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');
const loading = document.getElementById('loading');
const loadingText = document.getElementById('loadingText');
const badge = document.getElementById('badge');
const micBadge = document.getElementById('micBadge');
const vlmBadge = document.getElementById('vlmBadge');
const trackBadge = document.getElementById('trackBadge');
const btnStart = document.getElementById('btnStart');
const btnStop = document.getElementById('btnStop');
const btnMic = document.getElementById('btnMic');
const fpsCounter = document.getElementById('fpsCounter');
const detCount = document.getElementById('detCount');
const latencyEl = document.getElementById('latency');
const totalBadge = document.getElementById('totalBadge');
const objectList = document.getElementById('objectList');
const sceneDesc = document.getElementById('sceneDesc');
const voiceStatus = document.getElementById('voiceStatus');
const voiceTranscript = document.getElementById('voiceTranscript');
const correctionList = document.getElementById('correctionList');
const corrCount = document.getElementById('corrCount');
const btnClearCorrections = document.getElementById('btnClearCorrections');
const btnVLM = document.getElementById('btnVLM');
const btnTrack = document.getElementById('btnTrack');
const btnFineTune = document.getElementById('btnFineTune');
const btnExportSession = document.getElementById('btnExportSession');
const btnExportCSV = document.getElementById('btnExportCSV');
const btnExportHTML = document.getElementById('btnExportHTML');
const statFrames = document.getElementById('statFrames');
const statDetTotal = document.getElementById('statDetTotal');
const statVLM = document.getElementById('statVLM');
const statLatency = document.getElementById('statLatency');

let stream = null;
let isRunning = false;
let frameCount = 0;
let lastFpsTime = performance.now();
let lastDetections = [];
let recognition = null;
let isListening = false;
let useVLM = true;
let useTracking = true;
let trailHistory = {};
let totalFramesProcessed = 0;
let totalDetectionsAll = 0;
let totalVLMCalls = 0;
let latencySum = 0;
let latencyCount = 0;
const MAX_TRAIL = 20;

const DETECT_URL = '/detect';
const SEND_INTERVAL = 100;
const STORAGE_KEY = 'objrec_corrections';

const CATEGORY_COLORS = {
  humain: '#58a6ff', 'véhicule': '#f0883e', animal: '#3fb950',
  'sport': '#d29922', meuble: '#a371f7', 'cuisine': '#f778ba',
  aliment: '#d29922', default: '#58a6ff',
};

const TRACK_COLORS = ['#58a6ff', '#3fb950', '#f0883e', '#d29922', '#a371f7', '#f778ba', '#79c0ff', '#56d364'];

function getColor(cat) { return CATEGORY_COLORS[cat] || CATEGORY_COLORS.default; }
function getTrackColor(id) { return TRACK_COLORS[id % TRACK_COLORS.length]; }

function loadCorrections() { try { return JSON.parse(localStorage.getItem(STORAGE_KEY)) || []; } catch { return []; } }
function saveCorrections(c) { localStorage.setItem(STORAGE_KEY, JSON.stringify(c)); }

async function syncCorrections() {
  try {
    const serverCorrs = await (await fetch('/corrections')).json();
    const localCorrs = loadCorrections();
    for (const lc of localCorrs) {
      await fetch('/corrections', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(lc) });
    }
    for (const sc of serverCorrs) {
      if (!localCorrs.find(c => c.from === sc.from && c.to === sc.to)) localCorrs.push(sc);
    }
    saveCorrections(localCorrs);
  } catch {}
}

function addCorrection(from, to, fromEn) {
  const corrs = loadCorrections();
  if (!corrs.find(c => c.from === from && c.to === to)) {
    const corr = { from, to, from_en: fromEn || from, timestamp: Date.now() };
    corrs.push(corr);
    saveCorrections(corrs);
    fetch('/corrections', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(corr) });
    renderCorrections();
  }
}

function removeCorrection(idx) {
  const corrs = loadCorrections();
  corrs.splice(idx, 1);
  saveCorrections(corrs);
  renderCorrections();
}

function applyCorrections(dets) {
  const corrs = loadCorrections();
  if (!corrs.length) return dets;
  return dets.map(d => {
    const c = corrs.find(x => x.from === d.class || x.from_en === d.class_en);
    return c ? { ...d, class: c.to, corrected: true } : d;
  });
}

function renderCorrections() {
  const corrs = loadCorrections();
  corrCount.textContent = corrs.length;
  if (!corrs.length) { correctionList.innerHTML = '<li class="empty-corr">Aucune correction</li>'; return; }
  correctionList.innerHTML = corrs.map((c, i) => `
    <li><span class="corr-from">${c.from}</span><span class="corr-arrow">→</span><span class="corr-to">${c.to}</span>
    <button class="corr-remove" onclick="removeCorrection(${i})">✕</button></li>`).join('');
}

function updateSessionStats() {
  statFrames.textContent = totalFramesProcessed;
  statDetTotal.textContent = totalDetectionsAll;
  statVLM.textContent = totalVLMCalls;
  statLatency.textContent = latencyCount > 0 ? Math.round(latencySum / latencyCount) + 'ms' : '—';
}

function parseVoiceCorrection(text) {
  const t = text.toLowerCase().trim();
  const patterns = [
    /(?:ce n'?est pas|c'?est pas)\s+(?:un[e]?\s+)?(.+?)(?:,\s*)?(?:c'?est|mais c'?est|en fait c'?est|plutôt)\s+(?:un[e]?\s+)?(.+)/i,
    /(.+?)\s+(?:c'?est|est)\s+(?:en fait |plutôt )?(?:un[e]?\s+)?(.+)/i,
    /(?:remplacer?|corriger?|changer?)\s+(.+?)\s+(?:par|en|pour)\s+(.+)/i,
  ];
  for (const pat of patterns) {
    const m = t.match(pat);
    if (m && m[1] && m[2]) {
      return {
        from: m[1].trim().replace(/^(un|une|le|la|l')\s+/i, '').replace(/^./, c => c.toUpperCase()),
        to: m[2].trim().replace(/^(un|une|le|la|l')\s+/i, '').replace(/^./, c => c.toUpperCase()),
      };
    }
  }
  return null;
}

function initSpeechRecognition() {
  const SR = window.SpeechRecognition || window.webkitSpeechRecognition;
  if (!SR) { voiceStatus.textContent = '❌ Chrome requis'; btnMic.disabled = true; return; }
  recognition = new SR();
  recognition.lang = 'fr-FR';
  recognition.continuous = false;
  recognition.interimResults = true;
  recognition.onstart = () => { isListening = true; micBadge.classList.remove('hidden'); voiceStatus.textContent = '🎤 Écoute…'; voiceStatus.className = 'voice-status listening'; };
  recognition.onresult = (e) => {
    let t = '';
    for (let i = e.resultIndex; i < e.results.length; i++) t += e.results[i][0].transcript;
    voiceTranscript.textContent = `"${t}"`;
    if (e.results[e.resultIndex].isFinal) processVoiceInput(t);
  };
  recognition.onerror = (e) => { isListening = false; micBadge.classList.add('hidden'); voiceStatus.className = 'voice-status'; voiceStatus.textContent = e.error === 'no-speech' ? '🎤 Réessayez' : `❌ ${e.error}`; };
  recognition.onend = () => { isListening = false; micBadge.classList.add('hidden'); voiceStatus.className = 'voice-status'; btnMic.textContent = '🎤 Micro'; };
}

function processVoiceInput(text) {
  const corr = parseVoiceCorrection(text);
  if (corr) {
    const matched = lastDetections.find(d => d.class.toLowerCase().includes(corr.from.toLowerCase()) || corr.from.toLowerCase().includes(d.class.toLowerCase()));
    addCorrection(corr.from, corr.to, matched ? matched.class_en : corr.from);
    voiceStatus.textContent = `✅ "${corr.from}" → "${corr.to}"`;
    voiceStatus.className = 'voice-status success';
    setTimeout(() => { voiceStatus.className = 'voice-status'; }, 3000);
  } else {
    voiceStatus.textContent = `⚠️ Pas compris: "${text}"`;
    voiceStatus.className = 'voice-status error';
    setTimeout(() => { voiceStatus.className = 'voice-status'; }, 3000);
  }
  voiceTranscript.textContent = '';
}

function toggleMic() {
  if (!recognition) return;
  if (isListening) { recognition.stop(); btnMic.textContent = '🎤 Micro'; }
  else { recognition.start(); btnMic.textContent = '⏹ Stop'; }
}

async function initCamera() {
  loading.style.display = 'flex';
  loadingText.textContent = 'Accès caméra…';
  try {
    stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: 'environment', width: { ideal: 640 }, height: { ideal: 480 } }, audio: false });
    video.srcObject = stream;
    await video.play();
    canvas.width = video.videoWidth || 640;
    canvas.height = video.videoHeight || 480;
    loading.style.display = 'none';
    badge.classList.remove('hidden');
    if (useVLM) vlmBadge.classList.remove('hidden');
    if (useTracking) trackBadge.classList.remove('hidden');
    btnStart.disabled = true;
    btnStop.disabled = false;
    btnMic.disabled = false;
    btnFineTune.disabled = false;
    btnExportSession.disabled = false;
    btnExportCSV.disabled = false;
    btnExportHTML.disabled = false;
    isRunning = true;
    await syncCorrections();
    loop();
  } catch (err) { loadingText.textContent = 'Caméra refusée.'; }
}

function stopCamera() {
  isRunning = false;
  if (isListening && recognition) recognition.stop();
  if (stream) { stream.getTracks().forEach(t => t.stop()); stream = null; }
  video.srcObject = null;
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  btnStart.disabled = false;
  btnStop.disabled = true;
  btnMic.disabled = true;
  btnFineTune.disabled = true;
  btnExportSession.disabled = true;
  btnExportCSV.disabled = true;
  btnExportHTML.disabled = true;
  badge.classList.add('hidden');
  micBadge.classList.add('hidden');
  vlmBadge.classList.add('hidden');
  trackBadge.classList.add('hidden');
  fpsCounter.textContent = '— FPS';
  detCount.textContent = '— objets';
  latencyEl.textContent = '— ms';
}

function captureFrame() {
  const w = video.videoWidth, h = video.videoHeight;
  if (w === 0 || h === 0) return null;
  canvas.width = w; canvas.height = h;
  ctx.drawImage(video, 0, 0, w, h);
  return canvas.toDataURL('image/jpeg', 0.7);
}

async function sendFrame() {
  const data = captureFrame();
  if (!data) return null;
  try {
    const t0 = performance.now();
    const resp = await fetch(DETECT_URL, {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ image: data, tracking: useTracking, vlm: useVLM }),
    });
    const elapsed = performance.now() - t0;
    latencyEl.textContent = Math.round(elapsed) + ' ms';
    if (!resp.ok) return null;
    const result = await resp.json();
    if (result.latency_ms) latencyEl.textContent = Math.round(result.latency_ms) + ' ms';
    return result;
  } catch { return null; }
}

function drawDetections(detections) {
  const w = video.videoWidth, h = video.videoHeight;
  canvas.width = w; canvas.height = h;
  ctx.drawImage(video, 0, 0, w, h);

  for (const d of detections) {
    const [x1, y1, x2, y2] = d.bbox;
    const color = d.track_id != null ? getTrackColor(d.track_id) : d.corrected ? '#3fb950' : getColor(d.category);

    if (d.track_id != null && useTracking) {
      if (!trailHistory[d.track_id]) trailHistory[d.track_id] = [];
      trailHistory[d.track_id].push({ x: (x1 + x2) / 2, y: (y1 + y2) / 2 });
      if (trailHistory[d.track_id].length > MAX_TRAIL) trailHistory[d.track_id].shift();
      const trail = trailHistory[d.track_id];
      ctx.strokeStyle = color; ctx.lineWidth = 2; ctx.globalAlpha = 0.4;
      ctx.beginPath();
      trail.forEach((p, i) => i === 0 ? ctx.moveTo(p.x, p.y) : ctx.lineTo(p.x, p.y));
      ctx.stroke(); ctx.globalAlpha = 1;
    }

    ctx.strokeStyle = color;
    ctx.lineWidth = Math.max(2, Math.min(3, w / 250));
    ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);

    const prefix = d.corrected ? '✓ ' : '';
    const trackLabel = d.track_id != null ? ` [${d.track_id}]` : '';
    const label = `${prefix}${d.class}${trackLabel} ${Math.round(d.confidence * 100)}%`;
    const fontSize = Math.max(11, Math.min(14, w / 50));
    ctx.font = `bold ${fontSize}px -apple-system, sans-serif`;
    const tw = ctx.measureText(label).width;
    let ly = y1 - 24; if (ly < 0) ly = y2 + 4;
    ctx.fillStyle = color; ctx.fillRect(x1, ly, tw + 12, 22);
    ctx.fillStyle = '#fff'; ctx.fillText(label, x1 + 6, ly + 16);

    if (d.vlm_description) {
      const vlmText = `💬 "${d.vlm_description}"`;
      const vFontSize = Math.max(10, Math.min(12, w / 55));
      ctx.font = `italic ${vFontSize}px -apple-system, sans-serif`;
      const vtw = ctx.measureText(vlmText).width;
      let vly = ly - 22; if (vly < 0) vly = y2 + 26;
      ctx.fillStyle = 'rgba(0,0,0,0.75)'; ctx.fillRect(x1, vly, vtw + 12, 20);
      ctx.fillStyle = '#d29922'; ctx.fillText(vlmText, x1 + 6, vly + 14);
    }
  }
}

function updateResults(detections) {
  totalBadge.textContent = detections.length;
  if (!detections.length) { sceneDesc.textContent = '🔍 Aucun objet'; objectList.innerHTML = ''; return; }

  const unique = [...new Set(detections.map(d => d.class))];
  sceneDesc.textContent = `Scène : ${unique.join(', ')}`;

  objectList.innerHTML = detections.map(d => {
    const color = d.track_id != null ? getTrackColor(d.track_id) : d.corrected ? '#3fb950' : getColor(d.category);
    const trackBadge = d.track_id != null ? `<span class="track-badge" style="background:${color}">#${d.track_id}</span>` : '';
    const corrBadge = d.corrected ? '<span class="corr-badge">✓</span>' : '';
    const vlmLine = d.vlm_description ? `<div class="obj-vlm">💬 ${d.vlm_description}</div>` : '';
    return `
      <li style="border-left: 3px solid ${color}">
        <div class="obj-header">
          <span class="obj-name" style="color:${color}">${d.class} ${corrBadge} ${trackBadge}</span>
          <span class="obj-conf">${Math.round(d.confidence * 100)}%</span>
        </div>
        <div class="obj-desc">${d.description}</div>
        ${vlmLine}
        <span class="obj-category">${d.category}</span>
      </li>`;
  }).join('');
}

async function loop() {
  if (!isRunning) return;
  const t0 = performance.now();
  const data = await sendFrame();
  if (data && data.detections) {
    let dets = applyCorrections(data.detections);
    lastDetections = dets;
    drawDetections(dets);
    updateResults(dets);
    detCount.textContent = `${dets.length} objets`;
    totalFramesProcessed = data.frame || totalFramesProcessed + 1;
    totalDetectionsAll += dets.length;
    if (data.vlm_active) totalVLMCalls++;
    latencySum += data.latency_ms || (performance.now() - t0);
    latencyCount++;
    updateSessionStats();
  }
  frameCount++;
  const now = performance.now();
  if (now - lastFpsTime >= 1000) { fpsCounter.textContent = `${frameCount} FPS`; frameCount = 0; lastFpsTime = now; }
  setTimeout(loop, Math.max(0, SEND_INTERVAL - (performance.now() - t0)));
}

function downloadFile(url, filename) {
  const a = document.createElement('a');
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
}

function fineTuneCapture() {
  const label = prompt('Nom de l\'objet à capturer :');
  if (!label) return;
  const data = captureFrame();
  if (!data) return;
  fetch('/fine-tune', {
    method: 'POST', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ label, image: data }),
  }).then(r => r.json()).then(r => {
    if (r.status === 'ok') {
      voiceStatus.textContent = `📸 "${label}" sauvegardé (${r.count} total)`;
      voiceStatus.className = 'voice-status success';
      setTimeout(() => { voiceStatus.className = 'voice-status'; }, 3000);
    }
  });
}

btnStart.addEventListener('click', initCamera);
btnStop.addEventListener('click', stopCamera);
btnMic.addEventListener('click', toggleMic);
btnFineTune.addEventListener('click', fineTuneCapture);
btnClearCorrections.addEventListener('click', () => { saveCorrections([]); renderCorrections(); fetch('/corrections', { method: 'DELETE' }); });
btnVLM.addEventListener('click', () => { useVLM = !useVLM; btnVLM.classList.toggle('active', useVLM); vlmBadge.classList.toggle('hidden', !useVLM); });
btnTrack.addEventListener('click', () => { useTracking = !useTracking; btnTrack.classList.toggle('active', useTracking); trackBadge.classList.toggle('hidden', !useTracking); });
btnExportSession.addEventListener('click', () => downloadFile('/export', `session_${new Date().toISOString().slice(0,19).replace(/:/g,'-')}.json`));
btnExportCSV.addEventListener('click', () => downloadFile('/export/csv', `detections_${new Date().toISOString().slice(0,19).replace(/:/g,'-')}.csv`));
btnExportHTML.addEventListener('click', () => downloadFile('/export/html', `rapport_${new Date().toISOString().slice(0,19).replace(/:/g,'-')}.html`));

initSpeechRecognition();
renderCorrections();
