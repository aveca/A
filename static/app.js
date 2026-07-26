const video = document.getElementById('video');
const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');
const loading = document.getElementById('loading');
const loadingText = document.getElementById('loadingText');
const badge = document.getElementById('badge');
const micBadge = document.getElementById('micBadge');
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

let stream = null;
let isRunning = false;
let frameCount = 0;
let lastFpsTime = performance.now();
let lastDetections = [];
let recognition = null;
let isListening = false;

const DETECT_URL = '/detect';
const SEND_INTERVAL = 80;
const STORAGE_KEY = 'objrec_corrections';

const CATEGORY_COLORS = {
  humain: '#58a6ff',
  'véhicule motorisé': '#f0883e',
  animal: '#3fb950',
  'objet de sport': '#d29922',
  meuble: '#a371f7',
  'ustensile de cuisine': '#f778ba',
  aliment: '#d29922',
  fruit: '#3fb950',
  légume: '#3fb950',
  default: '#58a6ff',
};

function getColor(category) {
  return CATEGORY_COLORS[category] || CATEGORY_COLORS.default;
}

function loadCorrections() {
  try {
    return JSON.parse(localStorage.getItem(STORAGE_KEY)) || [];
  } catch { return []; }
}

function saveCorrections(corrs) {
  localStorage.setItem(STORAGE_KEY, JSON.stringify(corrs));
}

function addCorrection(originalClass, correctedClass, originalEn) {
  const corrs = loadCorrections();
  const exists = corrs.find(c => c.from === originalClass && c.to === correctedClass);
  if (!exists) {
    corrs.push({
      from: originalClass,
      to: correctedClass,
      from_en: originalEn || originalClass,
      timestamp: Date.now(),
    });
    saveCorrections(corrs);
    renderCorrections();
  }
}

function removeCorrection(idx) {
  const corrs = loadCorrections();
  corrs.splice(idx, 1);
  saveCorrections(corrs);
  renderCorrections();
}

function applyCorrections(detections) {
  const corrs = loadCorrections();
  if (corrs.length === 0) return detections;
  return detections.map(d => {
    const corr = corrs.find(c => c.from === d.class || c.from_en === d.class_en);
    if (corr) {
      return { ...d, class: corr.to, corrected: true };
    }
    return d;
  });
}

function renderCorrections() {
  const corrs = loadCorrections();
  corrCount.textContent = corrs.length;
  if (corrs.length === 0) {
    correctionList.innerHTML = '<li class="empty-corr">Aucune correction</li>';
    return;
  }
  correctionList.innerHTML = corrs.map((c, i) => `
    <li>
      <span class="corr-from">${c.from}</span>
      <span class="corr-arrow">→</span>
      <span class="corr-to">${c.to}</span>
      <button class="corr-remove" onclick="removeCorrection(${i})">✕</button>
    </li>
  `).join('');
}

function parseVoiceCorrection(text) {
  const t = text.toLowerCase().trim();
  let from = null, to = null;

  const patterns = [
    /(?:ce n'?est pas|c'?est pas|pas c'?est)\s+(?:un[e]?\s+)?(.+?)(?:,\s*)?(?:c'?est|mais c'?est|en fait c'?est|plutôt|plutot)\s+(?:un[e]?\s+)?(.+)/i,
    /(.+?)\s+(?:c'?est|est)\s+(?:en fait |plutôt |plutot )?(?:un[e]?\s+)?(.+)/i,
    /(?:remplacer?|corriger?|changer?)\s+(.+?)\s+(?:par|en|pour)\s+(.+)/i,
    /(?:au lieu de|pas)\s+(.+?)\s+(?:mettre?|c'?est|utiliser?)\s+(.+)/i,
    /(.+?)\s+(?:est|sont)\s+(?:en fait |plutôt |en réalité )?(?:un[e]?\s+)?(.+)/i,
  ];

  for (const pat of patterns) {
    const m = t.match(pat);
    if (m && m[1] && m[2]) {
      from = m[1].trim().replace(/^(un|une|le|la|l'|les|des|du|un une)\s+/i, '');
      to = m[2].trim().replace(/^(un|une|le|la|l'|les|des|du|un une)\s+/i, '');
      break;
    }
  }

  if (!from || !to) {
    const simpleMatch = t.match(/^(.+?)\s+(?:c'?est|est)\s+(.+)$/i);
    if (simpleMatch) {
      from = simpleMatch[1].trim();
      to = simpleMatch[2].trim();
    }
  }

  if (from && to && from.length > 1 && to.length > 1) {
    from = from.charAt(0).toUpperCase() + from.slice(1);
    to = to.charAt(0).toUpperCase() + to.slice(1);
    return { from, to };
  }
  return null;
}

function initSpeechRecognition() {
  const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
  if (!SpeechRecognition) {
    voiceStatus.textContent = '❌ Reconnaissance vocale non supportée (utilise Chrome)';
    btnMic.disabled = true;
    return;
  }

  recognition = new SpeechRecognition();
  recognition.lang = 'fr-FR';
  recognition.continuous = false;
  recognition.interimResults = true;
  recognition.maxAlternatives = 1;

  recognition.onstart = () => {
    isListening = true;
    micBadge.classList.remove('hidden');
    voiceStatus.textContent = '🎤 Écoute en cours… dictez votre correction';
    voiceStatus.classList.add('listening');
  };

  recognition.onresult = (event) => {
    let transcript = '';
    for (let i = event.resultIndex; i < event.results.length; i++) {
      transcript += event.results[i][0].transcript;
    }
    voiceTranscript.textContent = `"${transcript}"`;

    if (event.results[event.resultIndex].isFinal) {
      processVoiceInput(transcript);
    }
  };

  recognition.onerror = (event) => {
    isListening = false;
    micBadge.classList.add('hidden');
    voiceStatus.classList.remove('listening');
    if (event.error === 'no-speech') {
      voiceStatus.textContent = '🎤 Aucun son détecté, réessayez';
    } else {
      voiceStatus.textContent = `❌ Erreur : ${event.error}`;
    }
  };

  recognition.onend = () => {
    isListening = false;
    micBadge.classList.add('hidden');
    voiceStatus.classList.remove('listening');
    btnMic.textContent = '🎤 Micro';
  };
}

function processVoiceInput(text) {
  const correction = parseVoiceCorrection(text);
  if (correction) {
    const matchedDet = lastDetections.find(d =>
      d.class.toLowerCase().includes(correction.from.toLowerCase()) ||
      correction.from.toLowerCase().includes(d.class.toLowerCase())
    );
    const originalEn = matchedDet ? matchedDet.class_en : correction.from;
    addCorrection(correction.from, correction.to, originalEn);
    voiceStatus.textContent = `✅ Correction : "${correction.from}" → "${correction.to}"`;
    voiceStatus.classList.add('success');
    setTimeout(() => voiceStatus.classList.remove('success'), 3000);
  } else {
    voiceStatus.textContent = `⚠️ Impossible de comprendre : "${text}"`;
    voiceStatus.classList.add('error');
    setTimeout(() => voiceStatus.classList.remove('error'), 3000);
  }
  voiceTranscript.textContent = '';
}

function toggleMic() {
  if (!recognition) return;
  if (isListening) {
    recognition.stop();
    btnMic.textContent = '🎤 Micro';
  } else {
    recognition.start();
    btnMic.textContent = '⏹ Arrêter écoute';
  }
}

async function initCamera() {
  loading.style.display = 'flex';
  loadingText.textContent = 'Accès à la caméra…';
  try {
    stream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: 'environment', width: { ideal: 640 }, height: { ideal: 480 } },
      audio: false,
    });
    video.srcObject = stream;
    await video.play();
    canvas.width = video.videoWidth || 640;
    canvas.height = video.videoHeight || 480;
    loading.style.display = 'none';
    badge.classList.remove('hidden');
    btnStart.disabled = true;
    btnStop.disabled = false;
    btnMic.disabled = false;
    isRunning = true;
    loop();
  } catch (err) {
    loadingText.textContent = 'Caméra refusée. Autorise et réessaie.';
    console.error(err);
  }
}

function stopCamera() {
  isRunning = false;
  if (isListening && recognition) recognition.stop();
  if (stream) {
    stream.getTracks().forEach(t => t.stop());
    stream = null;
  }
  video.srcObject = null;
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  btnStart.disabled = false;
  btnStop.disabled = true;
  btnMic.disabled = true;
  badge.classList.add('hidden');
  micBadge.classList.add('hidden');
  fpsCounter.textContent = '— FPS';
  detCount.textContent = '— objets';
  latencyEl.textContent = '— ms';
}

function captureFrame() {
  const w = video.videoWidth;
  const h = video.videoHeight;
  if (w === 0 || h === 0) return null;
  canvas.width = w;
  canvas.height = h;
  ctx.drawImage(video, 0, 0, w, h);
  return canvas.toDataURL('image/jpeg', 0.7);
}

async function sendFrame() {
  const data = captureFrame();
  if (!data) return null;
  try {
    const t0 = performance.now();
    const resp = await fetch(DETECT_URL, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ image: data }),
    });
    const elapsed = performance.now() - t0;
    latencyEl.textContent = Math.round(elapsed) + ' ms';
    if (!resp.ok) return null;
    return await resp.json();
  } catch (e) {
    return null;
  }
}

function drawDetections(detections) {
  const w = video.videoWidth;
  const h = video.videoHeight;
  canvas.width = w;
  canvas.height = h;
  ctx.drawImage(video, 0, 0, w, h);
  ctx.lineWidth = Math.max(2, Math.min(3, w / 250));
  ctx.font = `bold ${Math.max(11, Math.min(14, w / 50))}px -apple-system, sans-serif`;

  for (const d of detections) {
    const [x1, y1, x2, y2] = d.bbox;
    const color = d.corrected ? '#3fb950' : getColor(d.category);
    ctx.strokeStyle = color;
    ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);

    const prefix = d.corrected ? '✓ ' : '';
    const label = `${prefix}${d.class} ${Math.round(d.confidence * 100)}%`;
    const tw = ctx.measureText(label).width;
    let ly = y1 - 24;
    if (ly < 0) ly = y2 + 4;

    ctx.fillStyle = color;
    ctx.fillRect(x1, ly, tw + 12, 22);
    ctx.fillStyle = '#fff';
    ctx.fillText(label, x1 + 6, ly + 16);
  }
}

function updateResults(detections) {
  totalBadge.textContent = detections.length;

  if (detections.length === 0) {
    sceneDesc.textContent = '🔍 Aucun objet — approche un objet de la caméra';
    objectList.innerHTML = '';
    return;
  }

  const classNames = detections.map(d => d.class);
  const unique = [...new Set(classNames)];
  sceneDesc.textContent = `Scène contenant : ${unique.join(', ')}`;

  objectList.innerHTML = detections.map(d => {
    const color = d.corrected ? '#3fb950' : getColor(d.category);
    const corrBadge = d.corrected ? '<span class="corr-badge">✓ corrigé</span>' : '';
    return `
      <li style="border-left: 3px solid ${color}">
        <div class="obj-header">
          <span class="obj-name" style="color:${color}">${d.class} ${corrBadge}</span>
          <span class="obj-conf">${Math.round(d.confidence * 100)}%</span>
        </div>
        <div class="obj-desc">${d.description}</div>
        <span class="obj-category">${d.category}</span>
      </li>`;
  }).join('');
}

async function loop() {
  if (!isRunning) return;
  const t0 = performance.now();
  const data = await sendFrame();
  if (data && data.detections) {
    let dets = data.detections;
    dets = applyCorrections(dets);
    lastDetections = dets;
    drawDetections(dets);
    updateResults(dets);
    detCount.textContent = `${dets.length} objets`;
  }
  frameCount++;
  const now = performance.now();
  if (now - lastFpsTime >= 1000) {
    fpsCounter.textContent = `${frameCount} FPS`;
    frameCount = 0;
    lastFpsTime = now;
  }
  const elapsed = performance.now() - t0;
  const wait = Math.max(0, SEND_INTERVAL - elapsed);
  setTimeout(loop, wait);
}

btnStart.addEventListener('click', initCamera);
btnStop.addEventListener('click', stopCamera);
btnMic.addEventListener('click', toggleMic);
btnClearCorrections.addEventListener('click', () => {
  saveCorrections([]);
  renderCorrections();
});

initSpeechRecognition();
renderCorrections();
