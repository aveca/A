const video = document.getElementById('video');
const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');
const loading = document.getElementById('loading');
const loadingText = document.getElementById('loadingText');
const badge = document.getElementById('badge');
const btnStart = document.getElementById('btnStart');
const btnStop = document.getElementById('btnStop');
const fpsCounter = document.getElementById('fpsCounter');
const detCount = document.getElementById('detCount');
const latencyEl = document.getElementById('latency');
const totalBadge = document.getElementById('totalBadge');
const objectList = document.getElementById('objectList');
const sceneDesc = document.getElementById('sceneDesc');

let stream = null;
let isRunning = false;
let frameCount = 0;
let lastFpsTime = performance.now();
let lastDetections = [];

const DETECT_URL = '/detect';
const SEND_INTERVAL = 80;

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
    isRunning = true;
    loop();
  } catch (err) {
    loadingText.textContent = 'Caméra refusée. Autorise et réessaie.';
    console.error(err);
  }
}

function stopCamera() {
  isRunning = false;
  if (stream) {
    stream.getTracks().forEach(t => t.stop());
    stream = null;
  }
  video.srcObject = null;
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  btnStart.disabled = false;
  btnStop.disabled = true;
  badge.classList.add('hidden');
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
    const color = getColor(d.category);
    ctx.strokeStyle = color;
    ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);

    const label = `${d.class} ${Math.round(d.confidence * 100)}%`;
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
    const color = getColor(d.category);
    return `
      <li style="border-left: 3px solid ${color}">
        <div class="obj-header">
          <span class="obj-name" style="color:${color}">${d.class}</span>
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
    lastDetections = data.detections;
    drawDetections(data.detections);
    updateResults(data.detections);
    detCount.textContent = `${data.detections.length} objets`;
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
