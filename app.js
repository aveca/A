const video = document.getElementById('video');
const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');
const loading = document.getElementById('loading');
const btnStart = document.getElementById('btnStart');
const btnStop = document.getElementById('btnStop');
const fpsCounter = document.getElementById('fpsCounter');
const objectList = document.getElementById('objectList');

let model = null;
let stream = null;
let animFrame = null;
let isRunning = false;

async function loadModel() {
  loading.style.display = 'flex';
  try {
    model = await cocoSsd.load({ base: 'mobilenet_v2' });
    loading.textContent = 'Modèle chargé ✓';
    setTimeout(() => { loading.style.display = 'none'; }, 800);
  } catch (err) {
    loading.textContent = 'Erreur chargement modèle : ' + err.message;
  }
}

async function startCamera() {
  try {
    stream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: 'environment', width: { ideal: 640 }, height: { ideal: 480 } },
      audio: false,
    });
    video.srcObject = stream;
    await video.play();

    canvas.width = video.videoWidth || 640;
    canvas.height = video.videoHeight || 480;

    btnStart.disabled = true;
    btnStop.disabled = false;
    isRunning = true;
    detectLoop();
  } catch (err) {
    alert('Accès caméra refusé : ' + err.message);
  }
}

function stopCamera() {
  isRunning = false;
  if (animFrame) { cancelAnimationFrame(animFrame); animFrame = null; }
  if (stream) {
    stream.getTracks().forEach(t => t.stop());
    stream = null;
  }
  video.srcObject = null;
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  btnStart.disabled = false;
  btnStop.disabled = true;
  fpsCounter.textContent = '—';
}

const COLORS = {};
function getColor(cls) {
  if (!COLORS[cls]) {
    const hue = (Object.keys(COLORS).length * 47) % 360;
    COLORS[cls] = `hsl(${hue}, 70%, 55%)`;
  }
  return COLORS[cls];
}

let lastTime = 0;
let frameCount = 0;

async function detectLoop() {
  if (!isRunning) return;

  const now = performance.now();
  frameCount++;

  if (now - lastTime >= 1000) {
    fpsCounter.textContent = `${frameCount} FPS`;
    frameCount = 0;
    lastTime = now;
  }

  if (model && video.readyState >= 2) {
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

    const predictions = await model.detect(video);

    const seen = {};
    const items = [];

    ctx.lineWidth = 3;
    ctx.font = 'bold 14px -apple-system, sans-serif';

    for (const p of predictions) {
      const { bbox, class: cls, score } = p;
      const [x, y, w, h] = bbox;
      const color = getColor(cls);

      ctx.strokeStyle = color;
      ctx.strokeRect(x, y, w, h);

      ctx.fillStyle = color;
      const label = `${cls} ${Math.round(score * 100)}%`;
      const tw = ctx.measureText(label).width;
      ctx.fillRect(x, y - 28, tw + 14, 28);

      ctx.fillStyle = '#000';
      ctx.fillText(label, x + 7, y - 8);

      const key = cls.toLowerCase();
      if (!seen[key]) {
        seen[key] = { cls, score: 0, count: 0 };
      }
      seen[key].count++;
      if (score > seen[key].score) seen[key].score = score;
    }

    for (const k of Object.keys(seen)) {
      items.push({ cls: seen[k].cls, count: seen[k].count, score: seen[k].score });
    }

    items.sort((a, b) => b.score - a.score);

    if (items.length === 0) {
      objectList.innerHTML = '<li>Aucun objet détecté</li>';
    } else {
      objectList.innerHTML = items.map(i =>
        `<li>${i.cls}${i.count > 1 ? ' ×' + i.count : ''} <span class="conf">${Math.round(i.score * 100)}%</span></li>`
      ).join('');
    }
  }

  animFrame = requestAnimationFrame(detectLoop);
}

btnStart.addEventListener('click', startCamera);
btnStop.addEventListener('click', stopCamera);

loadModel();
