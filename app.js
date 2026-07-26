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

const DESCRIPTIONS = {
  person: 'une personne (debout, assise ou en mouvement)',
  bicycle: 'un vélo',
  car: 'une voiture',
  motorcycle: 'une moto',
  airplane: 'un avion',
  bus: 'un bus',
  train: 'un train',
  truck: 'un camion',
  boat: 'un bateau',
  'traffic light': 'un feu tricolore',
  'fire hydrant': 'une bouche d\'incendie',
  'stop sign': 'un panneau stop',
  'parking meter': 'un parcmètre',
  bench: 'un banc',
  bird: 'un oiseau',
  cat: 'un chat',
  dog: 'un chien',
  horse: 'un cheval',
  sheep: 'un mouton',
  cow: 'une vache',
  elephant: 'un éléphant',
  bear: 'un ours',
  zebra: 'un zèbre',
  giraffe: 'une girafe',
  backpack: 'un sac à dos',
  umbrella: 'un parapluie',
  handbag: 'un sac à main',
  tie: 'une cravate',
  suitcase: 'une valise',
  frisbee: 'un frisbee',
  skis: 'des skis',
  snowboard: 'un snowboard',
  'sports ball': 'un ballon de sport',
  kite: 'un cerf-volant',
  'baseball bat': 'une batte de baseball',
  'baseball glove': 'un gant de baseball',
  skateboard: 'un skateboard',
  surfboard: 'un surfboard',
  'tennis racket': 'une raquette de tennis',
  bottle: 'une bouteille',
  'wine glass': 'un verre de vin',
  cup: 'une tasse',
  fork: 'une fourchette',
  knife: 'un couteau',
  spoon: 'une cuillère',
  bowl: 'un bol',
  banana: 'une banane',
  apple: 'une pomme',
  sandwich: 'un sandwich',
  orange: 'une orange',
  broccoli: 'du brocoli',
  carrot: 'une carotte',
  'hot dog': 'un hot-dog',
  pizza: 'une pizza',
  donut: 'un donut',
  cake: 'un gâteau',
  chair: 'une chaise',
  couch: 'un canapé',
  'potted plant': 'une plante en pot',
  bed: 'un lit',
  'dining table': 'une table à manger',
  toilet: 'des toilettes',
  tv: 'une télévision',
  laptop: 'un ordinateur portable',
  mouse: 'une souris',
  remote: 'une télécommande',
  keyboard: 'un clavier',
  'cell phone': 'un téléphone portable',
  microwave: 'un micro-ondes',
  oven: 'un four',
  toaster: 'un grille-pain',
  sink: 'un évier',
  refrigerator: 'un réfrigérateur',
  book: 'un livre',
  clock: 'une horloge',
  vase: 'un vase',
  scissors: 'des ciseaux',
  'teddy bear': 'un ours en peluche',
  'hair drier': 'un sèche-cheveux',
  toothbrush: 'une brosse à dents',
};

async function loadModel() {
  loading.style.display = 'flex';
  loading.innerHTML = '<div class="spinner"></div><span>Chargement du modèle IA (COCO-SSD)...</span>';
  try {
    model = await cocoSsd.load();
    loading.innerHTML = '<div class="spinner"></div><span>Modèle chargé ✓ Prêt à détecter</span>';
    setTimeout(() => { loading.style.display = 'none'; }, 600);
  } catch (err) {
    loading.innerHTML = `<span style="color:#f85149">Erreur : ${err.message}</span>`;
    console.error(err);
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
    alert('Accès caméra refusé. Autorise la caméra et réessaie.\n' + err.message);
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
    COLORS[cls] = `hsl(${hue}, 75%, 50%)`;
  }
  return COLORS[cls];
}

let lastTime = 0;
let frameCount = 0;
const MIN_CONFIDENCE = 0.4;

async function detectLoop() {
  if (!isRunning) return;

  const now = performance.now();
  frameCount++;
  if (now - lastTime >= 1000) {
    fpsCounter.textContent = `${frameCount} FPS`;
    frameCount = 0;
    lastTime = now;
  }

  if (model && video.readyState >= 2 && video.videoWidth > 0) {
    const w = video.videoWidth;
    const h = video.videoHeight;
    canvas.width = w;
    canvas.height = h;

    ctx.drawImage(video, 0, 0, w, h);

    let predictions;
    try {
      predictions = await model.detect(video);
    } catch (e) {
      predictions = [];
    }

    predictions = predictions.filter(p => p.score >= MIN_CONFIDENCE);

    const seen = {};

    for (const p of predictions) {
      const [x, y, bw, bh] = p.bbox;
      const cls = p.class || p.className || 'objet';
      const score = p.score;
      const color = getColor(cls);
      const label = `${cls} ${Math.round(score * 100)}%`;

      ctx.lineWidth = Math.max(2, Math.min(4, w / 200));
      ctx.strokeStyle = color;
      ctx.strokeRect(x, y, bw, bh);

      ctx.font = `bold ${Math.max(12, Math.min(16, w / 50))}px -apple-system, sans-serif`;
      const tw = ctx.measureText(label).width;
      const lh = 26;

      let labelY = y - lh - 2;
      if (labelY < 0) labelY = y + bh + 2;

      ctx.fillStyle = color;
      ctx.fillRect(x, labelY, tw + 12, lh);

      ctx.fillStyle = '#fff';
      ctx.fillText(label, x + 6, labelY + 18);

      const key = cls.toLowerCase();
      if (!seen[key]) {
        seen[key] = { cls, score: 0, count: 0, confs: [] };
      }
      seen[key].count++;
      seen[key].confs.push(score);
      if (score > seen[key].score) seen[key].score = score;
    }

    const items = Object.values(seen).sort((a, b) => b.score - a.score);

    if (items.length === 0) {
      objectList.innerHTML = '<li style="color:#8b949e">🔍 Aucun objet reconnu — approche un objet de la caméra</li>';
    } else {
      const total = items.reduce((s, i) => s + i.count, 0);
      const listHtml = items.map(i => {
        const avgConf = Math.round((i.confs.reduce((a, b) => a + b, 0) / i.confs.length) * 100);
        const desc = DESCRIPTIONS[i.cls.toLowerCase()] || `un(e) ${i.cls}`;
        return `<li><strong>${i.cls}</strong> ×${i.count} <span class="conf">${avgConf}%</span><br><span class="desc">→ ${desc}</span></li>`;
      }).join('');
      objectList.innerHTML = `<li style="color:#8b949e;width:100%">${total} objet(s) détecté(s) :</li>` + listHtml;
    }
  }

  animFrame = requestAnimationFrame(detectLoop);
}

btnStart.addEventListener('click', startCamera);
btnStop.addEventListener('click', stopCamera);

loadModel();
