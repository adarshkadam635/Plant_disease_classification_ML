'use strict';
const $ = (id) => document.getElementById(id);
const paths = {
  leaf: '<path d="M20 4c-8-2-16 2-15 9 1 7 9 8 13 3 3-4 2-8 2-12Z"/><path d="M3 21 16 8m-6 6 6 1m-6-1-1-5"/>',
  grid: '<rect x="3" y="3" width="7" height="7" rx="1.5"/><rect x="14" y="3" width="7" height="7" rx="1.5"/><rect x="3" y="14" width="7" height="7" rx="1.5"/><rect x="14" y="14" width="7" height="7" rx="1.5"/>',
  history: '<path d="M3 10a9 9 0 1 1 1 7M3 4v6h6m3-3v6l4 2"/>',
  book: '<path d="M12 5C8 2 4 3 2 4v15c3-1 7-1 10 2 3-3 7-3 10-2V4c-3-1-7-1-10 1Zm0 0v16"/>',
  help: '<circle cx="12" cy="12" r="9"/><path d="M9 9a3 3 0 1 1 5 2c-2 1-2 2-2 3m0 3h.01"/>',
  sprout: '<path d="M12 22V12M12 15C3 16 2 9 3 6c7-1 10 3 9 9Zm0-3c-1-7 3-11 9-10 1 6-2 11-9 10Z"/>',
  shield: '<path d="m12 2 8 4v6c0 5-8 10-8 10S4 17 4 12V6l8-4Z"/><path d="m8 12 3 3 5-6"/>',
  scan: '<path d="M8 3H3v5m13-5h5v5M3 16v5h5m13-5v5h-5M7 12h10"/>',
  arrow: '<path d="M4 12h16m-6-6 6 6-6 6"/>',
  layers: '<path d="m12 3 10 5-10 5L2 8l10-5Zm-9 9 9 5 9-5M3 16l9 5 9-5"/>',
  upload: '<path d="M12 16V3m-5 5 5-5 5 5M4 15v5h16v-5"/>',
  camera: '<path d="M8 5 6 8H3v12h18V8h-3l-2-3H8Z"/><circle cx="12" cy="13" r="4"/>',
  image: '<rect x="3" y="3" width="18" height="18" rx="3"/><circle cx="8" cy="8" r="1"/><path d="m3 17 5-5 4 4 4-6 5 6"/>',
  sparkles: '<path d="m12 3 2.5 6.5L21 12l-6.5 2.5L12 21l-2.5-6.5L3 12l6.5-2.5L12 3Zm7-1v4m-2-2h4"/>',
  sun: '<circle cx="12" cy="12" r="4"/><path d="M12 2v2m0 16v2M2 12h2m16 0h2M5 5l1 1m12 12 1 1M5 19l1-1M18 6l1-1"/>',
};
function icon(name) { return `<svg class="icon" viewBox="0 0 24 24" aria-hidden="true">${paths[name] || paths.leaf}</svg>`; }
document.querySelectorAll('[data-icon]').forEach((el) => { el.innerHTML = icon(el.dataset.icon); });
function escapeHtml(value) { return String(value).replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c])); }
let selectedFile = null, previewUrl = null, stream = null, busy = false, cameraSession = 0, currentReport = null;
let history = [];
try {
  const saved = JSON.parse(localStorage.getItem('leaflens-history') || '[]');
  if (Array.isArray(saved)) history = saved.filter(item => item && typeof item.plant === 'string' && typeof item.disease === 'string' && typeof item.filename === 'string' && Number.isFinite(item.confidence) && !isNaN(Date.parse(item.date))).slice(0, 50);
} catch { /* Storage may be unavailable in private browsers. */ }
const initialResult = $('result-content').innerHTML;
$('today').textContent = new Intl.DateTimeFormat(undefined, {day:'numeric', month:'short', year:'numeric'}).format(new Date());
function error(message = '') { $('scan-error').textContent = message; $('scan-error').hidden = !message; }
function resetResults() {
  currentReport = null;
  $('result-content').className = 'result-empty';
  $('result-content').innerHTML = initialResult;
  $('result-badge').textContent = 'WAITING FOR IMAGE';
}
function stopCamera() {
  cameraSession++;
  if (stream) stream.getTracks().forEach(track => track.stop());
  stream = null;
  $('camera').srcObject = null;
  $('capture').disabled = true;
  $('camera-placeholder').hidden = false;
  $('start-camera').disabled = false;
}
function setMode(mode) {
  if (busy) return;
  stopCamera();
  error();
  const camera = mode === 'camera';
  $('upload-tab').classList.toggle('selected', !camera);
  $('upload-tab').setAttribute('aria-pressed', String(!camera));
  $('camera-tab').classList.toggle('selected', camera);
  $('camera-tab').setAttribute('aria-pressed', String(camera));
  $('camera-panel').hidden = !camera;
  $('upload-panel').hidden = camera || !!selectedFile;
  $('preview-panel').hidden = camera || !selectedFile;
}
async function selectFile(file) {
  if (!file || busy) return;
  error();
  if (!['image/jpeg', 'image/png', 'image/webp'].includes(file.type)) return error('Choose a JPG, PNG, or WebP photo. HEIC photos must be converted first.');
  if (file.size > 10 * 1024 * 1024) return error('This photo is too large. Choose an image under 10 MB.');
  if (!file.size) return error('This image is empty. Please choose another photo.');
  const url = URL.createObjectURL(file);
  const img = new Image();
  try {
    img.src = url;
    await img.decode();
    if (img.naturalWidth < 32 || img.naturalHeight < 32 || img.naturalWidth * img.naturalHeight > 20000000) throw new Error('size');
  } catch {
    URL.revokeObjectURL(url);
    return error('Could not read this photo. Choose a valid image between 32 pixels per side and 20 megapixels.');
  }
  if (previewUrl) URL.revokeObjectURL(previewUrl);
  previewUrl = url;
  selectedFile = file;
  $('preview').src = url;
  $('file-name').textContent = file.name;
  $('file-size').textContent = file.size < 1024 * 1024 ? `${Math.round(file.size / 1024)} KB` : `${(file.size / 1024 / 1024).toFixed(1)} MB`;
  $('analyze').disabled = false;
  resetResults();
  setMode('upload');
}
$('upload-tab').onclick = () => setMode('upload');
$('camera-tab').onclick = () => setMode('camera');
$('dropzone').onclick = () => $('file-input').click();
for (const id of ['file-input', 'capture-input']) $(id).onchange = event => { selectFile(event.target.files[0]); event.target.value = ''; };
for (const type of ['dragenter', 'dragover']) $('dropzone').addEventListener(type, event => { event.preventDefault(); $('dropzone').classList.add('dragging'); });
for (const type of ['dragleave', 'drop']) $('dropzone').addEventListener(type, event => { event.preventDefault(); $('dropzone').classList.remove('dragging'); });
$('dropzone').addEventListener('drop', event => { if (event.dataTransfer.files.length > 1) error('Choose one leaf photo at a time.'); else selectFile(event.dataTransfer.files[0]); });
$('remove-image').onclick = () => {
  if (busy) return;
  selectedFile = null;
  if (previewUrl) URL.revokeObjectURL(previewUrl);
  previewUrl = null;
  $('preview').removeAttribute('src');
  $('analyze').disabled = true;
  resetResults();
  setMode('upload');
};
$('hero-scan').onclick = () => { $('dropzone').scrollIntoView({block:'center', behavior:matchMedia('(prefers-reduced-motion: reduce)').matches ? 'instant' : 'smooth'}); if (selectedFile) $('analyze').focus({preventScroll:true}); else $('dropzone').focus({preventScroll:true}); };
$('native-camera').onclick = () => { stopCamera(); $('capture-input').click(); };
$('start-camera').onclick = async () => {
  error();
  if (!navigator.mediaDevices?.getUserMedia || !window.isSecureContext) return error('Live camera requires HTTPS or localhost. Use the device camera option or upload an image.');
  const session = ++cameraSession;
  $('start-camera').disabled = true;
  try {
    const requestedStream = await navigator.mediaDevices.getUserMedia({video:{facingMode:{ideal:'environment'}, width:{ideal:1280}, height:{ideal:960}}, audio:false});
    if (session !== cameraSession) { requestedStream.getTracks().forEach(track => track.stop()); return; }
    stream = requestedStream;
    $('camera').srcObject = stream;
    await $('camera').play();
    if (session !== cameraSession) return;
    $('camera-placeholder').hidden = true;
    $('capture').disabled = false;
  } catch (err) {
    if (session !== cameraSession) return;
    stopCamera();
    error(err.name === 'NotAllowedError' ? 'Camera access was denied. Allow camera access in your browser, or upload a photo.' : 'No available camera could be opened. Try the device camera option or upload a photo.');
  }
};
$('capture').onclick = () => {
  const video = $('camera');
  if (!stream || !video.videoWidth) return error('The camera is still starting. Please wait a moment.');
  const canvas = document.createElement('canvas');
  canvas.width = video.videoWidth; canvas.height = video.videoHeight;
  canvas.getContext('2d').drawImage(video, 0, 0);
  $('capture').disabled = true;
  canvas.toBlob(blob => {
    if (!blob) { $('capture').disabled = false; return error('Could not capture the photo. Try again.'); }
    selectFile(new File([blob], `leaf-${Date.now()}.jpg`, {type:'image/jpeg'}));
  }, 'image/jpeg', 0.92);
};
window.addEventListener('pagehide', stopCamera);
document.addEventListener('visibilitychange', () => { if (document.hidden) stopCamera(); });
function renderReport(report) {
  const first = report.predictions[0];
  $('result-badge').textContent = first.confidence < 70 ? 'LOW CONFIDENCE' : 'ANALYSIS COMPLETE';
  $('result-content').className = '';
  $('result-content').innerHTML = `<div class="prediction-top"><small>${escapeHtml(first.plant)}</small><h3>${escapeHtml(first.disease)}</h3><p>${first.healthy ? 'Classified as healthy' : 'Possible condition detected'}</p></div><div class="confidence-heading"><span>Model confidence</span><strong>${first.confidence.toFixed(1)}%</strong></div><div class="confidence-bar"><progress max="100" value="${first.confidence}" aria-label="Model confidence"></progress></div>${report.predictions.slice(1).map(p => `<div class="alternative"><span>${escapeHtml(p.plant)} · ${escapeHtml(p.disease)}</span><strong>${p.confidence.toFixed(1)}%</strong></div>`).join('')}<p class="result-notice">${first.confidence < 70 ? 'The model is uncertain. Try a clearer photo. ' : ''}${escapeHtml(report.notice)}<br>ResNet-9 · ${(report.elapsed_ms / 1000).toFixed(2)} s</p><button id="download-report" class="button secondary download-result">Download result</button>`;
  $('download-report').onclick = () => {
    const url = URL.createObjectURL(new Blob([JSON.stringify(currentReport, null, 2)], {type:'application/json'}));
    const link = document.createElement('a'); link.href = url; link.download = 'leaflens-result.json'; link.click(); setTimeout(() => URL.revokeObjectURL(url), 1000);
  };
}
async function prepareUpload(file) {
  // Vercel accepts request bodies up to 4.5 MB; keep a margin below that limit.
  if (file.size <= 4 * 1024 * 1024) return file;
  const url = URL.createObjectURL(file);
  try {
    const image = new Image();
    image.src = url;
    await image.decode();
    const scale = Math.min(1, 1600 / Math.max(image.naturalWidth, image.naturalHeight));
    const canvas = document.createElement('canvas');
    canvas.width = Math.round(image.naturalWidth * scale);
    canvas.height = Math.round(image.naturalHeight * scale);
    const context = canvas.getContext('2d');
    context.fillStyle = '#ffffff';
    context.fillRect(0, 0, canvas.width, canvas.height);
    context.drawImage(image, 0, 0, canvas.width, canvas.height);
    const blob = await new Promise(resolve => canvas.toBlob(resolve, 'image/jpeg', 0.85));
    if (!blob || blob.size > 4 * 1024 * 1024) throw new Error('Please choose a smaller photo. This image could not be optimized for upload.');
    return blob;
  } finally {
    URL.revokeObjectURL(url);
  }
}
$('analyze').onclick = async () => {
  if (!selectedFile || busy) return;
  stopCamera(); error(); busy = true;
  ['analyze','upload-tab','camera-tab','remove-image'].forEach(id => $(id).disabled = true);
  $('analyze-label').textContent = 'Analyzing your leaf…';
  $('results').setAttribute('aria-busy', 'true');
  $('result-badge').textContent = 'ANALYZING';
  $('result-content').className = 'loading-state';
  $('result-content').innerHTML = '<span class="spinner" aria-hidden="true"></span><h3>A closer look…</h3><p>Checking your leaf with ResNet-9.</p>';
  try {
    const upload = await prepareUpload(selectedFile);
    const response = await fetch('/api/predict', {method:'POST', headers:{'Content-Type':upload.type}, body:upload, signal:AbortSignal.timeout(60000)});
    const report = await response.json();
    if (!response.ok) throw new Error(report.detail || 'Analysis failed. Please try again.');
    currentReport = {...report, filename:selectedFile.name, date:new Date().toISOString()};
    renderReport(currentReport);
    history.unshift({...report.predictions[0], filename:selectedFile.name, date:currentReport.date});
    history = history.slice(0,50);
    try { localStorage.setItem('leaflens-history', JSON.stringify(history)); } catch { error('Analysis complete, but this browser could not save history. Download the result to keep it.'); }
    renderHistory();
  } catch (err) {
    resetResults();
    error(err.name === 'TimeoutError' ? 'Analysis took too long. Try again with a smaller photo.' : err instanceof TypeError ? 'Could not reach the server. Check your connection and try again.' : err.message);
  } finally {
    busy = false;
    ['analyze','upload-tab','camera-tab','remove-image'].forEach(id => $(id).disabled = false);
    $('analyze-label').textContent = 'Analyze leaf';
    $('results').setAttribute('aria-busy', 'false');
  }
};
function renderHistory() {
  $('history-count').textContent = history.length;
  $('scan-total').textContent = history.length;
  const empty = '<div class="empty-history"><strong>A fresh start for your plants.</strong>Your completed scans will appear here.</div>';
  function rows(items) { return items.map(item => `<article class="history-row"><span>${icon('leaf')}</span><div class="history-name"><strong>${escapeHtml(item.plant)} · ${escapeHtml(item.disease)}</strong><small>${escapeHtml(item.filename)} · ${escapeHtml(new Date(item.date).toLocaleString())}</small></div><span class="history-confidence">${item.confidence.toFixed(1)}%<small>confidence</small></span></article>`).join(''); }
  $('recent-scans').innerHTML = history.length ? rows(history.slice(0,3)) : empty;
  $('all-scans').innerHTML = history.length ? rows(history) : empty;
  $('clear-history').disabled = !history.length;
}
$('clear-history').onclick = () => {
  if (!window.confirm('Clear all scan history saved in this browser?')) return;
  history = [];
  try { localStorage.removeItem('leaflens-history'); } catch { /* In-memory history is cleared. */ }
  renderHistory();
};
function navigate() {
  const requested = location.hash.slice(1) || 'dashboard';
  const view = ['dashboard','history','library','guide'].includes(requested) ? requested : 'dashboard';
  stopCamera();
  document.querySelectorAll('.view').forEach(el => el.hidden = el.id !== view);
  document.querySelectorAll('.nav-link').forEach(el => { const active = el.dataset.view === view; el.classList.toggle('active',active); if (active) el.setAttribute('aria-current','page'); else el.removeAttribute('aria-current'); });
  $('breadcrumb').textContent = {dashboard:'Overview',history:'Scan history',library:'Plant library',guide:'Photo guide'}[view];
  document.title = `LeafLens · ${$('breadcrumb').textContent}`;
}
window.addEventListener('hashchange', navigate);
let plantClasses = [];
function renderLibrary() {
  const query = $('plant-search').value.toLowerCase().trim();
  const groups = new Map();
  for (const label of plantClasses) {
    const [plant, condition] = label.split('___').map(s => s.replaceAll('_',' ').trim());
    if (!`${plant} ${condition}`.toLowerCase().includes(query)) continue;
    if (!groups.has(plant)) groups.set(plant,[]);
    groups.get(plant).push(condition);
  }
  $('plant-library').innerHTML = groups.size ? Array.from(groups, ([plant, conditions]) => `<article class="card plant-card"><h2>${icon('leaf')}${escapeHtml(plant)}</h2><small>${conditions.length} supported condition${conditions.length === 1 ? '' : 's'}</small><ul>${conditions.map(condition => `<li>${escapeHtml(condition)}</li>`).join('')}</ul></article>`).join('') : '<p>No matching plants or conditions.</p>';
}
$('plant-search').oninput = renderLibrary;
async function init() {
  renderHistory(); navigate();
  try {
    const [healthResponse, classesResponse] = await Promise.all([fetch('/api/health'), fetch('/api/classes')]);
    if (!healthResponse.ok || !classesResponse.ok) throw new Error('Server unavailable');
    const health = await healthResponse.json();
    $('model-status').classList.toggle('ready',health.model_ready);
    $('model-status').innerHTML = `${icon('layers')}<span>${escapeHtml(health.message)}</span>`;
    plantClasses = (await classesResponse.json()).classes;
    renderLibrary();
  } catch {
    $('model-status').innerHTML = `${icon('help')}<span>Cannot connect to the server. Refresh the page to try again.</span>`;
    $('plant-library').textContent = 'The plant library could not be loaded. Refresh to try again.';
  }
}
init();
async function updateTrainingStatus() {
  try {
    const response = await fetch('/api/training');
    if (!response.ok) return;
    const status = await response.json();
    const active = ['preparing', 'downloading', 'training', 'validating'].includes(status.state);
    if (active) {
      const healthResponse = await fetch('/api/health');
      const health = healthResponse.ok ? await healthResponse.json() : {model_ready:false};
      let message = status.state === 'downloading' ? 'Downloading the training dataset…' : 'Preparing the training run…';
      if (status.state === 'training') {
        const percent = Math.round(100 * status.completed_batches / status.batches_per_epoch);
        message = `Training ResNet-9 · epoch ${status.epoch}/${status.epochs} · ${percent}% of epoch · CPU`;
        if (status.estimated_training_seconds_remaining) message += ` · ~${(status.estimated_training_seconds_remaining / 3600).toFixed(1)} h training remaining, plus validation`;
      }
      if (status.state === 'validating') message = `Validating epoch ${status.epoch}/${status.epochs} · ${status.validated_images || 0}/${status.validation_images} images`;
      const availability = health.model_ready ? 'Leaf scans are available using the best validated checkpoint.' : 'Predictions become available after the first validated checkpoint is saved.';
      $('model-status').classList.toggle('ready', health.model_ready);
      $('model-status').innerHTML = `${icon('layers')}<span>${escapeHtml(message)}. ${availability}</span>`;
      if (health.model_ready && $('scan-error').textContent.includes('trained model')) error();
    } else if (status.state === 'complete') {
      const health = await (await fetch('/api/health')).json();
      $('model-status').classList.toggle('ready', health.model_ready);
      const accuracy = (status.best_validation_accuracy * 100).toFixed(2);
      $('model-status').innerHTML = `${icon('layers')}<span>Training complete · ${accuracy}% accuracy on the dataset validation split. ${health.model_ready ? 'Model ready for leaf scans.' : 'Loading trained weights…'}</span>`;
    } else if (['failed', 'interrupted'].includes(status.state)) {
      $('model-status').innerHTML = `${icon('help')}<span>Training ${escapeHtml(status.state)}. Check logs/training.err.log and resume the run.</span>`;
    }
  } catch { /* Keep the last known state when offline. */ }
}
updateTrainingStatus();
setInterval(updateTrainingStatus, 15000);
