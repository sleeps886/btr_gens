'use strict';
const q = id => document.getElementById(id);
const QUICK_KEY = 'btr-quick-count-v1';
const PLATFORM_KEY = 'btr-quick-platform-count-v1';
const TARGET_KEY = 'btr-quick-target-count-v1';
const JOB_KEY = 'btr-studio-job-v1';
const STATE_NAMES = {queued:'排队中',running:'生成中',stopping:'等待当前样本完成',stopped:'已停止',completed:'全部完成',completed_with_errors:'存在失败',failed:'未达标',interrupted:'已中断'};
let active = false, submitting = false, restoring = true, polling = false, jobId = null, pollTimer = null;
function readStorage(key) { try { return localStorage.getItem(key); } catch { return null; } }
function saveStorage(key, value) { try { localStorage.setItem(key, value); } catch { /* Generation does not require browser storage. */ } }
function element(tag, text) { const el = document.createElement(tag); if (text !== undefined) el.textContent = text; return el; }
function showNotice(text, error = false) {
  q('quickNotice').textContent = text;
  q('quickNotice').className = 'notice' + (error ? ' error' : '');
  q('quickNotice').hidden = !text;
}
function validCount(value) {
  if (!String(value).trim()) throw new Error('请输入目标样本数量。');
  const count = Number(value);
  if (!Number.isInteger(count) || count < 1 || count > 10000) throw new Error('目标样本数量须为 1～10,000 的整数。');
  return count;
}
function validPlatformCount(value) {
  const count = Number(value);
  if (!Number.isInteger(count) || count < 1 || count > 8) throw new Error('平台数量须为 1～8 的整数。');
  return count;
}
function validTargetCount(value) {
  const count = Number(value);
  if (!Number.isInteger(count) || count < 1 || count > 12) throw new Error('每个样本目标数量须为 1～12 的整数。');
  return count;
}
function successfulProgress(job) {
  const done = job.mode === 'auto' ? job.completed : job.completed + job.failed;
  return Math.min(100, 100 * done / job.total);
}
function controls() {
  const busy = active || submitting || restoring;
  q('quickCount').disabled = busy;
  q('quickPlatformCount').disabled = busy;
  q('quickTargetCount').disabled = busy;
  q('quickStart').disabled = busy;
  q('quickStartText').textContent = restoring ? '正在连接…' : submitting ? '正在提交…' : active ? '正在生成样本…' : '开始随机生成';
  q('quickStop').disabled = !active || submitting || q('quickBadge').dataset.state === 'stopping';
}
async function quickRequest(path, data) {
  const options = {signal: AbortSignal.timeout(20000)};
  if (data !== undefined) Object.assign(options, {method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(data)});
  const response = await fetch(path, options);
  let result;
  try { result = await response.json(); } catch { throw new Error('服务未返回有效结果，请确认本地服务正在运行。'); }
  if (!response.ok || result.ok === false) throw new Error(result.error || `请求失败：HTTP ${response.status}`);
  return result;
}
function downloadLink(path, text) {
  const link = element('a', text);
  link.href = '/batch/files/' + path.split('/').map(encodeURIComponent).join('/');
  link.target = '_blank'; link.rel = 'noopener';
  return link;
}
function samplePreviews(sample) {
  const kinds = [
    ['track', 'truth', '轨迹图'],
    ['hull', 'input', '舷侧阵 BTR'],
    ['tow', 'input', '拖曳阵 BTR']
  ];
  const platforms = new Map();
  for (const path of Object.keys(sample.files)) {
    const [sampleId, dir, filename, extra] = path.split('/');
    if (sampleId !== sample.sample_id || extra !== undefined) continue;
    const match = /^btr_(.+)_(track|hull|tow)\.png$/.exec(filename || '');
    if (!match || !kinds.some(([kind, folder]) => kind === match[2] && folder === dir)) continue;
    const [, platform, kind] = match;
    if (!platforms.has(platform)) platforms.set(platform, new Map());
    platforms.get(platform).set(kind, {path, filename});
  }
  return [...platforms.keys()].sort().map(platform => ({
    platform,
    images: kinds.flatMap(([kind, , title]) => {
      const image = platforms.get(platform).get(kind);
      return image ? [{title, ...image}] : [];
    })
  }));
}
function renderPreviews(jobId, sample) {
  for (const {platform, images} of samplePreviews(sample)) {
    const group = element('section'), gallery = element('div');
    group.className = 'quick-preview-group'; gallery.className = 'gallery';
    group.append(element('h3', `平台 ${platform} · ${sample.sample_id}`), gallery);
    for (const {title, path, filename} of images) {
      const figure = element('figure'), link = downloadLink(jobId+'/'+path, ''), image = element('img');
      image.src = link.href; image.alt = `平台 ${platform} · ${title}`; image.loading = 'lazy';
      const caption = element('figcaption');
      caption.append(element('strong', title), element('span', filename));
      link.append(image); figure.append(link, caption); gallery.append(figure);
    }
    q('quickGallery').append(group);
  }
}
function showTask(job) {
  jobId = job.id;
  saveStorage(JOB_KEY, jobId);
  active = ['queued','running','stopping'].includes(job.state);
  q('quickEmpty').hidden = true;
  q('quickResult').hidden = false;
  q('quickBadge').textContent = STATE_NAMES[job.state] || job.state;
  q('quickBadge').dataset.state = job.state;
  q('quickCompleted').textContent = job.completed;
  q('quickTotal').textContent = job.total;
  q('quickFailed').textContent = job.failed;
  q('quickFailureLabel').textContent = job.mode === 'auto' ? '失败尝试' : '失败样本';
  const progress = successfulProgress(job);
  q('quickProgress').value = progress;
  q('quickPercent').textContent = progress.toFixed(1) + '%';
  if (active) {
    q('quickCount').value = job.total;
    if (job.mode === 'auto' && Number.isInteger(job.platform_count)) q('quickPlatformCount').value = job.platform_count;
    if (job.mode === 'auto' && Number.isInteger(job.target_count)) q('quickTargetCount').value = job.target_count;
    q('quickStage').textContent = (job.state === 'stopping' ? '停止已请求，当前样本完成后结束。' : `正在生成第 ${job.current || 1} 个成功样本。`) + ` 已成功 ${job.completed} / ${job.total}`;
  } else if (job.state === 'completed') {
    q('quickStage').textContent = `生成完成，已输出 ${job.completed} 个成功样本。`;
  } else {
    q('quickStage').textContent = `${STATE_NAMES[job.state] || job.state}，当前保留 ${job.completed} / ${job.total} 个成功样本。`;
  }
  q('quickPath').textContent = job.out_dir;
  q('quickDownloads').replaceChildren(downloadLink(job.id+'/manifest.json', '任务清单 ↗'), downloadLink(job.id+'/batch_config.json', '本批实际配置 ↗'));
  if (job.completed) q('quickDownloads').append(downloadLink(job.id+'/model_inputs.jsonl', '模型输入索引 ↗'));
  const samples = job.samples || [];
  q('quickRows').replaceChildren();
  for (const sample of [...samples].reverse().slice(0,30)) {
    const row = element('tr');
    if (sample.state === 'failed') row.className = 'error';
    const nameCell = element('td');
    if (sample.state === 'completed') nameCell.append(downloadLink(job.id+'/'+sample.sample_id+'/semantic.json', sample.sample_id+' ↗'));
    else nameCell.textContent = sample.sample_id;
    row.append(nameCell, element('td', sample.label || '原仿真配置'), element('td', sample.state === 'completed' ? '成功' : sample.error || sample.state));
    q('quickRows').append(row);
  }
  const first = samples.find(s => Number.isInteger(s.seed));
  q('quickSeed').textContent = first ? `本批初始种子：${first.seed}。每次尝试的参数和种子详见 meta.json；失败尝试不计入成功数量。` : '当前还没有完成的尝试。';
  const latest = [...samples].reverse().find(s => s.state === 'completed' && s.files);
  if (latest) q('quickDownloads').prepend(downloadLink(job.id+'/'+latest.sample_id+'/semantic.json', '语义 JSON（字段字典格式）↗'));
  const key = latest ? job.id+'/'+latest.sample_id : '';
  if (q('quickGallery').dataset.key !== key) {
    q('quickGallery').dataset.key = key;
    q('quickGallery').replaceChildren();
    if (latest) renderPreviews(job.id, latest);
  }
  if (job.error) showNotice(job.error, true);
  controls();
}
async function refreshTask() {
  if (!jobId || polling) return;
  polling = true;
  try {
    showTask(await quickRequest('/batch/status/'+encodeURIComponent(jobId)));
    if (active) pollTimer = setTimeout(refreshTask, 1500);
  } catch (error) {
    showNotice('状态查询失败，将继续查询：'+error.message, true);
    pollTimer = setTimeout(refreshTask, 4000);
  } finally { polling = false; }
}
async function recoverTask() {
  const result = await quickRequest('/batch/active');
  if (result.jobs.length) {
    showTask(result.jobs[0]);
    clearTimeout(pollTimer); pollTimer = setTimeout(refreshTask, 500);
    return true;
  }
  return false;
}
async function startAuto(event) {
  event?.preventDefault();
  if (active || submitting || restoring) return;
  let count, platformCount, targetCount;
  try { count = validCount(q('quickCount').value); } catch (error) { showNotice(error.message, true); q('quickCount').focus(); return; }
  try { platformCount = validPlatformCount(q('quickPlatformCount').value); } catch (error) { showNotice(error.message, true); q('quickPlatformCount').focus(); return; }
  try { targetCount = validTargetCount(q('quickTargetCount').value); } catch (error) { showNotice(error.message, true); q('quickTargetCount').focus(); return; }
  submitting = true; controls(); showNotice('');
  saveStorage(QUICK_KEY, String(count));
  saveStorage(PLATFORM_KEY, String(platformCount));
  saveStorage(TARGET_KEY, String(targetCount));
  try {
    const response = await quickRequest('/batch/start', {mode:'auto', count, platform_count:platformCount, target_count:targetCount});
    showTask(response.job);
    clearTimeout(pollTimer); pollTimer = setTimeout(refreshTask, 500);
  } catch (error) {
    showNotice('提交未确认：'+error.message, true);
    try {
      if (await recoverTask()) showNotice('已找到服务器正在执行的任务，继续显示实际进度。');
    } catch { /* Never resend a start request automatically. */ }
  } finally { submitting = false; controls(); }
}
async function stopAuto() {
  if (!active || !jobId) return;
  q('quickStop').disabled = true;
  q('quickStop').textContent = '正在请求停止…';
  try {
    const response = await quickRequest('/batch/stop', {id:jobId});
    showTask(response.job);
    showNotice('停止已请求，当前样本完成后结束。已经生成的数据保留。');
  } catch (error) { showNotice(error.message, true); }
  finally { q('quickStop').textContent = '停止后续生成'; controls(); }
}
async function initQuick() {
  const saved = readStorage(QUICK_KEY);
  if (saved) { try { q('quickCount').value = validCount(saved); } catch { /* Keep default count. */ } }
  const savedPlatforms = readStorage(PLATFORM_KEY);
  if (savedPlatforms) { try { q('quickPlatformCount').value = validPlatformCount(savedPlatforms); } catch { /* Keep default platform count. */ } }
  const savedTargets = readStorage(TARGET_KEY);
  if (savedTargets) { try { q('quickTargetCount').value = validTargetCount(savedTargets); } catch { /* Keep default target count. */ } }
  q('quickForm').addEventListener('submit', startAuto);
  q('quickStop').addEventListener('click', stopAuto);
  try {
    if (!(await recoverTask())) {
      const id = readStorage(JOB_KEY);
      if (id) { try { showTask(await quickRequest('/batch/status/'+encodeURIComponent(id))); } catch { /* An old task may have been removed. */ } }
    }
  } catch (error) { showNotice('服务连接失败：'+error.message, true); }
  finally { restoring = false; controls(); }
}
initQuick();
