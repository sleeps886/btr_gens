from pathlib import Path
p=Path('btr_gens/btr_gui_index.html')
s=p.read_text(encoding='utf-8')
s=s.replace('<button class="btn primary" onclick="runSim()">', '<button id="singleRunButton" class="btn primary" onclick="runSim()">',1)
s=s.replace('    <div class="result" id="resultArea">', '''    <section style="padding:14px 20px;border-bottom:1px solid #30394a;background:#171e2b">
      <h2 style="margin:0 0 10px;color:#4fc3f7">批量生成 · 模型测试数据集</h2>
      <div style="display:flex;gap:10px;flex-wrap:wrap;align-items:center">
        <label>样本数 <input id="batchCount" type="number" min="1" max="1000" value="10" style="width:75px"></label>
        <label>模式 <select id="batchMode">
          <option value="noise">当前场景，仅更换噪声种子</option>
          <option value="perturb">当前场景，随机扰动参数</option>
          <option value="scenarios">S01～S16 预设场景轮换</option>
        </select></label>
        <label>种子 <input id="batchSeed" type="number" min="0" value="20260911" style="width:110px"></label>
        <label>扰动 ±% <input id="batchPercent" type="number" min="0" max="50" value="15" style="width:65px"></label>
        <button id="batchStart" class="btn primary" onclick="startBatch()">开始批量生成</button>
        <button id="batchStop" class="btn" onclick="stopBatch()" disabled>停止后续生成</button>
      </div>
      <p style="font-size:12px;color:#9aaac1">保存到 btr_out/批次目录。模型用图无真值参考线；真值独立保存，勿上传整个样本作为盲测输入。预设模式使用自身600秒场景参数，忽略左侧配置；扰动模式保留曲线公式和出现/消失时间。</p>
      <progress id="batchProgress" max="1" value="0" style="width:100%"></progress>
      <div id="batchStatus" role="status" style="margin:8px 0;overflow-wrap:anywhere">批量任务就绪</div>
      <div id="batchResults" style="max-height:260px;overflow:auto"></div>
    </section>
    <div class="result" id="resultArea">''',1)
s=s.replace('async function runSim(){\n  const status = document.getElementById("runStatus");\n  status.textContent = "运行中…";\n  const params = {', 'function collectParams(){\n  return {',1)
s=s.replace('  };\n  try{\n    const resp = await fetch("/run", {', '''  };
}
async function runSim(){
  const status = document.getElementById("runStatus");
  status.textContent = "运行中…";
  const params = collectParams();
  try{
    const resp = await fetch("/run", {''',1)
# Keep batch state separate from legacy single-run result panel.
pos=s.index('function renderResult(res){')
s=s[:pos]+'''let batchId = localStorage.getItem("btrActiveBatch") || "";
let batchTimer = null;
function batchBusy(busy){
  document.getElementById("batchStart").disabled = busy;
  document.getElementById("batchStop").disabled = !busy;
  document.getElementById("singleRunButton").disabled = busy;
}
function batchUrl(path){
  return "/batch/files/" + path.split("/").map(encodeURIComponent).join("/");
}
async function startBatch(){
  batchBusy(true);
  document.getElementById("batchStatus").textContent = "正在校验并创建批次…";
  try{
    const response = await fetch("/batch/start", {method:"POST", headers:{"Content-Type":"application/json"}, body:JSON.stringify({
      count:Number(document.getElementById("batchCount").value),
      mode:document.getElementById("batchMode").value,
      seed:Number(document.getElementById("batchSeed").value),
      percent:Number(document.getElementById("batchPercent").value), params:collectParams()
    })});
    const data = await response.json();
    if(!data.ok) throw new Error(data.error);
    batchId = data.job.id;
    localStorage.setItem("btrActiveBatch", batchId);
    await pollBatch();
  }catch(error){
    document.getElementById("batchStatus").textContent = "创建失败："+error.message;
    batchBusy(false);
  }
}
async function pollBatch(){
  clearTimeout(batchTimer);
  if(!batchId) return;
  try{
    const response = await fetch("/batch/status/"+encodeURIComponent(batchId));
    const job = await response.json();
    if(!response.ok){
      if(response.status === 404){ localStorage.removeItem("btrActiveBatch"); batchId=""; batchBusy(false); }
      throw new Error(job.error || "查询失败");
    }
    const active = ["queued","running","stopping"].includes(job.state);
    batchBusy(active);
    const states = {queued:"排队中",running:"生成中",stopping:"正在停止（等待当前样本完成）",stopped:"已停止",completed:"已完成",completed_with_errors:"完成，部分样本失败",failed:"批次异常"};
    document.getElementById("batchProgress").max = job.total;
    document.getElementById("batchProgress").value = job.completed+job.failed;
    document.getElementById("batchStatus").textContent = `${states[job.state] || job.state} · 当前 ${job.current}/${job.total} · 成功 ${job.completed} · 失败 ${job.failed} · ${job.out_dir}`+(job.error ? " · "+job.error : "");
    const area = document.getElementById("batchResults");
    area.replaceChildren();
    function link(label,path){
      const a=document.createElement("a"); a.textContent=label; a.href=batchUrl(path); a.target="_blank"; a.rel="noopener"; a.style.marginRight="14px"; return a;
    }
    area.append(link("批次manifest",job.id+"/manifest.json"),link("模型输入清单",job.id+"/model_inputs.jsonl"),link("批次参数",job.id+"/batch_config.json"));
    for(const sample of job.samples){
      const row=document.createElement("div"); row.style.marginTop="8px";
      const title=document.createElement("span"); title.textContent=sample.sample_id+" · "+(sample.state==="completed"?"成功":"失败："+sample.error)+" "; row.append(title);
      if(sample.state==="completed"){
        row.append(link("实际参数/真值元数据",job.id+"/"+sample.sample_id+"/meta.json"));
        for(const path of Object.keys(sample.files || {})){
          if(path.endsWith(".png")) row.append(link(path.split("/").pop(),job.id+"/"+path));
        }
      }
      area.append(row);
    }
    if(active) batchTimer=setTimeout(pollBatch,1500);
    else localStorage.removeItem("btrActiveBatch");
  }catch(error){
    document.getElementById("batchStatus").textContent="进度查询失败："+error.message+"；已知批次："+batchId;
    if(batchId) batchTimer=setTimeout(pollBatch,3000);
  }
}
async function stopBatch(){
  try{
    const response=await fetch("/batch/stop",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({id:batchId})});
    const data=await response.json();
    if(!data.ok) throw new Error(data.error);
    await pollBatch();
  }catch(error){ document.getElementById("batchStatus").textContent="停止失败："+error.message; }
}
if(batchId) pollBatch();

'''+s[pos:]
p.write_text(s,encoding='utf-8')
print('UI updated')
