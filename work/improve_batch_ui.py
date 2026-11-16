from pathlib import Path
p=Path('btr_gens/btr_gui_index.html')
s=p.read_text(encoding='utf-8')
css='''
  /* Right column has one scroll container: no compressed/overlapping results. */
  .right{overflow-y:auto;min-height:0;scrollbar-gutter:stable}
  .mapwrap,.foot,.batch-panel{flex-shrink:0}
  #mapcanvas{height:clamp(240px,36vh,460px)}
  .result{flex:none;overflow:visible;min-height:110px}
  .btn:disabled{opacity:.45;cursor:not-allowed;filter:none}
  .batch-panel{margin:14px;padding:18px;background:var(--panel);border:1px solid #34465e;border-radius:12px;box-shadow:0 8px 28px #0002}
  .batch-head{display:flex;align-items:center;justify-content:space-between;gap:12px;margin-bottom:16px}
  .batch-head h2{margin:0;font-size:17px;color:var(--acc)}
  .batch-head p{margin:4px 0 0;color:var(--dim);font-size:12px}
  .batch-badge{flex-shrink:0;background:#253348;color:#a8b9d1;border:1px solid #3c4c64;padding:3px 10px;border-radius:20px;font-size:12px}
  .batch-panel[data-state=running] .batch-badge{background:#12374c;color:#67d4ff}
  .batch-panel[data-state=completed] .batch-badge{background:#153e30;color:#6de5a3}
  .batch-panel[data-state=failed] .batch-badge{background:#48292d;color:#ffb1b1}
  .batch-fields{display:grid;grid-template-columns:minmax(100px,1fr) minmax(220px,2.5fr) minmax(130px,1.3fr) minmax(95px,1fr);gap:12px}
  .batch-fields label{display:flex;flex-direction:column;gap:6px;color:var(--dim);font-size:12px;min-width:0}
  .batch-fields input,.batch-fields select{width:100%;min-width:0;height:38px;padding:7px 10px;border:1px solid var(--line);border-radius:6px;background:var(--input);color:var(--txt);font:inherit;font-size:13px;color-scheme:dark}
  .batch-fields input:focus,.batch-fields select:focus{outline:2px solid #4fc3f755;border-color:var(--acc)}
  .batch-fields input:disabled{opacity:.4}
  .batch-actions{display:flex;align-items:center;gap:10px;margin-top:16px;flex-wrap:wrap}
  .batch-note{color:var(--dim);font-size:12px;margin:12px 0;line-height:1.7}
  .batch-monitor{padding:14px;background:#0e1623;border:1px solid var(--line);border-radius:8px}
  .batch-metrics{display:flex;justify-content:space-between;align-items:center;gap:12px;flex-wrap:wrap;margin-bottom:10px;font-size:13px}
  .batch-metrics strong{font-size:17px;font-variant-numeric:tabular-nums}
  #batchSuccess{color:var(--ok)} #batchFailure{color:var(--err)}
  #batchProgress{display:block;appearance:none;width:100%;height:8px;border:0;border-radius:8px;overflow:hidden;background:#283448;accent-color:var(--acc)}
  #batchProgress::-webkit-progress-bar{background:#283448;border-radius:8px}
  #batchProgress::-webkit-progress-value{background:linear-gradient(90deg,#229bd0,#6bd8f5);border-radius:8px;transition:width .3s}
  #batchStatus{margin-top:12px;white-space:pre-wrap;overflow-wrap:anywhere;font-size:13px;min-height:22px}
  #batchStatus[data-error=true]{color:#ffaaaa}
  .batch-output{display:flex;gap:8px;margin-top:12px;align-items:flex-start;color:var(--dim);font-size:12px}
  #batchOutput{font-family:Consolas,monospace;color:#c7d8ec;overflow-wrap:anywhere;user-select:all;flex:1}
  .batch-files{margin-top:14px;font-size:13px}
  .batch-files summary{cursor:pointer;color:var(--acc);padding:6px 0}
  #batchResults{max-height:300px;overflow:auto;padding:4px 0;line-height:1.9}
  @media(max-width:1150px){.left{width:370px;min-width:370px}.batch-fields{grid-template-columns:1fr 1fr}}
  @media(max-width:760px){.wrap{height:auto;display:block}.left{width:100%;min-width:0;max-height:45vh}.right{overflow:visible}.topbar{flex-wrap:wrap}.batch-fields{grid-template-columns:1fr}.mode-switch{margin-left:0}}
'''
s=s.replace('</style>',css+'</style>',1)
start=s.index('    <section style="padding:14px 20px;')
end=s.index('    <div class="result" id="resultArea">',start)
s=s[:start]+'''    <section id="batchPanel" class="batch-panel" data-state="idle" aria-labelledby="batchHeading">
      <div class="batch-head">
        <div><h2 id="batchHeading">批量生成</h2><p>一次生成多个独立样本，用于模型识别与轨迹评测</p></div>
        <span id="batchBadge" class="batch-badge">未开始</span>
      </div>
      <div class="batch-fields">
        <label>样本数量（次仿真）<input id="batchCount" type="number" min="1" max="1000" step="1" value="10" required></label>
        <label>生成方式<select id="batchMode" onchange="updateBatchHint()">
          <option value="noise">固定路径 · 只改变随机噪声</option>
          <option value="perturb">当前场景 · 随机扰动参数</option>
          <option value="scenarios">多种场景 · S01～S16 轮换</option>
        </select></label>
        <label>随机种子<input id="batchSeed" type="number" min="0" max="2147482647" step="1" value="20260911" required></label>
        <label>扰动范围（±%）<input id="batchPercent" type="number" min="0" max="50" value="15" required></label>
      </div>
      <p id="batchModeHint" class="batch-note"></p>
      <div class="batch-actions">
        <button id="batchStart" class="btn primary" onclick="startBatch()">▶ 开始生成 10 个样本</button>
        <button id="batchStop" class="btn" onclick="stopBatch()" disabled>停止后续生成</button>
        <span class="hint">上方“单次仿真”只生成 1 个样本</span>
      </div>
      <p class="batch-note">模型输入图与真值分开保存。请只向模型提供 input 文件，不要上传 truth 或整个样本文件夹。</p>
      <div class="batch-monitor">
        <div class="batch-metrics"><span>已处理 <strong id="batchDone">0 / 10</strong></span><span>成功 <strong id="batchSuccess">0</strong>　失败 <strong id="batchFailure">0</strong>　<span id="batchPercentText">0%</span></span></div>
        <progress id="batchProgress" aria-label="批量生成进度" max="10" value="0"></progress>
        <div id="batchStatus" role="status" aria-live="polite">尚未开始。设置数量后，点击“开始生成”按钮。</div>
        <div class="batch-output"><span>保存位置</span><span id="batchOutput">btr_out / batch_时间_编号 / sample_0001…</span></div>
      </div>
      <details id="batchDetails" class="batch-files"><summary>查看样本文件与批次清单</summary><div id="batchResults">任务完成后在这里查看文件。</div></details>
    </section>
'''+s[end:]
s=s.replace('onclick="runSim()">▶ 生成仿真','onclick="runSim()">▶ 单次仿真（1 个样本）',1)
s=s.replace('let batchTimer = null;','''let batchTimer = null;
let batchIsBusy = false;
function batchStatus(message, error=false){
  const status=document.getElementById("batchStatus");
  status.textContent=message; status.dataset.error=String(error);
  if(error){ document.getElementById("batchPanel").dataset.state="failed"; document.getElementById("batchBadge").textContent="需要处理"; }
}
function updateBatchHint(){
  const mode=document.getElementById("batchMode").value;
  const hints={noise:"使用左侧平台与目标配置，路径相同，仅随机噪声不同。",perturb:"在左侧配置上扰动航点路径、航速、强度和噪声；曲线公式与出现/消失时间保持不变。",scenarios:"自动轮换16类场景，忽略左侧参数，固定600秒。数量16表示每类1个，32表示每类2个；不足16时仅覆盖前几类。"};
  document.getElementById("batchModeHint").textContent=hints[mode];
  document.getElementById("batchPercent").disabled=batchIsBusy || mode!=="perturb";
  if(!batchIsBusy) document.getElementById("batchStart").textContent="▶ 开始生成 "+document.getElementById("batchCount").value+" 个样本";
}
async function batchRequest(url, options={}){
  const response=await fetch(url,{...options,signal:AbortSignal.timeout(20000)});
  const raw=await response.text();
  let data;
  try{ data=JSON.parse(raw); }catch{
    throw new Error("当前服务未提供批量接口，可能仍是旧版后端。请打开新版端口8766，或停止旧服务后重新启动程序。");
  }
  if(!response.ok || data.ok===false){
    const error=new Error(data.error || "请求失败（"+response.status+"）"); error.status=response.status; throw error;
  }
  return data;
}''',1)
s=s.replace('function batchBusy(busy){','''function batchBusy(busy){
  batchIsBusy=busy;
  for(const id of ["batchCount","batchMode","batchSeed"]) document.getElementById(id).disabled=busy;''',1)
s=s.replace('  document.getElementById("singleRunButton").disabled = busy;\n}', '''  document.getElementById("singleRunButton").disabled = busy;
  if(busy) document.getElementById("batchStart").textContent="生成中，请等待…";
  updateBatchHint();
}''',1)
s=s.replace('''  batchBusy(true);
  document.getElementById("batchStatus").textContent = "正在校验并创建批次…";''','''  if(batchIsBusy) return;
  for(const id of ["batchCount","batchSeed","batchPercent"]){
    const field=document.getElementById(id);
    if(!field.disabled && !field.reportValidity()) return;
  }
  batchBusy(true);
  batchStatus("正在校验参数并创建批次，请勿重复点击…");
  document.getElementById("batchPanel").dataset.state="running";
  document.getElementById("batchBadge").textContent="正在创建";''',1)
s=s.replace('const response = await fetch("/batch/start",','const data = await batchRequest("/batch/start",',1)
s=s.replace('''    const data = await response.json();
    if(!data.ok) throw new Error(data.error);
    batchId = data.job.id;''','''    batchId = data.job.id;
    document.getElementById("batchProgress").value=0;
    document.getElementById("batchDone").textContent="0 / "+data.job.total;
    document.getElementById("batchSuccess").textContent="0";
    document.getElementById("batchFailure").textContent="0";
    document.getElementById("batchPercentText").textContent="0%";
    document.getElementById("batchResults").replaceChildren();''',1)
s=s.replace('document.getElementById("batchStatus").textContent = "创建失败："+error.message;', 'batchStatus("创建失败："+error.message+(error.name==="TimeoutError"?" 请求超时不代表任务未创建，请稍后刷新确认，勿重复提交。":""),true);',1)
a=s.index('    const response = await fetch("/batch/status/')
b=s.index('    const active =',a)
s=s[:a]+'''    const job = await batchRequest("/batch/status/"+encodeURIComponent(batchId));
'''+s[b:]
a=s.index('    document.getElementById("batchStatus").textContent = `${states')
b=s.index('    const area =',a)
s=s[:a]+'''    const processed=job.completed+job.failed;
    document.getElementById("batchDone").textContent=processed+" / "+job.total;
    document.getElementById("batchSuccess").textContent=job.completed;
    document.getElementById("batchFailure").textContent=job.failed;
    document.getElementById("batchPercentText").textContent=Math.round(processed/job.total*100)+"%";
    document.getElementById("batchOutput").textContent=job.out_dir;
    document.getElementById("batchPanel").dataset.state=job.failed || job.state==="failed" ? "failed" : active ? "running" : job.state==="completed" ? "completed" : "idle";
    document.getElementById("batchBadge").textContent=states[job.state] || job.state;
    batchStatus((states[job.state] || job.state)+(active?` · 正在处理第 ${job.current} 个样本；单个样本完成后更新计数。`: ` · 共生成 ${job.completed} 个样本。`)+(job.error?"\\n"+job.error:""),job.state==="failed");
'''+s[b:]
s=s.replace('''    area.append(link("批次manifest",job.id+"/manifest.json"),link("模型输入清单",job.id+"/model_inputs.jsonl"),link("批次参数",job.id+"/batch_config.json"));''','''    area.append(link("批次清单",job.id+"/manifest.json"),link("批次参数",job.id+"/batch_config.json"));
    if(job.completed) area.append(link("模型输入清单",job.id+"/model_inputs.jsonl"));''',1)
s=s.replace('''    else localStorage.removeItem("btrActiveBatch");''','''    else { localStorage.setItem("btrActiveBatch",job.id); document.getElementById("batchDetails").open=true; }''',1)
s=s.replace('''    document.getElementById("batchStatus").textContent="进度查询失败："+error.message+"；已知批次："+batchId;
    if(batchId) batchTimer=setTimeout(pollBatch,3000);''','''    batchStatus("进度查询失败："+error.message+"；批次："+batchId,true);
    if(error.status===404){localStorage.removeItem("btrActiveBatch");batchId="";batchBusy(false);}
    if(batchId) batchTimer=setTimeout(pollBatch,3000);''',1)
s=s.replace('''    const response=await fetch("/batch/stop",''','''    const data=await batchRequest("/batch/stop",''',1)
s=s.replace('''    const data=await response.json();
    if(!data.ok) throw new Error(data.error);
    await pollBatch();''','''    await pollBatch();''',1)
s=s.replace('''}catch(error){ document.getElementById("batchStatus").textContent="停止失败："+error.message; }''','''}catch(error){ batchStatus("停止失败："+error.message,true); }''',1)
s=s.replace('if(batchId) pollBatch();','''document.getElementById("batchCount").addEventListener("input",updateBatchHint);
updateBatchHint();
if(batchId) pollBatch();''',1)
p.write_text(s,encoding='utf-8')
Path('work/gui_script.js').write_text(s.split('<script>')[1].split('</script>')[0],encoding='utf-8')
