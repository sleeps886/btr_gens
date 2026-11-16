
"use strict";
/* ================= 全局状态 ================= */
let mode = "single";               // 'single' | 'multi'
let platforms = [];                // 平台数据
let targets = [];                  // 目标数据
let editIdx = 0;                   // 地图上当前编辑的目标索引
let drawing = false;               // 划线中
let picking = false;               // 点选模式
let tempPts = [];                  // 划线过程中的原始点
let bounds = { lat:[29.90,30.15], lon:[121.80,122.30] };

const COLORS = ["#ffd400","#00ffcc","#ff6b6b","#ff9d00","#ff00cc","#00ffff","#aaff00","#ff8800"];
const PALETTE = ["#ff9d00","#ff00cc","#00ffff","#aaff00","#ffd400","#4fc3f7","#ff6b6b","#c77dff"];

/* ================= 平台 ================= */
function newPlatform(name, start_ll, heading, speed){
  return { name, start_ll, heading_deg:heading, speed_kn:speed,
           hull_m:[0,0], tow_m:[-150,0], color: COLORS[platforms.length % COLORS.length] };
}
function addPlatform(){
  const n = platforms.length;
  platforms.push(newPlatform(String.fromCharCode(65+n), [30.02,121.95], 30.0, 0.0));
  renderPlats(); drawMap();
}
function delPlatform(i){ platforms.splice(i,1); renderPlats(); drawMap(); }

function renderPlats(){
  const box = document.getElementById("platList");
  const single = (mode === "single");
  let html = "";
  platforms.forEach((p,i)=>{
    html += `<div class="card">
      <div class="cardhead"><b>平台 ${p.name}</b>
        <span class="spacer"></span>
        ${single ? "" : `<button class="btn danger" onclick="delPlatform(${i})">删除</button>`}
      </div>
      <div class="row"><label>名称</label><input type="text" value="${p.name}" onchange="platforms[${i}].name=this.value;drawMap()"></div>
      <div class="row"><label>经度 °E</label><input type="number" step="0.0001" value="${p.start_ll[1]}" onchange="platforms[${i}].start_ll[1]=parseFloat(this.value);drawMap()"></div>
      <div class="row"><label>纬度 °N</label><input type="number" step="0.0001" value="${p.start_ll[0]}" onchange="platforms[${i}].start_ll[0]=parseFloat(this.value);drawMap()"></div>
      <div class="row"><label>航向 °(北0顺)</label><input type="number" value="${p.heading_deg}" onchange="platforms[${i}].heading_deg=parseFloat(this.value)"></div>
      <div class="row"><label>航速 kn</label><input type="number" value="${p.speed_kn}" onchange="platforms[${i}].speed_kn=parseFloat(this.value)"></div>
      <div class="row"><label>舷侧阵偏移x</label><input type="number" value="${p.hull_m[0]}" onchange="platforms[${i}].hull_m[0]=parseFloat(this.value)"></div>
      <div class="row"><label>舷侧阵偏移y</label><input type="number" value="${p.hull_m[1]}" onchange="platforms[${i}].hull_m[1]=parseFloat(this.value)"></div>
      <div class="row"><label>拖曳阵偏移x</label><input type="number" value="${p.tow_m[0]}" onchange="platforms[${i}].tow_m[0]=parseFloat(this.value)"></div>
      <div class="row"><label>拖曳阵偏移y</label><input type="number" value="${p.tow_m[1]}" onchange="platforms[${i}].tow_m[1]=parseFloat(this.value)"></div>
      <div class="row"><label>颜色</label><input type="color" value="${p.color}" onchange="platforms[${i}].color=this.value;drawMap()"></div>
    </div>`;
  });
  box.innerHTML = html;
  if(!single){
    const add = document.createElement("div");
    add.innerHTML = `<button class="btn" onclick="addPlatform()">＋ 添加平台</button>`;
    box.appendChild(add);
  }
}

/* ================= 目标 ================= */
function newTarget(name){
  return { name, start_ll:[30.00,122.00], speed_kn:15.0, amp:0.25,
           sigma_hull:4.0, sigma_tow:1.6, t_on:0, t_off:600,
           color: PALETTE[targets.length % PALETTE.length],
           path_type:"waypoints", waypoints:[[30.00,122.05]], curve:"", drawn:[] };
}
function addTarget(){
  targets.push(newTarget("T"+(targets.length+1)));
  renderTargets(); refreshEditSel();
  if(editIdx >= targets.length) editIdx = targets.length-1;
  drawMap();
}
function delTarget(i){
  targets.splice(i,1);
  if(editIdx >= targets.length) editIdx = targets.length-1;
  renderTargets(); refreshEditSel(); drawMap();
}

function renderTargets(){
  const box = document.getElementById("tgtList");
  let html = "";
  targets.forEach((t,i)=>{
    const wp = (t.waypoints||[]).map(p=>p[0].toFixed(4)+", "+p[1].toFixed(4)).join("\n");
    html += `<div class="card">
      <div class="cardhead"><b>${t.name}</b>
        <span class="spacer"></span>
        <button class="btn" onclick="editTargetOnMap(${i})">🎯 在地图上划线</button>
        <button class="btn danger" onclick="delTarget(${i})">删除</button>
      </div>
      <div class="row"><label>名称</label><input type="text" value="${t.name}" onchange="targets[${i}].name=this.value"></div>
      <div class="row"><label>起点经度</label><input type="number" step="0.0001" value="${t.start_ll[1]}" onchange="targets[${i}].start_ll[1]=parseFloat(this.value);drawMap()"></div>
      <div class="row"><label>起点纬度</label><input type="number" step="0.0001" value="${t.start_ll[0]}" onchange="targets[${i}].start_ll[0]=parseFloat(this.value);drawMap()"></div>
      <div class="row"><label>航速 kn</label><input type="number" value="${t.speed_kn}" onchange="targets[${i}].speed_kn=parseFloat(this.value)"></div>
      <div class="row"><label>强度 amp</label><input type="number" step="0.01" value="${t.amp}" onchange="targets[${i}].amp=parseFloat(this.value)"></div>
      <div class="row"><label>σ 舷侧阵</label><input type="number" step="0.1" value="${t.sigma_hull}" onchange="targets[${i}].sigma_hull=parseFloat(this.value)"></div>
      <div class="row"><label>σ 拖曳阵</label><input type="number" step="0.1" value="${t.sigma_tow}" onchange="targets[${i}].sigma_tow=parseFloat(this.value)"></div>
      <div class="row"><label>出现 t_on</label><input type="number" value="${t.t_on}" onchange="targets[${i}].t_on=parseFloat(this.value)"></div>
      <div class="row"><label>消失 t_off</label><input type="number" value="${t.t_off}" onchange="targets[${i}].t_off=parseFloat(this.value)"></div>
      <div class="row"><label>颜色</label><input type="color" value="${t.color}" onchange="targets[${i}].color=this.value;drawMap()"></div>
      <div class="pathrow">
        <label style="width:96px;color:var(--dim);font-size:12.5px">路径方式</label>
        <select onchange="targets[${i}].path_type=this.value">
          <option value="waypoints" ${t.path_type==="waypoints"?"selected":""}>航点列表</option>
          <option value="curve" ${t.path_type==="curve"?"selected":""}>曲线公式</option>
          <option value="draw" ${t.path_type==="draw"?"selected":""}>划线取点</option>
        </select>
      </div>
      <div class="row ${t.path_type==="curve"?"":"collapsed"}" id="curveRow${i}">
        <label>曲线 f(t)</label>
        <textarea placeholder="返回 (lat, lon) 的表达式, 如: (30.0225 + 0.0045*np.sin(2*np.pi*t/400.0), 121.9690 + 0.0001227*t)">${t.curve||""}</textarea>
      </div>
      <div class="row ${t.path_type!=="curve"?"":"collapsed"}" id="wpRow${i}">
        <label>航点(每行)</label>
        <textarea placeholder="每行一个航点: 纬度, 经度">${wp}</textarea>
      </div>
    </div>`;
  });
  box.innerHTML = html;

  /* 绑定曲线/航点输入与路径方式联动 */
  targets.forEach((t,i)=>{
    const card = box.children[i];
    const sel = card.querySelector(".pathrow select");
    const curTa = card.querySelector(`#curveRow${i} textarea`);
    const wpTa  = card.querySelector(`#wpRow${i} textarea`);
    sel.addEventListener("change",()=>{
      t.path_type = sel.value;
      card.querySelector(`#curveRow${i}`).classList.toggle("collapsed", t.path_type!=="curve");
      card.querySelector(`#wpRow${i}`).classList.toggle("collapsed", t.path_type==="curve");
    });
    curTa.addEventListener("change",()=>{ t.curve = curTa.value; });
    wpTa.addEventListener("change",()=>{
      t.waypoints = parseWaypoints(wpTa.value);
    });
  });
}

function parseWaypoints(text){
  const pts=[];
  text.split("\n").forEach(line=>{
    line=line.trim().replace(/[()]/g,"");
    if(!line) return;
    const m=line.split(/[,，\s]+/);
    if(m.length>=2 && !isNaN(parseFloat(m[0])) && !isNaN(parseFloat(m[1])))
      pts.push([parseFloat(m[0]),parseFloat(m[1])]);
  });
  return pts;
}

function refreshEditSel(){
  const sel = document.getElementById("editTarget");
  sel.innerHTML = targets.map((t,i)=>`<option value="${i}" ${i===editIdx?"selected":""}>${t.name}</option>`).join("");
}

/* ================= 地图 ================= */
const canvas = document.getElementById("mapcanvas");
const ctx = canvas.getContext("2d");

function lon2x(lon){ return (lon - bounds.lon[0])/(bounds.lon[1]-bounds.lon[0]) * canvas.width; }
function lat2y(lat){ return (1 - (lat - bounds.lat[0])/(bounds.lat[1]-bounds.lat[0])) * canvas.height; }
function x2lon(x){ return bounds.lon[0] + x/canvas.width*(bounds.lon[1]-bounds.lon[0]); }
function y2lat(y){ return bounds.lat[0] + (1 - y/canvas.height)*(bounds.lat[1]-bounds.lat[0]); }

function autoBounds(){
  let las=[], los=[];
  platforms.forEach(p=>{ las.push(p.start_ll[0]); los.push(p.start_ll[1]); });
  targets.forEach(t=>{
    las.push(t.start_ll[0]); los.push(t.start_ll[1]);
    (t.waypoints||[]).forEach(w=>{ las.push(w[0]); los.push(w[1]); });
  });
  if(las.length){
    const pad = Math.max(0.005, (Math.max(...las)-Math.min(...las))*0.15);
    bounds.lat = [Math.min(...las)-pad, Math.max(...las)+pad];
  }
  if(los.length){
    const pad = Math.max(0.005, (Math.max(...los)-Math.min(...los))*0.15);
    bounds.lon = [Math.min(...los)-pad, Math.max(...los)+pad];
  }
  drawMap();
}

function drawMap(){
  ctx.fillStyle = "#05080f";
  ctx.fillRect(0,0,canvas.width,canvas.height);
  /* 网格 (每 0.02°) */
  ctx.strokeStyle = "#15202f"; ctx.lineWidth = 1;
  ctx.beginPath();
  for(let lat=Math.ceil(bounds.lat[0]*50)/50; lat<=bounds.lat[1]; lat+=0.02){
    const y = lat2y(lat);
    ctx.moveTo(0,y); ctx.lineTo(canvas.width,y);
  }
  for(let lon=Math.ceil(bounds.lon[0]*50)/50; lon<=bounds.lon[1]; lon+=0.02){
    const x = lon2x(lon);
    ctx.moveTo(x,0); ctx.lineTo(x,canvas.height);
  }
  ctx.stroke();
  /* 参考点 */
  ctx.strokeStyle = "#3d4a61";
  ctx.beginPath();
  const rx = lon2x(122), ry = lat2y(30);
  ctx.moveTo(rx-8,ry); ctx.lineTo(rx+8,ry); ctx.moveTo(rx,ry-8); ctx.lineTo(rx,ry+8);
  ctx.stroke();

  /* 平台位置 (方块) */
  platforms.forEach(p=>{
    const x = lon2x(p.start_ll[1]), y = lat2y(p.start_ll[0]);
    ctx.fillStyle = p.color;
    ctx.fillRect(x-4,y-4,8,8);
  });

  /* 目标: 起点圆点 + 航点折线 */
  targets.forEach((t,i)=>{
    ctx.strokeStyle = t.color; ctx.fillStyle = t.color;
    ctx.lineWidth = 2;
    const sx = lon2x(t.start_ll[1]), sy = lat2y(t.start_ll[0]);
    ctx.beginPath(); ctx.arc(sx,sy,4,0,2*Math.PI); ctx.fill();
    ctx.beginPath();
    const pts = [t.start_ll, ...(t.waypoints||[])];
    if(pts.length>1){
      ctx.moveTo(lon2x(pts[0][1]), lat2y(pts[0][0]));
      for(let k=1;k<pts.length;k++) ctx.lineTo(lon2x(pts[k][1]), lat2y(pts[k][0]));
      ctx.stroke();
    }
    /* 当前编辑目标高亮 */
    if(i===editIdx){
      ctx.strokeStyle = "#ffffff"; ctx.lineWidth = 1;
      ctx.strokeRect(sx-7,sy-7,14,14);
    }
  });

  /* 划线过程中的临时路径 */
  if(tempPts.length>1 && drawing){
    ctx.strokeStyle = targets[editIdx] ? targets[editIdx].color : "#fff";
    ctx.lineWidth = 2.5;
    ctx.beginPath();
    ctx.moveTo(tempPts[0].x, tempPts[0].y);
    for(let k=1;k<tempPts.length;k++) ctx.lineTo(tempPts[k].x, tempPts[k].y);
    ctx.stroke();
  }

  /* 范围信息 */
  document.getElementById("rngLat").textContent =
      bounds.lat[0].toFixed(4)+" ~ "+bounds.lat[1].toFixed(4)+" °N";
  document.getElementById("rngLon").textContent =
      bounds.lon[0].toFixed(4)+" ~ "+bounds.lon[1].toFixed(4)+" °E";
  syncRangeInputs();
}

/* ---- 经纬度范围手动设置 ---- */
function applyRangeInput(){
  const laLo = parseFloat(document.getElementById("latLo").value);
  const laHi = parseFloat(document.getElementById("latHi").value);
  const loLo = parseFloat(document.getElementById("lonLo").value);
  const loHi = parseFloat(document.getElementById("lonHi").value);
  if(isNaN(laLo)||isNaN(laHi)||isNaN(loLo)||isNaN(loHi)){
    alert("经纬度范围必须为数字"); return;
  }
  if(laLo>=laHi || loLo>=loHi){
    alert("范围输入无效: 下限必须小于上限"); return;
  }
  bounds.lat = [laLo, laHi]; bounds.lon = [loLo, loHi];
  drawMap();
}
function syncRangeInputs(){
  document.getElementById("latLo").value = bounds.lat[0].toFixed(4);
  document.getElementById("latHi").value = bounds.lat[1].toFixed(4);
  document.getElementById("lonLo").value = bounds.lon[0].toFixed(4);
  document.getElementById("lonHi").value = bounds.lon[1].toFixed(4);
}

/* ---- 鼠标交互 ---- */
canvas.addEventListener("mousedown", e=>{
  if(!targets.length) return;
  const r = canvas.getBoundingClientRect();
  const x = (e.clientX-r.left)/r.width*canvas.width;
  const y = (e.clientY-r.top)/r.height*canvas.height;
  if(picking){ addPointXY(x,y); drawMap(); return; }
  drawing = true; tempPts = [{x,y}]; drawMap();
});
canvas.addEventListener("mousemove", e=>{
  if(!drawing) return;
  const r = canvas.getBoundingClientRect();
  const x = (e.clientX-r.left)/r.width*canvas.width;
  const y = (e.clientY-r.top)/r.height*canvas.height;
  const last = tempPts[tempPts.length-1];
  if(Math.hypot(x-last.x, y-last.y) >= 6){  // 最小采样间距 6px
    tempPts.push({x,y}); drawMap();
  }
});
canvas.addEventListener("mouseup", ()=>{
  if(drawing){
    drawing = false;
    if(tempPts.length>=2) finalizeDraw();
    else tempPts=[];
    drawMap();
  }
});
canvas.addEventListener("mousemove", e=>{  // 坐标显示
  const r = canvas.getBoundingClientRect();
  const x = (e.clientX-r.left)/r.width*canvas.width;
  const y = (e.clientY-r.top)/r.height*canvas.height;
  document.getElementById("cursorInfo").textContent =
    `经度 ${x2lon(x).toFixed(5)}°E · 纬度 ${y2lat(y).toFixed(5)}°N`;
});

function addPointXY(x,y){
  const t = targets[editIdx];
  if(!t) return;
  t.waypoints = t.waypoints||[];
  t.waypoints.push([y2lat(y), x2lon(x)]);
  t.path_type = "waypoints";
  renderTargets(); refreshEditSel();
}

function finalizeDraw(){
  /* 划线结束: 按屏幕间隔重采样(去除过密点), 写入该目标航点 */
  const t = targets[editIdx];
  if(!t) return;
  const pts = [];
  let last = null;
  const MINPIX = 14;   // 最小点距 (px), 过滤划线过程中的密点
  tempPts.forEach(p=>{
    if(!last || Math.hypot(p.x-last.x, p.y-last.y) >= MINPIX){
      pts.push(p); last = p;
    }
  });
  t.waypoints = pts.map(p=>[y2lat(p.y), x2lon(p.x)]);
  t.path_type = "waypoints";
  renderTargets(); refreshEditSel();
}

function toggleDraw(){
  if(!targets.length){ alert("请先添加目标"); return; }
  drawing=false; picking=false;
  document.getElementById("btnDraw").classList.toggle("on");
  document.getElementById("btnPick").classList.remove("on");
  tempPts=[];
}
function togglePick(){
  picking=!picking;
  drawing=false;
  document.getElementById("btnPick").classList.toggle("on", picking);
  document.getElementById("btnDraw").classList.remove("on");
}
function undoPoint(){
  const t = targets[editIdx];
  if(t && (t.waypoints||[]).length){
    t.waypoints.pop();
    renderTargets(); drawMap();
  }
}
function clearPoints(){
  const t = targets[editIdx];
  if(t){ t.waypoints=[]; renderTargets(); drawMap(); }
}
function editTargetOnMap(i){
  editIdx = i; refreshEditSel();
  document.getElementById("btnDraw").classList.add("on");
  document.getElementById("btnPick").classList.remove("on");
  picking=false;
  drawMap();
}
document.getElementById("editTarget").addEventListener("change", e=>{
  editIdx = parseInt(e.target.value);
  drawing=false; picking=false;
  document.getElementById("btnDraw").classList.remove("on");
  document.getElementById("btnPick").classList.remove("on");
  drawMap();
});

/* ================= 模式切换 ================= */
function setMode(m){
  mode = m;
  document.getElementById("modeSingle").classList.toggle("on", m==="single");
  document.getElementById("modeMulti").classList.toggle("on", m==="multi");
  document.getElementById("platModeHint").textContent =
      m==="single" ? "(单平台固定, 可改参数)" : "(可添加多个平台)";
  if(m==="single" && platforms.length!==1){
    platforms = [platforms[0] || newPlatform("A",[30.02,121.95],30.0,0.0)];
  }
  renderPlats(); drawMap();
}
document.getElementById("modeSingle").onclick = ()=>setMode("single");
document.getElementById("modeMulti").onclick  = ()=>setMode("multi");

/* ================= 提交运行 ================= */
function collectParams(){
  return {
    T_MAX: parseFloat(document.getElementById("T_MAX").value),
    DT: parseFloat(document.getElementById("DT").value),
    B_MIN: parseFloat(document.getElementById("B_MIN").value),
    B_MAX: parseFloat(document.getElementById("B_MAX").value),
    DB_STEP: 0.25,
    NOISE_STD: parseFloat(document.getElementById("NOISE_STD").value),
    CLUTTER_LAM: parseFloat(document.getElementById("CLUTTER_LAM").value),
    DB_RANGE: [parseFloat(document.getElementById("DB_LO").value),
               parseFloat(document.getElementById("DB_HI").value)],
    SEED: parseInt(document.getElementById("SEED").value),
    EARTH_R: parseFloat(document.getElementById("EARTH_R").value),
    REF_LAT: parseFloat(document.getElementById("REF_LAT").value),
    REF_LON: parseFloat(document.getElementById("REF_LON").value),
    OUT_DIR: document.getElementById("OUT_DIR").value.trim(),
    platforms: platforms.map(p=>({
      name:p.name, start_ll:p.start_ll.slice(), heading_deg:p.heading_deg,
      speed_kn:p.speed_kn, hull_m:p.hull_m.slice(), tow_m:p.tow_m.slice(),
      color:p.color,
      bounds:{lat:[29.85,30.20], lon:[121.75,122.35]}
    })),
    targets: targets.map(t=>{
      const base = {
        name:t.name, start:t.start_ll.slice(), speed_kn:t.speed_kn,
        amp:t.amp, sigma_hull:t.sigma_hull, sigma_tow:t.sigma_tow,
        t_on:t.t_on, t_off:t.t_off, color:t.color, path_type:t.path_type
      };
      if(t.path_type==="curve") base.curve_func = t.curve;
      else base.waypoints = (t.waypoints||[]).map(w=>w.slice());
      return base;
    })
  };
}
async function runSim(){
  const status = document.getElementById("runStatus");
  status.textContent = "运行中…";
  const params = collectParams();
  try{
    const resp = await fetch("/run", {
      method:"POST",
      headers:{"Content-Type":"application/json"},
      body: JSON.stringify(params)
    });
    const res = await resp.json();
    if(!res.ok){ showError(res.error); return; }
    renderResult(res);
    status.textContent = "✔ 完成 — 输出目录: btr_out/"+res.out_dir;
  }catch(err){
    showError("请求失败: "+err.message);
  }
}

let batchId = localStorage.getItem("btrActiveBatch") || "";
let batchTimer = null;
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
}
function batchBusy(busy){
  batchIsBusy=busy;
  for(const id of ["batchCount","batchMode","batchSeed"]) document.getElementById(id).disabled=busy;
  document.getElementById("batchStart").disabled = busy;
  document.getElementById("batchStop").disabled = !busy;
  document.getElementById("singleRunButton").disabled = busy;
  if(busy) document.getElementById("batchStart").textContent="生成中，请等待…";
  updateBatchHint();
}
function batchUrl(path){
  return "/batch/files/" + path.split("/").map(encodeURIComponent).join("/");
}
async function startBatch(){
  if(batchIsBusy) return;
  for(const id of ["batchCount","batchSeed","batchPercent"]){
    const field=document.getElementById(id);
    if(!field.disabled && !field.reportValidity()) return;
  }
  batchBusy(true);
  batchStatus("正在校验参数并创建批次，请勿重复点击…");
  document.getElementById("batchPanel").dataset.state="running";
  document.getElementById("batchBadge").textContent="正在创建";
  try{
    const data = await batchRequest("/batch/start", {method:"POST", headers:{"Content-Type":"application/json"}, body:JSON.stringify({
      count:Number(document.getElementById("batchCount").value),
      mode:document.getElementById("batchMode").value,
      seed:Number(document.getElementById("batchSeed").value),
      percent:Number(document.getElementById("batchPercent").value), params:collectParams()
    })});
    batchId = data.job.id;
    document.getElementById("batchProgress").value=0;
    document.getElementById("batchDone").textContent="0 / "+data.job.total;
    document.getElementById("batchSuccess").textContent="0";
    document.getElementById("batchFailure").textContent="0";
    document.getElementById("batchPercentText").textContent="0%";
    document.getElementById("batchResults").replaceChildren();
    localStorage.setItem("btrActiveBatch", batchId);
    await pollBatch();
  }catch(error){
    batchStatus("创建失败："+error.message+(error.name==="TimeoutError"?" 请求超时不代表任务未创建，请稍后刷新确认，勿重复提交。":""),true);
    batchBusy(false);
  }
}
async function pollBatch(){
  clearTimeout(batchTimer);
  if(!batchId) return;
  try{
    const job = await batchRequest("/batch/status/"+encodeURIComponent(batchId));
    const active = ["queued","running","stopping"].includes(job.state);
    batchBusy(active);
    const states = {queued:"排队中",running:"生成中",stopping:"正在停止（等待当前样本完成）",stopped:"已停止",completed:"已完成",completed_with_errors:"完成，部分样本失败",failed:"批次异常",interrupted:"服务重启，批次已中断"};
    document.getElementById("batchProgress").max = job.total;
    document.getElementById("batchProgress").value = job.completed+job.failed;
    const processed=job.completed+job.failed;
    document.getElementById("batchDone").textContent=processed+" / "+job.total;
    document.getElementById("batchSuccess").textContent=job.completed;
    document.getElementById("batchFailure").textContent=job.failed;
    document.getElementById("batchPercentText").textContent=Math.round(processed/job.total*100)+"%";
    document.getElementById("batchOutput").textContent=job.out_dir;
    document.getElementById("batchPanel").dataset.state=job.failed || job.state==="failed" ? "failed" : active ? "running" : job.state==="completed" ? "completed" : "idle";
    document.getElementById("batchBadge").textContent=states[job.state] || job.state;
    batchStatus((states[job.state] || job.state)+(active?` · 正在处理第 ${job.current} 个样本；单个样本完成后更新计数。`: ` · 共生成 ${job.completed} 个样本。`)+(job.error?"\n"+job.error:""),job.state==="failed");
    const area = document.getElementById("batchResults");
    area.replaceChildren();
    function link(label,path){
      const a=document.createElement("a"); a.textContent=label; a.href=batchUrl(path); a.target="_blank"; a.rel="noopener"; a.style.marginRight="14px"; a.style.color="#4fc3f7"; return a;
    }
    area.append(link("批次清单",job.id+"/manifest.json"),link("批次参数",job.id+"/batch_config.json"));
    if(job.completed) area.append(link("模型输入清单",job.id+"/model_inputs.jsonl"));
    for(const sample of job.samples){
      const row=document.createElement("div"); row.style.marginTop="8px";
      const title=document.createElement("span"); title.textContent=sample.sample_id+" · "+(sample.state==="completed"?"成功":"失败："+sample.error)+" "; row.append(title);
      if(sample.state==="completed"){
        row.append(link("实际参数/真值元数据",job.id+"/"+sample.sample_id+"/meta.json"));
        if(sample.files && sample.files[sample.sample_id+"/semantic.json"]){
          row.append(link("语义结构 JSON（真值参考）",job.id+"/"+sample.sample_id+"/semantic.json"));
        }
        for(const path of Object.keys(sample.files || {})){
          if(path.endsWith(".png")) row.append(link(path.split("/").pop(),job.id+"/"+path));
        }
      }
      area.append(row);
    }
    if(active) batchTimer=setTimeout(pollBatch,1500);
    else { localStorage.setItem("btrActiveBatch",job.id); document.getElementById("batchDetails").open=true; }
  }catch(error){
    batchStatus("进度查询失败："+error.message+"；批次："+batchId,true);
    if(error.status===404){localStorage.removeItem("btrActiveBatch");batchId="";batchBusy(false);}
    if(batchId) batchTimer=setTimeout(pollBatch,3000);
  }
}
async function stopBatch(){
  try{
    const data=await batchRequest("/batch/stop",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({id:batchId})});
    await pollBatch();
  }catch(error){ batchStatus("停止失败："+error.message,true); }
}
document.getElementById("batchCount").addEventListener("input",updateBatchHint);
updateBatchHint();
if(batchId) pollBatch();

function renderResult(res){
  const area = document.getElementById("resultArea");
  let html = `<h2>✅ 仿真完成 · 共 ${res.platforms.length} 个平台</h2>`;
  res.platforms.forEach(pl=>{
    html += `<div class="plcard">
      <h3>平台 ${pl.name} (双站视差 |Δ|=${pl.parallax.toFixed(1)}° @t=300s)</h3>
      <div class="txtlinks">
        <a href="${pl.txts.track}" download>track.txt</a>
        <a href="${pl.txts.hull}" download>hull.txt</a>
        <a href="${pl.txts.tow}" download>tow.txt</a>
      </div>
      <div class="imgrow">
        <figure><img src="${pl.imgs.track}"><figcaption>轨迹图</figcaption></figure>
        <figure><img src="${pl.imgs.hull}"><figcaption>舷侧阵 BTR</figcaption></figure>
        <figure><img src="${pl.imgs.tow}"><figcaption>拖曳阵 BTR</figcaption></figure>
      </div>
    </div>`;
  });
  html += `<div class="logbox">${escapeHtml(res.log||"")}</div>`;
  area.innerHTML = html;
}
function showError(msg){
  const area = document.getElementById("resultArea");
  area.innerHTML = `<div class="errbox">❌ ${escapeHtml(msg)}</div>`;
  document.getElementById("runStatus").textContent = "✘ 运行出错";
}
function escapeHtml(s){
  return s.replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;");
}

/* ================= 初始化 ================= */
platforms.push(newPlatform("A",[30.0623,122.0415],30.0,0.0));
targets.push(newTarget("T1"));
targets[0].name="T1"; targets[0].start_ll=[30.0360,122.1240]; targets[0].speed_kn=19;
targets[0].waypoints=[[30.0360,122.0630]]; targets[0].amp=0.30;
targets.push(newTarget("T2"));
targets[1].name="T2"; targets[1].start_ll=[30.0225,121.9690]; targets[1].speed_kn=23;
targets[1].path_type="curve";
targets[1].curve="(30.0225 + 0.0045*np.sin(2*np.pi*t/400.0), 121.9690 + 0.0001227*t)";
targets[1].amp=0.22; targets[1].sigma_hull=4.0; targets[1].sigma_tow=1.6; targets[1].t_on=60; targets[1].t_off=560;
targets.push(newTarget("T3"));
targets[2].name="T3"; targets[2].start_ll=[30.0045,121.9780]; targets[2].speed_kn=29;
targets[2].waypoints=[[30.0045,122.0580]]; targets[2].amp=0.18; targets[2].sigma_hull=5.0;
targets[2].sigma_tow=2.0; targets[2].t_on=150; targets[2].t_off=450;
targets.push(newTarget("T4"));
targets[3].name="T4"; targets[3].start_ll=[30.0630,122.0620]; targets[3].speed_kn=10;
targets[3].waypoints=[[30.0630,122.0090]]; targets[3].amp=0.10; targets[3].sigma_hull=6.0;
targets[3].sigma_tow=2.4; targets[3].t_on=100; targets[3].t_off=560;
setMode("single");
renderPlats(); renderTargets(); refreshEditSel(); autoBounds(); drawMap();
