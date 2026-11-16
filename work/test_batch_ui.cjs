const fs=require('fs'),vm=require('vm'),assert=require('assert/strict');
const html=fs.readFileSync('btr_gens/btr_gui_index.html','utf8');
const source=html.slice(html.indexOf('let batchId ='),html.indexOf('function renderResult(res){'));
const elements=new Map();
function el(id){if(!elements.has(id))elements.set(id,{value:id==='batchMode'?'noise':id==='batchCount'?'10':'15',disabled:false,dataset:{},textContent:'',style:{},addEventListener(){},replaceChildren(){},append(){}});return elements.get(id);}
const storage=new Map();
const c=vm.createContext({document:{getElementById:el,createElement:()=>({style:{}})},localStorage:{getItem:k=>storage.get(k),setItem:(k,v)=>storage.set(k,v),removeItem:k=>storage.delete(k)},AbortSignal,setTimeout,clearTimeout,encodeURIComponent,fetch:async()=>({ok:false,status:404,text:async()=>'not found'})});
vm.runInContext(source,c);
(async()=>{
 assert.equal(el('batchPercent').disabled,true);
 el('batchMode').value='perturb';vm.runInContext('updateBatchHint()',c);assert.equal(el('batchPercent').disabled,false);
 el('batchCount').value='16';vm.runInContext('updateBatchHint()',c);assert.match(el('batchStart').textContent,/16/);
 vm.runInContext('batchBusy(true)',c);assert.equal(el('batchStart').disabled,true);assert.equal(el('batchCount').disabled,true);
 vm.runInContext('batchBusy(false)',c);assert.equal(el('batchStart').disabled,false);
 await assert.rejects(vm.runInContext('batchRequest("/batch/start")',c),/旧版后端/);
 c.fetch=async()=>({ok:false,status:400,text:async()=>JSON.stringify({ok:false,error:'样本数无效'})});
 await assert.rejects(vm.runInContext('batchRequest("/batch/start")',c),/样本数无效/);
 c.fetch=async()=>({ok:true,status:200,text:async()=>JSON.stringify({ok:true,job:{id:'batch_test'}})});
 assert.equal((await vm.runInContext('batchRequest("/batch/start")',c)).job.id,'batch_test');
 console.log('PASS: mode hints, button count, busy controls, old backend error, structured error, successful response');
})().catch(e=>{console.error(e);process.exitCode=1});
