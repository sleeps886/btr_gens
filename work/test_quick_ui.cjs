const fs=require('fs'),vm=require('vm'),assert=require('assert/strict');
const elements=new Map();
const el=id=>{if(!elements.has(id))elements.set(id,{value:'3',textContent:'',className:'',hidden:false,disabled:false,dataset:{},focus(){}});return elements.get(id);};
const storage=new Map();
const context=vm.createContext({document:{getElementById:el},setTimeout:()=>1,clearTimeout,AbortSignal,localStorage:{getItem:k=>storage.get(k),setItem:(k,v)=>storage.set(k,v)}});
vm.runInContext(fs.readFileSync('btr_gens/btr_quick.js','utf8').replace(/\ninitQuick\(\);\s*$/,''),context);
(async()=>{
 for(const v of ['',0,-1,1.2,10001,'x'])assert.throws(()=>vm.runInContext(`validCount(${JSON.stringify(v)})`,context));
 assert.equal(vm.runInContext('validCount("100")',context),100);
 for(const v of ['',0,1.2,9,'x'])assert.throws(()=>vm.runInContext(`validPlatformCount(${JSON.stringify(v)})`,context));
 assert.equal(vm.runInContext('validPlatformCount("8")',context),8);
 for(const v of ['',0,1.2,13,'x'])assert.throws(()=>vm.runInContext(`validTargetCount(${JSON.stringify(v)})`,context));
 assert.equal(vm.runInContext('validTargetCount("12")',context),12);
 context.sample={sample_id:'sample_0001',files:{
  'sample_0001/input/btr_B_hull.png':'hash',
  'sample_0001/input/btr_A_hull.png':'hash',
  'sample_0001/input/btr_A_tow.png':'hash',
  'sample_0001/truth/btr_A_track.png':'hash',
  'sample_0001/truth/btr_B_track.png':'hash',
  'sample_0001/input/platform.json':'hash'
 }};
 const previews=JSON.parse(vm.runInContext('JSON.stringify(samplePreviews(sample))',context));
 assert.deepEqual(previews.map(({platform,images})=>[platform,images.map(({title,path})=>[title,path.split('/').pop()])]),[
  ['A',[['轨迹图','btr_A_track.png'],['舷侧阵 BTR','btr_A_hull.png'],['拖曳阵 BTR','btr_A_tow.png']]],
  ['B',[['轨迹图','btr_B_track.png'],['舷侧阵 BTR','btr_B_hull.png']]]
 ]);
 assert.equal(vm.runInContext('successfulProgress({mode:"auto",completed:2,failed:4,total:4})',context),50);
 assert.equal(vm.runInContext('successfulProgress({completed:2,failed:1,total:4})',context),75);
 context.calls=[];context.job={id:'batch_test',mode:'auto',state:'queued',total:3};
 let resolve;
 context.pending=new Promise(r=>resolve=r);
 vm.runInContext('restoring=false; showTask=j=>{active=true;jobId=j.id}; quickRequest=async(path,data)=>{calls.push({path,data});return await pending}',context);
 const start=vm.runInContext('startAuto()',context);
 assert.equal(el('quickCount').disabled,true);
 assert.equal(el('quickPlatformCount').disabled,true);
 assert.equal(el('quickTargetCount').disabled,true);
 await vm.runInContext('startAuto()',context);
 assert.equal(context.calls.length,1,'prevent double submission');
 assert.deepEqual(JSON.parse(JSON.stringify(context.calls[0])),{path:'/batch/start',data:{mode:'auto',count:3,platform_count:3,target_count:3}});
 resolve({job:context.job});await start;
 assert.equal(el('quickStart').disabled,true);
 assert.equal(storage.get('btr-quick-count-v1'),'3');
 assert.equal(storage.get('btr-quick-platform-count-v1'),'3');
 assert.equal(storage.get('btr-quick-target-count-v1'),'3');
 console.log('PASS: input validation, success-only progress, platform count request, submission lock and saved quantities');
})().catch(e=>{console.error(e);process.exitCode=1});
