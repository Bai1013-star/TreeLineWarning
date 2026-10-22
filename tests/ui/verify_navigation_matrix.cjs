// Finite state-transition coverage, not a claim to enumerate infinite histories.
const {chromium}=require(process.env.PLAYWRIGHT_MODULE);
const fs=require('node:fs'),assert=require('node:assert/strict');
(async()=>{
 const real=process.env.REAL_SERVERS?JSON.parse(fs.readFileSync(process.env.REAL_SERVERS,'utf8')):null;
 const url=real?real.desktop+'&browser-test=1':'http://127.0.0.1:18765/desktop.html?warning_url=http%3A%2F%2F127.0.0.1%3A18765%2F&browser-test=1';
 const out=process.env.REAL_OUTPUT,records=[],inventory={},errors=[],external=[];
 const browser=await chromium.launch({channel:'msedge',headless:true});
 const page=await browser.newPage({viewport:{width:1700,height:1100}});
 page.on('pageerror',e=>errors.push(e.message));page.on('dialog',d=>d.accept());
 if(real)await page.route('**/*',route=>{
   const u=new URL(route.request().url());
   if(['http:','https:'].includes(u.protocol)&&!['127.0.0.1','localhost'].includes(u.hostname)){
     external.push(u.origin);return route.abort();
   }return route.continue();
 });
 const warning=page.frameLocator('#moduleFrame');
 const tasks=[
 ['ribbonPreprocess','processing','点云预处理'],['ribbonGround','processing','地面点去除'],
 ['ribbonWire','processing','导线重建'],['ribbonTrees','processing','单木分割'],
 ['ribbonReview','review'],['ribbonData','warning','#dataPage'],
 ['openWarningModule','warning','#inputPage'],['ribbonExternalBatch','warning','#batchEntry'],
 ['ribbonTraining','warning','#trainingPage'],['ribbonHistory','warning','#historyPage'],
 ['ribbonTwin','twin'],['exportWorkOrder','warning','#resultPage']];
 async function verify(task){
   if(task[1]==='processing')await page.locator('#stageHost').filter({hasText:task[2]}).waitFor();
   else if(task[1]==='review'){
     await page.locator('#viewerWrap').waitFor({state:'visible'});
     if(real)await page.frameLocator('#viewer').locator('#treetable tbody tr').first().waitFor({timeout:120000});
     else await page.frameLocator('#viewer').locator('#fixtureTitle').filter({hasText:'/step2'}).waitFor();
   }else if(task[1]==='twin'){
     await page.locator('#auxiliaryFrame').waitFor({state:'visible'});
     if(real)await page.frameLocator('#auxiliaryFrame').locator('#futureBody:not(.hidden)').waitFor({timeout:120000});
     else await page.frameLocator('#auxiliaryFrame').locator('#fixtureTitle').waitFor();
   }else await warning.locator(task[2]).waitFor({state:'visible'});
   await page.waitForFunction(id=>document.querySelector('.ribbon-buttons .active-task')?.id===id,task[0]);
 }
 async function select(task){await page.locator('#'+task[0]).click();await verify(task);}
 async function capture(task){
   const selector='button,input[type="button"],input[type="submit"],a[href]';
   const root=task[1]==='warning'?warning:task[1]==='review'?page.frameLocator('#viewer'):task[1]==='twin'?page.frameLocator('#auxiliaryFrame'):page;
   inventory[task[0]]=await root.locator(selector).evaluateAll(ns=>ns.filter(n=>n.getBoundingClientRect().width&&n.getBoundingClientRect().height).map(n=>({id:n.id,text:n.textContent.trim().slice(0,80),disabled:!!n.disabled})));
 }
 try{
   await page.goto(url);await page.locator('#stageHost, #viewerWrap').first().waitFor();
   await page.waitForFunction(()=>!document.querySelector('#footerState').textContent.includes('正在连接'));
   // Seed one REAL warning computation, preserving it across task switches.
   await page.locator('#openWarningModule').click();await warning.locator('#singleMode').click();
   await warning.locator('#height').fill('10.25');await warning.locator('#clearance_distance').fill('6');
   await warning.locator('#generateReport').click();await warning.locator('#resultPage').waitFor({state:'visible',timeout:120000});
   for(const source of tasks){
     await select(source);await capture(source);
     for(const target of tasks){
       if(source[0]===target[0])continue;
       const start=Date.now();
       await select(target);
       await page.locator('#navigateBack:not([disabled])').click();await verify(source);
       // Continue forward after Back, then Back again; catches duplicate route
       // pushes, expired snapshots, and polling that overwrites restored pages.
       await select(target);await page.locator('#navigateBack:not([disabled])').click();await verify(source);
       records.push({source:source[0],target:target[0],backTo:source[0],continued:true,passed:true,ms:Date.now()-start});
       console.log(JSON.stringify(records.at(-1)));
     }
   }
   assert.deepEqual(errors,[]);assert.deepEqual(external,[]);
   await page.screenshot({path:out+'/navigation_matrix.png'});
 }finally{
   fs.writeFileSync(out+'/navigation_matrix.json',JSON.stringify({basis:real?'real_pointcloud_project':'navigation_fixture',expectedOrderedPairs:132,tested:records.length,records,inventory,pageErrors:errors,external},null,2));
   await browser.close();
 }
})().catch(e=>{console.error(e);process.exitCode=1});
