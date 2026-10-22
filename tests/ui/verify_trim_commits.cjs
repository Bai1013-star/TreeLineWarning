const {chromium}=require(process.env.PLAYWRIGHT_MODULE);
const fs=require('node:fs'),assert=require('node:assert/strict');
(async()=>{
 const urls=JSON.parse(fs.readFileSync(process.env.REAL_SERVERS,'utf8'));
 const browser=await chromium.launch({channel:'msedge',headless:true});
 const page=await browser.newPage({viewport:{width:1800,height:1300}}),checks=[];
 page.on('dialog',d=>d.accept());
 const sum=()=>page.evaluate(()=>[...window.TreeLineTreeReviewApp.state.treeMap.values()].reduce((n,t)=>n+t.point_count,0));
 const idle=()=>page.waitForFunction(()=>!window.TreeLineTreeReviewApp.state.editingBusy&&!window.TreeLineTreeReviewApp.state.mutation,{},{timeout:180000});
 const point=()=>page.evaluate(()=>{
  const v=window.TreeLineTreeReviewApp.state.viewer,r=v.canvasRect();
  for(let i=0;i<v.treeXYZ.length;i++){
   if(!v.treeVisible[i])continue;
   const p=new v.THREE.Vector3(...v.treeXYZ[i].map((x,j)=>x-v.origin[j])).project(v.camera);
   const x=r.left+(p.x+1)*r.width/2,y=r.top+(1-p.y)*r.height/2;
   if(p.z<-1||p.z>1||x<r.left+40||x>r.right-40||y<r.top+90||y>r.bottom-40)continue;
   if(document.elementFromPoint(x,y)!==v.canvas)continue;
   const hit=v.hitAt(x,y);if(hit)return{x,y,treeId:hit.treeId};
  }throw Error('No visible clickable point');
 });
 try{
  await page.goto(urls.processing+'step2');
  await page.waitForFunction(()=>window.TreeLineTreeReviewApp?.state.treeMap.size>0,{},{timeout:120000});
  await page.locator('#toggle-labels').uncheck();await page.locator('#btn-view-top').click();
  const before=await sum();
  await page.locator('#btn-trim-component').click();
  const p=await point();await page.mouse.click(p.x,p.y);await page.locator('#btn-edit-execute').click();
  await page.waitForFunction(()=>window.TreeLineTreeReviewApp.state.activePreview,{},{timeout:180000});
  await idle();assert.equal(await sum(),before);
  await page.locator('#btn-trim-confirm').click();await idle();
  const afterComponent=await sum();assert.ok(afterComponent<before);
  await page.locator('#btn-undo').click();await idle();assert.equal(await sum(),before);
  checks.push({name:'团块精确预览、确认剔除、撤销恢复',before,after:afterComponent,restored:await sum()});
  const q=await point();await page.locator('#btn-trim-lasso').click();
  await page.mouse.move(q.x-18,q.y-18);await page.mouse.down();
  await page.mouse.move(q.x+18,q.y-18,{steps:5});await page.mouse.move(q.x+18,q.y+18,{steps:5});
  await page.mouse.move(q.x-18,q.y+18,{steps:5});await page.mouse.move(q.x-18,q.y-18,{steps:5});await page.mouse.up();
  await page.waitForFunction(()=>window.TreeLineTreeReviewApp.state.activePreview,{},{timeout:180000});
  await idle();assert.equal(await sum(),before);
  await page.locator('#btn-trim-confirm').click();await idle();
  const afterLasso=await sum();assert.ok(afterLasso<before);
  await page.locator('#btn-undo').click();await idle();assert.equal(await sum(),before);
  checks.push({name:'鼠标套索、完整点云精确预览、确认、撤销',before,after:afterLasso,restored:await sum()});
  fs.writeFileSync(process.env.REAL_OUTPUT+'/trim_commit_checks_20261001.json',JSON.stringify(checks,null,2));
  console.log(JSON.stringify(checks));
  await page.locator('#btn-confirm').click();await page.waitForURL('**/step3',{timeout:600000});
 }finally{await browser.close()}
})().catch(e=>{console.error(e);process.exitCode=1});
