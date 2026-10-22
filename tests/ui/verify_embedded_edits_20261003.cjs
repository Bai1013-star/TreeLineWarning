// Exercise the production desktop shell, not the standalone /step2 page.
const {chromium}=require(process.env.PLAYWRIGHT_MODULE);
const fs=require('node:fs'),assert=require('node:assert/strict');
(async()=>{
 const urls=JSON.parse(fs.readFileSync(process.env.REAL_SERVERS,'utf8'));
 const browser=await chromium.launch({channel:'msedge',headless:true});
 const page=await browser.newPage({viewport:{width:1600,height:980}});
 const checks=[],errors=[],consoleErrors=[];let dialogs=0,accept=true,commit=null;
 page.on('response',async r=>{if(new URL(r.url()).pathname.startsWith('/api/editor/mutation')){try{const v=await r.json();if(v.kind==='TRIM_COMMIT'||v.result?.display_delta)commit=v;}catch{}}});
 page.on('dialog',async d=>{dialogs++;await (accept?d.accept():d.dismiss());});
 page.on('pageerror',e=>errors.push(e.message));
 page.on('console',m=>{if(m.type()==='error')consoleErrors.push(m.text());});
 await page.route('**/*',r=>{const u=new URL(r.request().url());return ['127.0.0.1','localhost'].includes(u.hostname)||['data:','blob:'].includes(u.protocol)?r.continue():r.abort();});
 try{
  await page.goto(urls.desktop);
  await page.waitForFunction(()=>!['未打开项目',''].includes(document.querySelector('#projectName').textContent),{},{timeout:120000});
  await page.locator('#ribbonReview').click();
  const f=page.frameLocator('#viewer');
  await f.locator('#treetable tbody tr').first().waitFor({timeout:180000});
  const frame=page.frames().find(f=>new URL(f.url()).pathname==='/step2');
  const stats=()=>frame.evaluate(()=>{const s=window.TreeLineTreeReviewApp.state;return {trees:s.treeMap.size,points:[...s.treeMap.values()].reduce((n,t)=>n+t.point_count,0),journal:s.committedJournalCount};});
  const idle=()=>frame.waitForFunction(()=>{const s=window.TreeLineTreeReviewApp.state;return !s.editingBusy&&!s.mutation;},{},{timeout:240000});
  const count=async n=>{await frame.waitForFunction(n=>{const s=window.TreeLineTreeReviewApp.state;return !s.editingBusy&&!s.mutation&&(s.treeMap.size===n||document.querySelector('#editmsg').textContent.startsWith('失败'));},n,{timeout:240000});assert.equal((await stats()).trees,n,await f.locator('#editmsg').textContent());};
  const note=(name,detail)=>{checks.push({name,detail});console.log(JSON.stringify({name,detail}));};
  await idle();
  if(process.env.EXPECT_BLOCKED){
   const before=await stats();await f.locator('#treetable tbody tr').nth(0).click();await f.locator('#treetable tbody tr').nth(1).click({modifiers:['Control']});
   await f.locator('#btn-merge').click();await idle();assert.deepEqual(await stats(),before);assert.equal(dialogs,0);
   assert.ok(consoleErrors.some(e=>e.includes('allow-modals')));
   note('复现完整主界面合并被 sandbox 禁止确认框',consoleErrors);
  }else{
   while((await stats()).journal>0){await f.locator('#btn-undo').click();await idle();}
   const before=await stats(),rows=f.locator('#treetable tbody tr');
   await rows.nth(0).click();await rows.nth(1).click({modifiers:['Control']});accept=false;
   await f.locator('#btn-merge').click();await idle();assert.deepEqual(await stats(),before);accept=true;
   note('完整主界面合并取消不修改',true);
   await f.locator('#btn-edit-execute').click();await count(before.trees-1);assert.equal((await stats()).points,before.points);
   note('完整主界面执行合并实际提交',await stats());
   const merged=await frame.evaluate(()=>[...window.TreeLineTreeReviewApp.state.treeMap.values()].find(t=>t.merged)?.tree_id);
   await f.locator(`#treetable tr[data-tid="${merged}"]`).click();await f.locator('#btn-split').click();
   await f.locator('#btn-edit-execute').click();await count(before.trees);assert.equal((await stats()).points,before.points);
   const splitShown=await frame.evaluate(()=>{const s=window.TreeLineTreeReviewApp.state,v=s.viewer;const shown=new Set(v.treeIds.filter((id,i)=>v.treeVisible[i]));return [...s.treeMap.keys()].every(id=>shown.has(id));});
   assert.equal(splitShown,true,'both split children must actually appear in GPU ownership');
   note('完整主界面自动分开不丢点',await stats());
   await f.locator('#btn-undo').click();await count(before.trees-1);await f.locator('#btn-undo').click();await count(before.trees);
   assert.deepEqual(await stats(),before);
   // Delete a visible object that is NOT selected. The camera stays fixed.
   await rows.nth(0).click();await f.locator('#toggle-labels').uncheck();await f.locator('#btn-view-side').click();
   const selectedId=Number(await rows.nth(0).getAttribute('data-tid'));
   await f.locator('#btn-trim-lasso').click();
   const target=await frame.evaluate(selectedId=>{
    const v=window.TreeLineTreeReviewApp.state.viewer,r=v.canvasRect();
    for(let i=0;i<v.treeXYZ.length;i++){
     if(!v.treeVisible[i]||Number(v.treeIds[i])===selectedId)continue;
     const p=new v.THREE.Vector3(...v.treeXYZ[i].map((x,j)=>x-v.origin[j])).project(v.camera);
     const x=r.left+(p.x+1)*r.width/2,y=r.top+(1-p.y)*r.height/2;
     if(p.z<-1||p.z>1||x<r.left+40||x>r.right-40||y<r.top+90||y>r.bottom-40)continue;
     if(document.elementFromPoint(x,y)?.classList.contains('lasso-overlay'))return {x,y,id:Number(v.treeIds[i])};
    }throw Error('no visible unselected point');
   },selectedId);
   const frameBox=await page.locator('#viewer').boundingBox();
   const polygon=[[target.x-24,target.y-24],[target.x+24,target.y-24],[target.x+24,target.y+24],[target.x-24,target.y+24],[target.x-24,target.y-24]];
   const ndc=await frame.evaluate(polygon=>{const r=document.querySelector('#tree-review-stage').getBoundingClientRect();return polygon.map(p=>[2*(p[0]-r.left)/r.width-1,1-2*(p[1]-r.top)/r.height]);},polygon);
   await page.mouse.move(frameBox.x+polygon[0][0],frameBox.y+polygon[0][1]);await page.mouse.down();
   for(const p of polygon.slice(1))await page.mouse.move(frameBox.x+p[0],frameBox.y+p[1],{steps:12});await page.mouse.up();
   console.log('draw',JSON.stringify({frameBox,target,mode:await frame.evaluate(()=>window.TreeLineTreeReviewApp.state.trimMode),message:await f.locator('#editmsg').textContent()}));
   await page.screenshot({path:process.env.REAL_OUTPUT+'/embedded_lasso_drawing.png'});
   await frame.waitForFunction(()=>window.TreeLineTreeReviewApp.state.activePreview,{},{timeout:240000});await idle();
   const preview=await frame.evaluate(()=>window.TreeLineTreeReviewApp.state.activePreview);
   assert.ok(preview.tree_ids.includes(target.id));assert.ok(preview.point_count>0);
   // The preview confirmation must have scrolled into the panel's visible area.
   await frame.waitForFunction(()=>{const e=document.querySelector('#btn-trim-confirm'),r=e.getBoundingClientRect(),p=e.closest('.side').getBoundingClientRect();return r.top>=p.top&&r.bottom<=p.bottom;},{},{timeout:10000});
   await f.locator('#btn-trim-confirm').click();await idle();assert.equal((await stats()).points,before.points-preview.point_count);
   const matrix=await frame.evaluate(()=>window.TreeLineTreeReviewApp.state.viewer.currentViewProjection());
   const hits=await frame.evaluate(({matrix,polygon})=>{
    const v=window.TreeLineTreeReviewApp.state.viewer;
    function inside(x,y){let c=false;for(let i=0,j=polygon.length-1;i<polygon.length;j=i++){const a=polygon[i],b=polygon[j];if((a[1]>y)!==(b[1]>y)&&x<(b[0]-a[0])*(y-a[1])/(b[1]-a[1])+a[0])c=!c;}return c;}
    const m=new v.THREE.Matrix4().set(...matrix.flat());let hits=0;
    for(let i=0;i<v.treeXYZ.length;i++){if(!v.treeVisible[i])continue;const p=new v.THREE.Vector3(...v.treeXYZ[i]).applyMatrix4(m);if(p.z>=-1&&p.z<=1&&inside(p.x,p.y))hits++;}return hits;
   },{matrix,polygon:ndc});
   fs.writeFileSync(process.env.REAL_OUTPUT+'/lasso_debug.json',JSON.stringify({preview,commit,ndc,matrix,stats:await stats(),hits,visible:await frame.evaluate(()=>{const v=window.TreeLineTreeReviewApp.state.viewer;return {total:v.treeXYZ.length,visible:v.treeVisible.filter(Boolean).length,base:v.displayBaseId,revision:v.editorRevision,indices:v.treeOriginalIndices.slice(0,10)};})},null,2));
   assert.equal(hits,0,'committed display must have no editable points inside exact polygon');
   note('主界面实际套索删除未选对象、圈内显示清空', {removed:preview.point_count,hits,selectedId,targetId:target.id});
   await page.screenshot({path:process.env.REAL_OUTPUT+'/embedded_lasso_deleted.png'});
   await page.locator('#refreshViewer').click();
   await page.waitForFunction(()=>{try{const f=document.querySelector('#viewer');return f.contentWindow.location.href===f.src&&f.contentWindow.TreeLineTreeReviewApp?.state.viewer?.treeXYZ?.length>0;}catch{return false;}},{},{timeout:180000});
   const reopened=page.frames().find(f=>new URL(f.url()).pathname==='/step2');
   await reopened.waitForFunction(()=>window.TreeLineTreeReviewApp?.state.viewer?.treeXYZ?.length>0,{},{timeout:180000});
   assert.equal(await reopened.evaluate(()=>[...window.TreeLineTreeReviewApp.state.treeMap.values()].reduce((n,t)=>n+t.point_count,0)),before.points-preview.point_count);
   await page.frameLocator('#viewer').locator('#btn-undo').click();
   await reopened.waitForFunction(n=>{const s=window.TreeLineTreeReviewApp.state;return !s.editingBusy&&!s.mutation&&[...s.treeMap.values()].reduce((n,t)=>n+t.point_count,0)===n;},before.points,{timeout:240000});
   note('刷新后删除保留、撤销完整恢复',true);assert.ok(dialogs>=3);assert.deepEqual(errors,[]);
  }
  fs.writeFileSync(process.env.REAL_OUTPUT+'/'+(process.env.EXPECT_BLOCKED?'embedded_before':'embedded_after')+'.json',JSON.stringify({checks,dialogs,errors,consoleErrors},null,2));
 }finally{await browser.close();}
})().catch(e=>{console.error(e);process.exitCode=1;});
