// Bounded real-UI checks. No claim that unbounded histories were enumerated.
const {chromium}=require(process.env.PLAYWRIGHT_MODULE);
const fs=require('node:fs'),assert=require('node:assert/strict'),crypto=require('node:crypto');
const digest=v=>crypto.createHash('sha256').update(JSON.stringify(v)).digest('hex');
(async()=>{
 const urls=JSON.parse(fs.readFileSync(process.env.REAL_SERVERS,'utf8'));
 const out=process.env.REAL_OUTPUT,checks=[],errors=[],external=[];
 const browser=await chromium.launch({channel:'msedge',headless:true});
 const page=await browser.newPage({viewport:{width:1700,height:1080}});
 page.on('pageerror',e=>errors.push(e.message));page.on('dialog',d=>d.accept());
 await page.route('**/*',r=>{const u=new URL(r.request().url());if(['http:','https:'].includes(u.protocol)&&!['127.0.0.1','localhost'].includes(u.hostname)){external.push(u.origin);return r.abort();}return r.continue();});
 const review=()=>page.frames().find(f=>new URL(f.url()).pathname==='/step2');
 const frame=()=>page.frameLocator('#viewer');
 const loaded=async()=>{await frame().locator('#treetable tbody tr').first().waitFor({timeout:180000});await review().waitForFunction(()=>{const s=window.TreeLineTreeReviewApp?.state;return s?.viewer&&s.editorDisplay&&!s.editingBusy&&!s.mutation;},{},{timeout:180000});};
 const infrastructure=async()=>review().evaluate(()=>{const s=window.TreeLineTreeReviewApp.state,d=s.editorDisplay.infrastructure;return {data:d,rendered:s.viewer.content.children.filter(c=>c.userData.role?.startsWith('PROTECTED_')).map(c=>({role:c.userData.role,positions:[...c.geometry.attributes.position.array]}))};});
 const stats=async()=>review().evaluate(()=>{const s=window.TreeLineTreeReviewApp.state;return {count:s.treeMap.size,points:[...s.treeMap.values()].reduce((n,t)=>n+t.point_count,0),operations:s.committedJournalCount};});
 const idle=async()=>review().waitForFunction(()=>{const s=window.TreeLineTreeReviewApp.state;return !s.editingBusy&&!s.mutation;},{},{timeout:240000});
 try{
  await page.goto(urls.desktop+'&browser-test=1');
  await page.waitForFunction(()=>!document.querySelector('#footerState').textContent.includes('正在连接'));
  await page.locator('#ribbonReview').click();await loaded();
  // This is a dedicated acceptance project. Earlier failed UI runs may leave
  // their own test journal; restore it before taking the zero-edit baseline.
  while((await stats()).operations>0){await frame().locator('#btn-undo').click();await idle();}
  const before=await stats(),fixed=await infrastructure(),hash=digest(fixed);
  assert.equal(fixed.data.status,'AVAILABLE');assert.equal(fixed.data.pole_count,2);
  assert.equal(fixed.rendered.filter(r=>r.role==='PROTECTED_POLE_FIT').length,2);
  assert.ok(fixed.data.protected_point_count>0);assert.equal(fixed.data.editable,false);
  checks.push({name:'原始杆线点、两根杆拟合与固定导线可见',protected:fixed.data.protected_point_count,hash});
  // The local exit control also works after a cold page load / empty history.
  await frame().locator('#btn-review-back').click();await page.locator('#viewerWrap').waitFor({state:'hidden'});
  await page.waitForTimeout(1800);assert.equal(await page.locator('#viewerWrap').isVisible(),false);
  checks.push({name:'复核退出后不会被状态刷新拉回复核页',passed:true});
  await page.locator('#ribbonReview').click();await loaded();assert.equal(digest(await infrastructure()),hash);
  const rows=frame().locator('#treetable tbody tr');
  await rows.nth(0).click();await rows.nth(1).click({modifiers:['Control']});
  await frame().locator('#btn-merge').click();await idle();
  if ((await stats()).operations===before.operations) {
    await frame().locator('#btn-edit-execute').click();await idle();
  }
  assert.equal((await stats()).count,before.count-1);assert.equal(digest(await infrastructure()),hash);
  checks.push({name:'执行合并后原始杆线及拟合 GPU 几何逐值不变',passed:true});
  await frame().locator('#btn-undo').click();await idle();assert.deepEqual(await stats(),before);assert.equal(digest(await infrastructure()),hash);
  // Cold reload preserves the fixed evidence and the undo-restored memberships.
  await page.reload();await page.locator('#ribbonReview').click();await loaded();
  assert.deepEqual(await stats(),before);assert.equal(digest(await infrastructure()),hash);
  checks.push({name:'重新打开保留杆线保护与撤销结果',passed:true});
  await frame().locator('#btn-review-back').click();await page.locator('#viewerWrap').waitFor({state:'hidden'});
  // Standalone /step2 uses the same button but returns to the main shell.
  await page.goto(urls.processing+'step2');
  await page.locator('#btn-review-back').waitFor();await page.locator('#btn-review-back').click();
  await page.waitForURL(urls.processing);checks.push({name:'独立复核页面返回主界面',passed:true});
  assert.deepEqual(errors,[]);assert.deepEqual(external,[]);
  await page.screenshot({path:out+'/review_exit.png'});
 }finally{
  fs.writeFileSync(out+'/protected_review_ui.json',JSON.stringify({checks,errors,external},null,2));await browser.close();
 }
})().catch(e=>{console.error(e);process.exitCode=1});
