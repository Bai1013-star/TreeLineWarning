const {chromium}=require(process.env.PLAYWRIGHT_MODULE);
const fs=require('node:fs'),assert=require('node:assert/strict');
(async()=>{
 const urls=JSON.parse(fs.readFileSync(process.env.REAL_SERVERS,'utf8'));
 const browser=await chromium.launch({channel:'msedge',headless:true});
 const page=await browser.newPage({viewport:{width:1600,height:1100}});page.on('dialog',d=>d.accept());
 try{
  await page.goto(urls.processing+'step2');
  await page.waitForFunction(()=>window.TreeLineTreeReviewApp?.state.treeMap.size>0,{},{timeout:120000});
  const originalCount=await page.locator('#treetable tbody tr').count();
  assert.ok(originalCount>=2);
  const sum=()=>page.evaluate(()=>[...window.TreeLineTreeReviewApp.state.treeMap.values()].reduce((n,t)=>n+t.point_count,0));
  const before=await sum();
  // Selecting two rows and clicking merge now submits directly after confirmation.
  await page.locator('#treetable tbody tr').nth(0).click();await page.locator('#treetable tbody tr').nth(1).click({modifiers:['Control']});await page.locator('#btn-merge').click();
  await page.waitForFunction(n=>window.TreeLineTreeReviewApp.state.treeMap.size===n&&!window.TreeLineTreeReviewApp.state.editingBusy,originalCount-1,{timeout:180000});
  const merged=await page.evaluate(()=>[...window.TreeLineTreeReviewApp.state.treeMap.values()].find(t=>t.merged)?.tree_id);
  assert.ok(merged);await page.locator('#btn-selection-clear').click();await page.locator(`#treetable tr[data-tid="${merged}"]`).click();await page.locator('#btn-split').click();await page.locator('#btn-edit-execute').click();
  await page.waitForFunction(n=>window.TreeLineTreeReviewApp.state.treeMap.size===n&&!window.TreeLineTreeReviewApp.state.editingBusy,originalCount,{timeout:180000});
  assert.equal(await sum(),before);
  await page.locator('#btn-undo').click();await page.waitForFunction(n=>window.TreeLineTreeReviewApp.state.treeMap.size===n&&!window.TreeLineTreeReviewApp.state.editingBusy,originalCount-1,{timeout:180000});
  await page.locator('#btn-undo').click();await page.waitForFunction(n=>window.TreeLineTreeReviewApp.state.treeMap.size===n&&!window.TreeLineTreeReviewApp.state.editingBusy,originalCount,{timeout:180000});
  assert.equal(await sum(),before);
  const restoredCount=await page.locator('#treetable tbody tr').count();
  assert.equal(restoredCount,originalCount);
  await page.reload();await page.waitForFunction(n=>window.TreeLineTreeReviewApp?.state.treeMap.size===n,originalCount,{timeout:180000});
  assert.equal(await sum(),before);
  const record={passed:true,originalCount,restoredCount,fullResolutionMembers:before,splitPreservedEveryPoint:true,undoRestored:true,reloadRestored:true,scope:'仅独立验收项目，先合并两棵真实候选树，再测试分开按钮和两次撤销；不改变用户项目'};
  fs.writeFileSync(process.env.REAL_OUTPUT+'/split_checks.json',JSON.stringify(record,null,2));console.log(JSON.stringify(record));
  await page.locator('#btn-confirm').click();await page.waitForURL('**/step3',{timeout:600000});console.log('独立验收项目已重新定稿');
 }finally{await browser.close()}
})().catch(e=>{console.error(e);process.exitCode=1});
