// Restore the single TEST merge left by the obsolete split-driver failure.
const {chromium}=require(process.env.PLAYWRIGHT_MODULE);
const fs=require('node:fs'),assert=require('node:assert/strict');
(async()=>{
 const urls=JSON.parse(fs.readFileSync(process.env.REAL_SERVERS,'utf8'));
 const browser=await chromium.launch({channel:'msedge',headless:true});
 const page=await browser.newPage({viewport:{width:1600,height:1100}});
 page.on('dialog',d=>d.accept());
 try{
  await page.goto(urls.processing+'step2');
  await page.waitForFunction(()=>window.TreeLineTreeReviewApp?.state.treeMap.size===5,{},{timeout:120000});
  const countPoints=()=>page.evaluate(()=>[...TreeLineTreeReviewApp.state.treeMap.values()].reduce((n,t)=>n+t.point_count,0));
  assert.equal(await countPoints(),3714695);
  await page.locator('#btn-undo').click();
  await page.waitForFunction(()=>TreeLineTreeReviewApp.state.treeMap.size===6&&!TreeLineTreeReviewApp.state.editingBusy,{},{timeout:180000});
  assert.equal(await countPoints(),3714695);
  const contextTimes=[],contexts=[];
  for(let i=0;i<3;i++){
   const start=Date.now();const response=await page.request.get(urls.warning+'integrated/context',{timeout:180000});
   assert.ok(response.ok());contexts.push(await response.json());contextTimes.push((Date.now()-start)/1000);
  }
  assert.equal(contexts[2].trees.length,6);assert.equal(contexts[2].finalized,false);
  assert.deepEqual(contexts[0].trees,contexts[1].trees);assert.deepEqual(contexts[1].trees,contexts[2].trees);
  assert.ok(contextTimes[2]<2);
  fs.writeFileSync(process.env.REAL_OUTPUT+'/review_cache_timings.json',JSON.stringify({passed:true,seconds:contextTimes,rows_unchanged:true,count:6,finalized:false},null,2));
  for(let i=0;i<6;i++)await page.locator('#treetable tbody tr').nth(i).click();
  await page.locator('#btn-confirm').click();await page.waitForURL('**/step3',{timeout:600000});
  const final=await (await page.request.get(urls.warning+'integrated/context',{timeout:120000})).json();
  assert.equal(final.trees.length,6);assert.equal(final.finalized,true);
  fs.writeFileSync(process.env.REAL_OUTPUT+'/baseline_restore.json',JSON.stringify({passed:true,trees:6,points:3714695,reason:'恢复仅因旧分开验收脚本失败留下的一次测试合并',originalDBH:final.trees.map(t=>[t.tree_code,t.point_cloud_diameter])},null,2));
  console.log(JSON.stringify({passed:true,trees:6,points:3714695,reviewContextSeconds:contextTimes}));
 }finally{await browser.close()}
})().catch(e=>{console.error(e);process.exit(1)});
