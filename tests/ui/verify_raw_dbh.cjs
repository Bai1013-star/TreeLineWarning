const {chromium}=require(process.env.PLAYWRIGHT_MODULE);
const fs=require('node:fs'),assert=require('node:assert/strict');
(async()=>{
 const urls=JSON.parse(fs.readFileSync(process.env.REAL_SERVERS,'utf8'));
 const browser=await chromium.launch({channel:'msedge',headless:true});
 const page=await browser.newPage({viewport:{width:1600,height:1100},acceptDownloads:true});
 const workspace=await(await page.request.get(urls.processing+'api/desktop/workspace')).json();
 const count=workspace.batch.final_tree_count;
 const errors=[],checks=[]; page.on('pageerror',e=>errors.push(e.message));
 await page.goto(urls.processing+'dbh-review');
 await page.waitForFunction(n=>window.TreeLineDbhReview?.state.trees.length===n,count,{timeout:300000});
 const trees=await page.evaluate(()=>window.TreeLineDbhReview.state.trees);
 assert.equal(await page.locator('#tree-list button:not([disabled])').count(),count);
 const measured=trees.filter(t=>t.dbh_cm!=null);
 assert.equal(measured.length,count,'每棵树都必须从原始成员点云最低点+1.30 m取得胸径');
 for(const tree of measured){
   await page.locator(`[data-tree-id="${tree.persistent_tree_id}"]`).click();
   await page.locator('#btn-prepare').click();
   await page.waitForFunction(()=>window.TreeLineDbhReview.state.rendererReady,{},{timeout:300000});
   const data=await page.evaluate(()=>{
     const s=window.TreeLineDbhReview.state,r=s.review;
     return {id:r.display_tree_id,dbh:r.current_dbh_cm,readOnly:r.read_only,
       ground:r.ground_references[0],minHag:Math.min(...r.display.hag_m),
       maxHag:Math.max(...r.display.hag_m),context:document.querySelector('#dbh-context-label').textContent,
       points:r.display.xyz.length};
   });
   assert.equal(data.readOnly,true);assert.ok(data.minHag<.10);
   assert.ok(Math.abs(data.ground.breast_height_z_absolute_m-data.ground.ground_z_absolute_m-1.3)<1e-8);
   assert.ok(Math.abs(data.ground.ground_z_absolute_m-tree.dbh_ground_z_absolute_m)<1e-8);
   assert.equal(await page.locator('#btn-confirm').isDisabled(),true);
   for(const id of ['btn-view-reset','btn-view-top','btn-view-side'])await page.locator('#'+id).click();
   await page.screenshot({path:process.env.REAL_OUTPUT+`/dbh_1p30_${data.id}.png`,fullPage:true});
   checks.push(data);console.log(JSON.stringify(data));
 }
 checks.push({allTreesMeasured:`${measured.length}/${count} original-member lowest+1.30 m DBH values`});
 const projection=await(await page.request.get(urls.processing+'api/final',{timeout:180000})).json();
 fs.writeFileSync(process.env.REAL_OUTPUT+'/dbh_v5_final_projection.json',JSON.stringify(projection,null,2));
 const before=JSON.stringify(trees.map(t=>[t.persistent_tree_id,t.dbh_cm]));
 await page.reload();await page.waitForFunction(n=>window.TreeLineDbhReview?.state.trees.length===n,count,{timeout:300000});
 const after=await page.evaluate(()=>JSON.stringify(window.TreeLineDbhReview.state.trees.map(t=>[t.persistent_tree_id,t.dbh_cm])));
 assert.equal(before,after);checks.push({reopen:`${count} rows and ${measured.length} actual-slab values preserved`});
 assert.deepEqual(errors,[]);checks.push({pageErrors:errors});
 fs.writeFileSync(process.env.REAL_OUTPUT+'/dbh_v5_ui_checks.json',JSON.stringify(checks,null,2));
 await browser.close();
})().catch(e=>{console.error(e);process.exitCode=1});
