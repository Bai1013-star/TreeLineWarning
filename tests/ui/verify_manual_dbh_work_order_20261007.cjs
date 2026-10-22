// Three bounded acceptance rounds on an existing real LAS analysis.
// Manual numbers below are explicit TEST operator input, not field observations.
const {chromium}=require(process.env.PLAYWRIGHT_MODULE);
const fs=require('node:fs'),assert=require('node:assert/strict');
(async()=>{
 const urls=JSON.parse(fs.readFileSync(process.env.REAL_SERVERS,'utf8'));
 const out=process.env.REAL_OUTPUT,checks=[],errors=[],external=[];
 const browser=await chromium.launch({channel:'msedge',headless:true});
 const page=await browser.newPage({viewport:{width:1700,height:1100},acceptDownloads:true});
 page.on('pageerror',e=>errors.push(e.message));page.on('dialog',d=>d.accept());
 await page.route('**/*',r=>{const u=new URL(r.request().url());if(['http:','https:'].includes(u.protocol)&&!['127.0.0.1','localhost'].includes(u.hostname)){external.push(u.origin);return r.abort();}return r.continue()});
 const initial=await(await page.request.get(urls.warning+'integrated/context')).json();
 assert.equal(initial.status,'READY');
 const missing=initial.trees.filter(t=>t.point_cloud_diameter==null),auto=initial.trees.filter(t=>t.point_cloud_diameter!=null);
 assert.ok(missing.length>0);assert.ok(auto.length>0);
 const snapshots=initial.trees.map(t=>[t.tree_key,t.point_cloud_diameter]);
 const w=()=>page.frameLocator('#moduleFrame');
 async function ready(){await page.goto(urls.desktop+'&browser-test=1');await page.locator('#openWarningModule').click();await w().locator('#integratedMode').click();await w().locator('#integratedTree option').nth(initial.trees.length-1).waitFor({state:'attached',timeout:180000})}
 async function json(url,body){const r=await page.request.post(urls.warning+url,{data:body});assert.ok(r.ok(),await r.text());return r.json()}
 try{
  for(let round=1;round<=3;round++){
   await ready();
   await w().locator('#integratedTree').selectOption(missing[0].tree_key);
   await w().locator('#integratedTreeCount').fill(round===1?'1':'2');await w().locator('#integratedTreeCount').blur();
   const first=w().locator('#integratedDbhFields [data-manual-dbh-key]').first();
   await first.fill('No');await w().locator('#saveManualDbh').click();
   try{await w().locator('#manualDbhStatus').filter({hasText:'已保存在'}).waitFor()}
   catch(error){console.error(await w().locator('#integratedDbhFields').innerText());throw error}
   assert.equal(await first.inputValue(),'No');
   await first.fill(String(26+round/10));
   if(round!==1){await w().locator('[data-integrated-extra-tree]').selectOption(missing[1]?.tree_key||auto[0].tree_key);const second=w().locator('#integratedDbhFields [data-manual-dbh-key]').nth(1);if(!(await second.getAttribute('readonly')))await second.fill('24.5')}
   await w().locator('#integratedObservationDate').fill('2026-10-07');await w().locator('#integratedTargetDate').fill('2027-10-07');
   await w().locator('#integratedManualWeather').click();await w().locator('#saveIntegratedWeather').click();
   await w().locator('#integratedGenerate').click();
   if(round===1)await w().locator('#resultPage').waitFor({state:'visible',timeout:180000});else{await w().locator('#integratedMultiResults [data-open-tree-result="0"]').waitFor({timeout:180000});await w().locator('#integratedMultiResults [data-open-tree-result="0"]').click()}
   const report=await page.frames().find(f=>f.url().startsWith(urls.warning)).evaluate(()=>predictionResultsByMode.formula);
   assert.equal(report.dbh_input.value_cm,26+round/10);assert.equal(report.dbh_input.source,'MANUAL_FIELD_MEASUREMENT');assert.equal(report.dbh_input.point_cloud_value_cm,null);
   assert.match(await w().locator('#reportOutput').innerText(),/人工实测/);
   // Actual same-origin parent/native bridge contract, with a test-controlled path.
   await page.evaluate(()=>{window.exportCalls=[];window.pywebview={api:{save_export_file:async(...args)=>{window.exportCalls.push(args);return {ok:true,path:'D:/验收测试/自选保存位置/工单.html'}}}}});
   await w().locator('#workOrderSaveScope').selectOption(round===1?'current':'all');
   await w().locator('#saveWorkOrder').click();await page.waitForFunction(()=>window.exportCalls.length===1);
   const args=await page.evaluate(()=>window.exportCalls[0]);assert.ok(args[0].endsWith('.html'));
   const html=Buffer.from(args[1].split(',').at(-1),'base64').toString('utf8');assert.match(html,/人工实测/);assert.match(html,/26\.[123]/);assert.match(html,/<meta charset="UTF-8">/);
   assert.equal((html.match(/配电线路树线隐患智能预警工单/g)||[]).length,round===1?1:2);
   fs.writeFileSync(out+'/test_work_order_round_'+round+'.html',html);
   await w().locator('#workOrderSaveStatus').filter({hasText:'自选保存位置'}).waitFor();
   await page.evaluate(()=>{window.pywebview.api.save_export_file=async()=>({ok:false,cancelled:true})});await w().locator('#saveWorkOrder').click();await w().locator('#workOrderSaveStatus').filter({hasText:'已取消'}).waitFor();
   await page.locator('#ribbonHistory').click();await w().locator('#historyPage').waitFor({state:'visible'});await page.locator('#navigateBack').click();await w().locator('#resultPage').waitFor({state:'visible'});
   assert.match(await w().locator('#reportOutput').innerText(),/人工实测/);
   await w().locator('#backToInputPage').click();assert.equal(await first.inputValue(),String(26+round/10));
   // Reopen the page and read persisted API context, independently of drafts.
   await ready();await w().locator('#integratedTree').selectOption(missing[0].tree_key);assert.equal(await first.inputValue(),String(26+round/10));
   const reopened=await(await page.request.get(urls.warning+'integrated/context')).json();assert.deepEqual(reopened.trees.map(t=>[t.tree_key,t.point_cloud_diameter]),snapshots);
   assert.equal(reopened.trees.find(t=>t.tree_key===missing[0].tree_key).manual_diameter_cm,26+round/10);
   await page.screenshot({path:out+'/manual_dbh_round_'+round+'.png'});
   checks.push({round,existing_real_analysis:true,prediction_tree_count:round===1?1:2,manual_dbh:26+round/10,raw_unchanged:true,report_saved_bridge:true,cancel:true,back_history:true,reopen:true});
   console.log(JSON.stringify(checks.at(-1)));
  }
  // Leave audit project clean: remove test-only manual measurements.
  for(const t of missing)await json('integrated/measurements',{tree_key:t.tree_key,manual_diameter_cm:'No'});
  const raw=await(await page.request.get(urls.warning+'integrated/context')).json();assert.ok(raw.trees.filter(t=>t.point_cloud_diameter==null).every(t=>t.diameter==null));
  assert.deepEqual(errors,[]);assert.deepEqual(external,[]);
 }finally{fs.writeFileSync(out+'/manual_dbh_work_order_checks.json',JSON.stringify({checks,errors,external,coverage:'Three prediction/back/save/reopen rounds on existing real analysis; not three new full LAS imports. Test native responses are separate from real OS dialog acceptance.'},null,2));await browser.close()}
})().catch(e=>{console.error(e);process.exitCode=1});
