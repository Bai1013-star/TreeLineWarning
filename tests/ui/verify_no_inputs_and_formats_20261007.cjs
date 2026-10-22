const {chromium}=require(process.env.PLAYWRIGHT_MODULE);
const fs=require('node:fs'),assert=require('node:assert/strict');
(async()=>{
 const browser=await chromium.launch({channel:'msedge',headless:true});
 const page=await browser.newPage({acceptDownloads:true}),checks=[],errors=[];
 page.on('pageerror',e=>errors.push(e.message));page.on('dialog',d=>d.accept());
 const out=process.env.REAL_OUTPUT,base='http://127.0.0.1:18765/';
 try{
  await page.route('**/integrated/context',r=>r.fulfill({status:404,body:'{}'}));
  await page.goto(base+'?workspace=single');
  await page.locator('#use_diameter').check();assert.equal(await page.locator('#diameter').inputValue(),'No');
  await page.locator('#generateReport').click();await page.locator('#resultPage').waitFor({state:'visible',timeout:120000});
  const result=await page.evaluate(()=>predictionResultsByMode.formula);
  assert.equal(result.dbh_input.value_cm,null);assert.equal(result.input_snapshot.diameter,null);assert.equal(result.selected_target_prediction.predicted_diameter,null);
  assert.match(await page.locator('#reportOutput').innerText(),/No/);
  const downloaded=page.waitForEvent('download');await page.locator('#saveWorkOrder').click();await(await downloaded).saveAs(out+'/standalone_No_work_order.html');
  checks.push({name:'手动输入 No、预测及工单保存',passed:true});
  await page.goto(base+'?workspace=batch');
  for(const format of ['xlsx','csv','tsv','txt','json']){
   const columns=['tree_id','species','age','height','diameter','crown_width','voltage_level_kv','horizontal_distance','vertical_distance','clearance_distance'];
   const rows=[[99001,'杨树',20,10,'No',3,110,4,3,5],[99002,'油松',30,11,24.6,2,110,4,3,5]];
   const response=await page.request.post(base+'exports/table',{data:{format,columns,rows}});assert.ok(response.ok(),await response.text());
   await page.locator('#batchFile').setInputFiles({name:'No_with_measured.'+format,mimeType:'application/octet-stream',buffer:await response.body()});
   await page.locator('#loadBatchData').click();await page.locator('#runBatchPrediction:not([disabled])').waitFor({timeout:120000});
   assert.equal(await page.locator('#batchPreview tbody tr').count(),2);
   assert.equal(await page.evaluate(()=>batchRecords[0].record.diameter),undefined);
   assert.equal(await page.evaluate(()=>batchRecords[1].record.diameter),24.6);
   checks.push({name:'外部 '+format+' 导出再导入，No 保留缺失语义',passed:true});
  }
  await page.locator('#integratedManualWeather').click();await page.locator('#saveIntegratedWeather').click();
  await page.locator('#runBatchPrediction').click();await page.locator('#downloadBatchResults').waitFor({state:'visible',timeout:120000});
  assert.equal(await page.locator('#batchResult td.ok').count(),2);
  assert.deepEqual(errors,[]);
 }finally{fs.writeFileSync(out+'/No_inputs_formats_checks.json',JSON.stringify({checks,errors},null,2));await browser.close()}
 console.log(JSON.stringify({checks,errors}));
})().catch(e=>{console.error(e);process.exitCode=1});
