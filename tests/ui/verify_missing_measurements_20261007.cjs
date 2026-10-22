const {chromium}=require(process.env.PLAYWRIGHT_MODULE);
const fs=require('node:fs'),assert=require('node:assert/strict');
(async()=>{
 const out=process.env.REAL_OUTPUT, errors=[], checks=[];
 const browser=await chromium.launch({channel:'msedge',headless:true});
 const page=await browser.newPage({viewport:{width:1700,height:1100},acceptDownloads:true});
 page.on('pageerror',e=>errors.push(e.message));page.on('dialog',d=>d.accept());
 const base='http://127.0.0.1:18766/';
 async function context(){return (await page.request.get(base+'integrated/context')).json()}
 await page.request.post(base+'integrated/measurements',{data:{tree_key:'fixture-1',manual_diameter_cm:null}});
 await page.request.post(base+'integrated/measurements',{data:{tree_key:'fixture-1',manual_fields:{height:null,crown_width:null,horizontal_distance:null,clearance_distance:null}}});
 try{
  await page.goto(base+'?workspace=single&embedded=1');
  assert.equal(await page.locator('#suiteExcel').count(),0);
  await page.locator('#integratedMode').click();
  await page.locator('#integratedTree option').nth(2).waitFor({state:'attached'});
  await page.locator('#integratedTree').selectOption('fixture-1');
  const dbh=page.locator('[data-manual-dbh-key="fixture-1"]');
  assert.equal(await dbh.inputValue(),'未获取');
  assert.equal(await dbh.getAttribute('readonly'),null);
  assert.equal(await page.locator('[data-manual-geometry-key]').count(),4);
  assert.equal(await page.locator('[data-measurement-field="vertical_distance"]').count(),0);
  await dbh.fill('23.4');
  for(const [key,value] of Object.entries({height:11,crown_width:3,horizontal_distance:4,clearance_distance:5}))
   await page.locator(`[data-measurement-field="${key}"]`).fill(String(value));
  await page.locator('#saveManualDbh').click();
  await page.locator('#manualDbhStatus').filter({hasText:'已保存在'}).waitFor();
  const saved=(await context()).trees[0];
  assert.equal(saved.diameter,23.4);assert.equal(saved.height,11);
  assert.equal(saved.point_cloud_measurements.height,null);assert.equal(saved.vertical_distance,3);
  assert.equal(saved.measurement_sources.height,'MANUAL_FIELD_MEASUREMENT');
  checks.push('缺失胸径和四项几何值真实填写保存，不覆盖原始垂直距离');
  await page.reload();await page.locator('#integratedMode').click();
  await page.locator('#integratedTree option').nth(2).waitFor({state:'attached'});
  await page.locator('#integratedTree').selectOption('fixture-1');
  assert.equal(await dbh.inputValue(),'23.4');
  assert.equal(await page.locator('[data-measurement-field="height"]').inputValue(),'11');
  await page.locator('#integratedManualWeather').click();await page.locator('#saveIntegratedWeather').click();
  await page.locator('#integratedGenerate').click();await page.locator('#resultPage').waitFor({state:'visible',timeout:60000});
  const report=await page.evaluate(()=>predictionResultsByMode.formula);
  assert.equal(report.input_snapshot.height,11);assert.equal(report.dbh_input.source,'MANUAL_FIELD_MEASUREMENT');
  assert.ok(report.integrated_source.manual_fields.includes('height'));
  await page.locator('#backToInputPage').click();assert.equal(await dbh.inputValue(),'23.4');
  await page.screenshot({path:out+'/缺失测量人工补填.png',fullPage:true});
  checks.push('重开与预测返回保留人工实测值，报告标明人工来源');
  // Polling may replace the object while a save is in flight. Stable scope
  // remains the same, so both the save and refreshed context must stay valid.
  await page.route('**/integrated/measurements',async route=>{
   const response=await route.fetch();
   await page.evaluate(()=>{integratedContext={...integratedContext,trees:integratedContext.trees.map(t=>({...t}))}});
   await route.fulfill({response});
  });
  await dbh.fill('25.7');await page.locator('#saveManualDbh').click();
  await page.locator('#manualDbhStatus').filter({hasText:'已保存在'}).waitFor();
  assert.equal(await page.evaluate(()=>integratedContext.trees[0].manual_diameter_cm),25.7);
  await page.unroute('**/integrated/measurements');
  checks.push('同一项目自动刷新上下文不阻止保存，并更新最新上下文');
  // Model a Back/project switch while the real save request is in flight.
  // The old response must not mutate the newly selected context or its draft.
  await page.route('**/integrated/measurements',async route=>{
   const response=await route.fetch();
   await page.evaluate(()=>{globalThis.auditNewContext={...integratedContext,project_root:'ANOTHER_TEST_PROJECT',trees:integratedContext.trees.map(t=>({...t,manual_diameter_cm:null,diameter:null}))};integratedContext=globalThis.auditNewContext});
   await route.fulfill({response});
  });
  await dbh.fill('27.8');await page.locator('#saveManualDbh').click();
  await page.waitForFunction(()=>document.querySelector('#saveManualDbh')?.disabled===false);
  assert.equal(await page.evaluate(()=>auditNewContext.trees[0].manual_diameter_cm),null);
  await page.unroute('**/integrated/measurements');
  checks.push('保存期间切换项目，旧响应不串写新项目输入');
  await page.goto(base+'?workspace=batch&embedded=1');
  await page.locator('#batchFile').setInputFiles({name:'缺失字段.csv',mimeType:'text/csv',buffer:Buffer.from('tree_id,species,age,height,diameter,crown_width,voltage_level_kv,clearance_distance\nEXT-1,杨树,20,No,No,3,110,5\nEXT-2,松树,30,12,No,4,110,No\n')});
  await page.locator('#loadBatchData').click();await page.locator('#batchParseStatus').filter({hasText:'解析完成'}).waitFor();
  const rows=page.locator('#externalBatchTableBody tr');assert.equal(await rows.count(),2);
  await rows.nth(0).locator('[data-external-field="height"]').fill('11');
  await rows.nth(0).locator('[data-external-field="diameter"]').fill('24.5');
  await rows.nth(1).locator('[data-external-field="clearance_distance"]').fill('6');
  await page.locator('#loadExternalBatchTable').click();await page.locator('#runBatchPrediction').waitFor();
  assert.equal(await page.locator('#runBatchPrediction').isEnabled(),true);
  await page.locator('#integratedManualWeather').click();await page.locator('#saveIntegratedWeather').click();
  await page.locator('#runBatchPrediction').click();
  await page.locator('#batchResult').filter({hasText:'已完成 2 / 2'}).waitFor({timeout:60000});
  await page.locator('#batchResultFormat').selectOption('json');
  const downloadPromise=page.waitForEvent('download');await page.locator('#downloadBatchResults').click();
  const download=await downloadPromise;await download.saveAs(out+'/补填外部预测.json');
  assert.ok(JSON.parse(fs.readFileSync(out+'/补填外部预测.json','utf8')));
  await page.screenshot({path:out+'/外部缺失文件补填预测.png',fullPage:true});
  checks.push('外部CSV独立导入，错误行不丢失，补填后2/2预测且JSON实际导出');
  assert.deepEqual(errors,[]);checks.push('页面脚本无异常，顶部重复导出不存在');
  fs.writeFileSync(out+'/missing_measurement_checks.json',JSON.stringify({checks,errors},null,2));
  console.log(JSON.stringify({checks,errors}));
 }finally{await browser.close()}
})().catch(e=>{console.error(e);process.exit(1)});
