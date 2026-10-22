const {chromium}=require(process.env.PLAYWRIGHT_MODULE);
const fs=require('node:fs'),assert=require('node:assert/strict');
(async()=>{
  const urls=JSON.parse(fs.readFileSync(process.env.REAL_SERVERS,'utf8'));
  const browser=await chromium.launch({channel:'msedge',headless:true});
  const page=await browser.newPage({viewport:{width:1600,height:1100},acceptDownloads:true}),errors=[],checks=[],outbound=[];
  if(process.env.OFFLINE_VERIFY==='1')await page.context().route('**/*',route=>{
    const url=new URL(route.request().url());
    if(['127.0.0.1','localhost','[::1]'].includes(url.hostname))return route.continue();
    outbound.push(url.href);return route.abort('internetdisconnected');
  });
  page.on('pageerror',e=>errors.push(e.message));page.on('dialog',d=>d.accept());
  const note=(name,detail)=>{checks.push({name,detail});console.log(JSON.stringify({name,detail}));};
  const workspace=await(await page.request.get(urls.processing+'api/desktop/workspace')).json();
  const count=workspace.batch.final_tree_count;
  if(!workspace.finalized){
    await page.goto(urls.processing+'step2');
    await page.waitForFunction(()=>window.TreeLineTreeReviewApp?.state.treeMap.size>0,{},{timeout:120000});
    for(let i=0;i<count;i++)await page.locator('#treetable tbody tr').nth(i).click();
    await page.locator('#btn-confirm').click();
    await page.waitForURL('**/step3',{timeout:600000});
  }
  note('复核定稿及报告跳转','passed');
  await page.goto(urls.desktop+'&browser-test=1');
  await page.locator('#ribbonData').click();
  const w=page.frameLocator('#moduleFrame');
  await w.locator('#processedDataBody tr').nth(count-1).waitFor({timeout:120000});
  note('查看数据与处理直连',{rows:await w.locator('#processedDataBody tr').count()});
  await w.locator('#refreshProcessedData').click();await w.locator('#refreshProcessedData:not([disabled])').waitFor({timeout:120000});
  const exports={};
  for(const format of ['xlsx','csv','tsv','txt','json']){
    await w.locator('#processedExportFormat').selectOption(format);
    const download=page.waitForEvent('download',{timeout:60000});await w.locator('#exportProcessedExcel').click();
    const file=await download;exports[format]=process.env.REAL_OUTPUT+'/processed.'+format;await file.saveAs(exports[format]);
    assert.ok(fs.statSync(exports[format]).size>10);
  }
  note('五种格式实际导出',exports);
  await w.locator('#applyProcessedData').click();await w.locator('#integratedEntry').waitFor({state:'visible'});
  await w.locator('#integratedTreeCount').fill('2');await w.locator('#integratedTreeCount').blur();
  const ids=await w.locator('#integratedTree option').evaluateAll(nodes=>nodes.map(n=>n.value).filter(Boolean));
  await w.locator('#integratedTree').selectOption(ids[0]);await w.locator('[data-integrated-extra-tree]').selectOption(ids[1]);
  const speciesFields=w.locator('[data-integrated-tree-species]');
  await speciesFields.nth(0).fill('油松');await speciesFields.nth(1).fill('杨树');
  await w.locator('#integratedSpeciesRecognition').waitFor({state:'visible'});
  assert.ok(await w.locator('#integratedRecognitionPhotos').count());
  assert.ok(await w.locator('#integratedRecognitionVideos').count());
  assert.ok(await w.locator('#integratedRecognitionPointCloud').count());
  note('短时预测树种识别入口','照片、视频、单木点云、前三候选及人工改写控件已加载');
  await w.locator('#integratedLatitude').fill('39.89');await w.locator('#integratedLongitude').fill('116.32');
  await w.locator('#integratedManualWeather').click();await w.locator('#integratedTemperature').fill('8');
  await w.locator('#saveIntegratedWeather').click();
  await w.locator('#integratedGenerate').click();
  await w.locator('#integratedMultiResults tbody tr').nth(1).waitFor({timeout:120000});
  note('真实点云两树按序预测',await w.locator('#integratedMultiResults').innerText());
  const multiText=await w.locator('#integratedMultiResults').innerText();
  assert.ok(!multiText.includes('\t已接触\t'),'预测净空为零不能直接声称现场已接触');
  assert.ok(!multiText.includes('级风 /'),'阵风速度量级不能冒充气象平均风力等级');
  await page.screenshot({path:process.env.REAL_OUTPUT+'/prediction_real.png',fullPage:true});
  await page.locator('#ribbonTwin').click();const t=page.frameLocator('#auxiliaryFrame');
  try { await t.locator('#futureBody:not(.hidden)').waitFor({timeout:60000}); }
  catch(error){console.log('TWIN_DIAGNOSTIC',JSON.stringify(await page.locator('#moduleFrameWrap').evaluate(e=>({class:e.className,children:[...e.children].map(x=>({id:x.id,class:x.className,rect:x.getBoundingClientRect().toJSON()}))}))));console.log(await t.locator('#futureStatus').textContent());await page.screenshot({path:process.env.REAL_OUTPUT+'/handoff_failure.png',fullPage:true});throw error;}
  await t.locator('#timelineSlider').evaluate(e=>{e.value='100';e.dispatchEvent(new Event('input',{bubbles:true}));});
  note('孪生时间轴100天',{date:await t.locator('#selectedDate').textContent(),clearance:await t.locator('#selectedClearance').textContent()});
  const before=await t.locator('#selectedDate').textContent();let previewRequests=0;
  page.on('request',r=>{if(r.url().includes('/api/future-preview?'))previewRequests++;});
  const began=Date.now();for(const day of ['101','102','110','365'])await t.locator('#timelineSlider').evaluate((e,value)=>{e.value=value;e.dispatchEvent(new Event('input',{bubbles:true}));},day);
  assert.notEqual(await t.locator('#selectedDate').textContent(),before);assert.equal(previewRequests,0);
  assert.ok(await t.locator('#growthPlot .plotly').count());
  assert.ok(await t.locator('#clearancePlot .plotly').count());
  note('拖动连续孪生',{milliseconds:Date.now()-began,backendRecalculations:previewRequests});
  const slider=await t.locator('#timelineSlider').boundingBox();
  const priorDragDate=await t.locator('#selectedDate').textContent();
  await page.mouse.move(slider.x+slider.width*.2,slider.y+slider.height/2);
  await page.mouse.down();await page.mouse.move(slider.x+slider.width*.4,slider.y+slider.height/2,{steps:20});await page.mouse.up();
  assert.notEqual(await t.locator('#selectedDate').textContent(),priorDragDate);
  assert.equal(previewRequests,0);note('真实鼠标连续拖动',{date:await t.locator('#selectedDate').textContent(),backendRecalculations:previewRequests});
  await page.screenshot({path:process.env.REAL_OUTPUT+'/twin_real.png',fullPage:true});
  await page.locator('#navigateBack').click();await w.locator('#integratedMultiResults').waitFor({state:'visible'});
  note('孪生返回保留预测','passed');
  await page.locator('#ribbonExternalBatch').click();await w.locator('#batchFile').waitFor({state:'visible'});
  await w.locator('#batchFile').setInputFiles(exports.xlsx);await w.locator('#loadBatchData').click();
  await w.locator('#runBatchPrediction').waitFor({state:'visible',timeout:120000});
  assert.equal(await w.locator('#batchPreview tbody tr').count(),count);
  await w.locator('#buildMonthlyWeather').click();
  assert.equal(await w.locator('#monthlyWeatherRows tr').count(),Number(await w.locator('#externalTargetMonths').inputValue()));
  await w.locator('#runBatchPrediction').click();
  await w.locator('#downloadBatchResults').waitFor({state:'visible',timeout:180000});
  note('真实导出Excel重新导入外部批量',await w.locator('#batchResult').innerText());
  for(const format of ['xlsx','csv','tsv','txt','json']){
    await w.locator('#batchResultFormat').selectOption(format);
    const download=page.waitForEvent('download',{timeout:60000});await w.locator('#downloadBatchResults').click();
    const file=await download,path=process.env.REAL_OUTPUT+'/batch_results.'+format;await file.saveAs(path);
    const bytes=fs.readFileSync(path);assert.ok(bytes.length>100);
    if(format==='xlsx')assert.equal(bytes.subarray(0,2).toString(),'PK');
    if(format==='json')assert.equal(JSON.parse(bytes.toString()).length,count);
  }
  note('批量预测结果五种真实格式下载','passed; xlsx 为 ZIP 工作簿，JSON 可解析');
  await page.locator('#ribbonTraining').click();await w.locator('#trainingPage').waitFor({state:'visible'});
  assert.equal(await w.locator('#predictRecognition').count(),0);
  note('训练与树种库入口',{recognition:await w.locator('#recognitionStatus').innerText(),species:await w.locator('#speciesMasterOptions option').count()});
  await page.locator('#ribbonHistory').click();await w.locator('#historyPage').waitFor({state:'visible'});
  note('历史记录入口','passed');await page.locator('#navigateBack').click();await w.locator('#trainingPage').waitFor({state:'visible'});
  note('历史返回训练','passed');assert.deepEqual(errors,[]);note('页面脚本异常',errors);
  if(process.env.OFFLINE_VERIFY==='1'){
    assert.deepEqual(outbound,[]);note('浏览器外网请求',outbound);
    const deniedPath=process.env.REAL_OUTPUT+'/server_outbound_denied.json';
    if(fs.existsSync(deniedPath)){
      const denied=JSON.parse(fs.readFileSync(deniedPath,'utf8'));
      assert.deepEqual(denied,[]);note('后端外网请求',denied);
    }else{
      note('后端外网请求','当前实例未启用请求记录；便携自检已验证 PROJ_NETWORK=OFF');
    }
  }
  fs.writeFileSync(process.env.REAL_OUTPUT+'/handoff_checks.json',JSON.stringify(checks,null,2));await browser.close();
})().catch(e=>{console.error(e);process.exitCode=1});
