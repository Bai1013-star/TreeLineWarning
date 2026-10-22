const {chromium}=require(process.env.PLAYWRIGHT_MODULE);
const fs=require('node:fs'),assert=require('node:assert/strict');
(async()=>{
 const browser=await chromium.launch({channel:'msedge',headless:true});
 const page=await browser.newPage({acceptDownloads:true});
 const checks=[],errors=[];page.on('pageerror',e=>errors.push(e.message));
 page.on('dialog',d=>d.accept());
 const note=(name,detail)=>{checks.push({name,detail});console.log(JSON.stringify({name,detail}));};
 const base='http://127.0.0.1:18765/';
 await page.goto(base+'?workspace=single');
 await page.locator('#singleMode').waitFor({state:'visible'});
 await page.locator('#species').fill('油松');await page.locator('#height').fill('12.34');
 // Complete explicitly entered fixture observations, not hidden model defaults.
 for(const field of ['tree_id','diameter','vertical_distance','horizontal_distance','wind_offset_distance','maximum_sag_winter','maximum_sag_summer','wind_speed','rainfall','humidity','temperature','sunlight_hours']){
   const toggle=page.locator('#use_'+field);if(await toggle.count())await toggle.check();
 }
 await page.locator('#diameter').fill('20');
 await page.locator('#generateReport').click();await page.locator('#resultPage').waitFor({state:'visible',timeout:120000});
 const saved=await(await page.request.get(base+'history')).json();
 const ownId=saved.history[0].id;assert.ok(ownId);
 assert.equal(saved.history[0].height,12.34);
 await page.locator('#addForecastToTraining').click();
 await page.locator('#addForecastToTraining').filter({hasText:'已加入增量训练'}).waitFor({timeout:60000});
 note('加入训练按钮','实际写入隔离测试数据池；不作为真实树木观测或科学训练准确率证据');
 await page.locator('#addComparison').click();await page.locator('#inputPage').waitFor({state:'visible'});
 await page.locator('#generateReport').click();await page.locator('#resultPage').waitFor({state:'visible',timeout:120000});
 assert.equal(await page.evaluate(()=>activeResultMode),'ai');note('运行另一方法对比','本地另一方法实际运行成功');
 await page.goto(base+'?workspace=history');await page.locator('#refreshHistory').click();
 await page.waitForLoadState('networkidle');
 await page.locator(`[data-history-manage-id="${ownId}"]`).waitFor();
 await page.locator('#selectAllHistory').click();assert.ok(await page.locator('#deleteSelectedHistory').isEnabled());
 await page.locator('#selectAllHistory').click();assert.ok(await page.locator('#deleteSelectedHistory').isDisabled());
 await page.locator(`[data-history-manage-id="${ownId}"]`).click();
 await page.locator('#loadHistoryIntoForm:not([disabled])').waitFor();
 await page.locator('#closeHistoryDetailBottom').click();
 await page.locator(`[data-history-manage-id="${ownId}"]`).click();
 await page.locator('#loadHistoryIntoForm:not([disabled])').waitFor();await page.locator('#loadHistoryIntoForm').click();
 await page.locator('#inputPage').waitFor({state:'visible'});assert.equal(await page.locator('#height').inputValue(),'12.34');
 note('历史刷新/全选取消/详情关闭/载入输入','真实保存的输入恢复成功');
 await page.goto(base+'?workspace=history');await page.locator(`[data-history-manage-id="${ownId}"]`).click();
 await page.locator('#openHistoryFullReport:not([disabled])').waitFor();await page.locator('#openHistoryFullReport').click();
 await page.locator('#resultPage').waitFor({state:'visible'});assert.ok((await page.locator('#reportOutput').innerText()).includes('工单'));
 note('历史完整报告','实际打开成功');
 await page.goto(base+'?workspace=batch');await page.locator('#batchFile').waitFor({state:'visible'});
 const dl=page.waitForEvent('download');await page.locator('#downloadExternalBatchTemplate').click();
 await(await dl).saveAs(process.env.REAL_OUTPUT+'/external_template.csv');
 await page.locator('#externalManualTable summary').click();
 await page.locator('#addExternalBatchRow').click();assert.equal(await page.locator('#externalBatchTableBody tr').count(),2);
 await page.locator('[data-remove-external-row]').last().click();assert.equal(await page.locator('#externalBatchTableBody tr').count(),1);
 await page.locator('#addExternalBatchRow').click();
 for(const row of await page.locator('#externalBatchTableBody tr').all()){
  for(const [key,value] of Object.entries({species:'油松',height:'10',diameter:'20',crown_width:'3',conductor_height:'15',horizontal_distance:'4',vertical_distance:'3',clearance_distance:'5'}))
   await row.locator(`[data-external-field="${key}"]`).fill(value);
 }
 await page.locator('#loadExternalBatchTable').click();await page.locator('#runBatchPrediction').waitFor({state:'visible'});
 assert.equal(await page.locator('#batchPreview tbody tr').count(),2);
 await page.locator('#integratedManualWeather').click();await page.locator('#saveIntegratedWeather').click();
 await page.locator('#externalTargetMonths').fill('1');await page.locator('#runBatchPrediction').click();
 await page.locator('#printBatchWorkOrder').waitFor({state:'visible',timeout:120000});
 const popupWait=page.waitForEvent('popup');await page.locator('#printBatchWorkOrder').click();const popup=await popupWait;
 await popup.locator('tbody tr').nth(1).waitFor();assert.equal(await popup.locator('tbody tr').count(),2);
 assert.equal(await popup.locator('body').evaluate(e=>getComputedStyle(e).backgroundColor),'rgb(16, 24, 32)');
 note('表格模板/增行删行/页面表格解析/多树精简工单','2棵实际预测，按树号排列，深色工单弹窗成功；未执行物理打印');
 await popup.close();
 const workOrderDownload=page.waitForEvent('download');await page.locator('#saveBatchWorkOrder').click();
 const workOrder=await workOrderDownload;const workOrderPath=process.env.REAL_OUTPUT+'/external_two_tree_work_order.html';
 await workOrder.saveAs(workOrderPath);const html=fs.readFileSync(workOrderPath,'utf8');
 assert.match(html,/<meta charset="utf-8">/i);assert.match(html,/EXT-1/);assert.match(html,/EXT-2/);assert.match(html,/<\/html>/);
 note('外部批量工单另存','实际保存了含两棵树结果的独立 UTF-8 HTML 工单；浏览器下载路径验收，非原生系统对话框验收');
 await page.locator('#clearBatchData').click();assert.equal(await page.locator('#batchPreview').isVisible(),false);
 note('清空外部表格','回到空白输入状态');assert.deepEqual(errors,[]);
 fs.writeFileSync(process.env.REAL_OUTPUT+'/remaining_prediction_buttons.json',JSON.stringify({checks,errors},null,2));
 await browser.close();
})().catch(e=>{console.error(e);process.exitCode=1});
