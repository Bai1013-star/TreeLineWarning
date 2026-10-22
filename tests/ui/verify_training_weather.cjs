const {chromium}=require(process.env.PLAYWRIGHT_MODULE);
const fs=require('node:fs'),assert=require('node:assert/strict');
(async()=>{
 const browser=await chromium.launch({channel:'msedge',headless:true});
 const page=await browser.newPage({acceptDownloads:true}),checks=[],errors=[];
 page.on('pageerror',e=>errors.push(e.message));page.on('dialog',d=>d.accept());
 const note=(name,detail)=>{checks.push({name,detail});console.log(JSON.stringify({name,detail}));};
 const base=(process.env.TRAINING_URL||'http://127.0.0.1:18765/').replace(/\/$/,'');
 try {
  await page.goto(base+'/?workspace=training');
  await page.locator('#trainingPage').waitFor({state:'visible'});
  const download=page.waitForEvent('download');await page.locator('#downloadTrainingTemplate').click();
  const path=process.env.REAL_OUTPUT+'/audit_training_template.csv';await(await download).saveAs(path);
  assert.ok(fs.statSync(path).size>200);note('训练模板真实下载',path);
  await page.locator('#historyFile').setInputFiles(path);await page.locator('#importHistoryFile').click();
  await page.locator('#trainHistory:not([disabled])').waitFor({timeout:120000});
  note('训练文件导入与数据池',await page.locator('#trainingPoolStatus').innerText());
  await page.locator('#trainingEpochs').selectOption('5');await page.locator('#trainHistory').click();
  await page.locator('#trainingTrainResult').getByText(/训练完成/).waitFor({timeout:180000});
  note('真实增量训练5轮',await page.locator('#trainingTrainResult').innerText());
  assert.ok(await page.locator('[data-training-run-delete]').count());
  await page.locator('[data-training-run-delete]').first().click();
  await page.locator('#trainingTrainResult').waitFor();note('测试训练记录删除','按钮已执行');
  const checkboxes=page.locator('[data-training-dataset]');await checkboxes.last().check();
  await page.locator('#deleteSelectedTrainingDatasets').click();
  await page.locator('#trainingDeleteResult').getByText(/已永久删除/).waitFor();note('测试数据池批次删除',await page.locator('#trainingDeleteResult').innerText());
  // Real single-tree LAS subset, not a botanical validation label.
  for(const species of ['杨树','油松']){
   await page.locator('#recognitionSpecies').fill(species);
   await page.locator('#recognitionPointCloud').setInputFiles(process.env.REAL_OUTPUT+'/audit_recognition.las');
   await page.locator('#saveRecognitionSample').click();
   await page.locator('#recognitionOutput').getByText(/已保存/).waitFor({timeout:120000});
  }
  await page.locator('#trainRecognition').click();
  await page.locator('#recognitionOutput').getByText(/模型已训练/).waitFor({timeout:120000});
  await page.locator('#predictRecognition').click();
  await page.locator('#recognitionOutput').getByText(/首选候选/).waitFor({timeout:120000});
  note('LAS样本保存、训练、离线候选识别',await page.locator('#recognitionOutput').innerText());
  const catalog=await(await page.request.get(base+'/species/catalog')).json();
  note('预测与训练共享树种库',{count:catalog.species.length,manualInput:true});
  if(process.env.OFFLINE_VERIFY==='1'){note('气象官网联网请求','离线验收不请求官网；不能算作在线接入通过');}
  else {
  const response=await page.request.post(base+'/weather/cma/auto',{data:{latitude:39.89,longitude:116.32},timeout:90000});
  const weather=await response.json();note('真实气象官网请求（未模拟）',{status:response.status(),body:weather});
  }
  assert.deepEqual(errors,[]);note('页面脚本异常',errors);
 }finally{fs.writeFileSync(process.env.REAL_OUTPUT+'/training_weather_audit.json',JSON.stringify(checks,null,2));await browser.close();}
})().catch(e=>{console.error(e);process.exitCode=1});
