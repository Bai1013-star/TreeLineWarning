const {chromium}=require(process.env.PLAYWRIGHT_MODULE);
const fs=require('node:fs');
const assert=require('node:assert/strict');

(async()=>{
  const real=JSON.parse(fs.readFileSync(process.env.REAL_SERVERS,'utf8'));
  const output=process.env.REAL_OUTPUT;
  const browser=await chromium.launch({channel:'msedge',headless:true});
  const page=await browser.newPage({acceptDownloads:true,viewport:{width:1600,height:1000}});
  const errors=[],external=[];
  page.on('pageerror',error=>errors.push(error.message));
  page.on('dialog',dialog=>dialog.accept());
  await page.route('**/*',route=>{
    const url=new URL(route.request().url());
    if(['http:','https:'].includes(url.protocol)&&!['127.0.0.1','localhost'].includes(url.hostname)){
      external.push(url.origin);return route.abort();
    }
    return route.continue();
  });
  try{
    await page.goto(real.desktop+'&browser-test=1');
    await page.locator('#ribbonTraining').click();
    const frame=page.frameLocator('#moduleFrame');
    await frame.locator('#trainingPage').waitFor({state:'visible'});
    assert.match(await frame.locator('#trainingDatasetManagerTitle').innerText(),/历史增量数据管理/);
    assert.ok(await frame.locator('#deleteSelectedTrainingDatasets').count());
    assert.ok(await frame.locator('#saveRecognitionSample').count());
    assert.ok(await frame.locator('#trainRecognition').count());
    assert.ok(await frame.locator('#predictRecognition').count());
    const templateWait=page.waitForEvent('download');
    await frame.locator('#downloadTrainingTemplate').click();
    const download=await templateWait;
    const templatePath=output+'/training_template_20261011.csv';
    await download.saveAs(templatePath);
    assert.ok(fs.statSync(templatePath).size>200);
    await frame.locator('#historyFile').setInputFiles(templatePath);
    await frame.locator('#importHistoryFile').click();
    await frame.locator('#trainHistory:not([disabled])').waitFor({timeout:120000});
    const dataset=frame.locator('[data-training-dataset]').last();
    await dataset.waitFor();
    const datasetId=await dataset.getAttribute('data-training-dataset');
    await dataset.check();
    assert.equal(await frame.locator('#deleteSelectedTrainingDatasets').isEnabled(),true);
    await frame.locator('#deleteSelectedTrainingDatasets').click();
    await frame.locator('#trainingDeleteResult').getByText(/已永久删除/).waitFor({timeout:120000});
    assert.equal(await frame.locator(`[data-training-dataset="${datasetId}"]`).count(),0);
    assert.deepEqual(errors,[]);
    assert.deepEqual(external,[]);
    const result={template_download:true,incremental_import:true,dataset_visible:true,dataset_delete:true,
      recognition_sample_train_predict_controls:true,page_errors:errors,offline:true};
    fs.writeFileSync(output+'/training_manager_20261011.json',JSON.stringify(result,null,2));
    console.log(JSON.stringify(result));
  }finally{await browser.close();}
})().catch(error=>{console.error(error);process.exitCode=1});
