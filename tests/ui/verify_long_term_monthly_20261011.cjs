const {chromium}=require(process.env.PLAYWRIGHT_MODULE);
const fs=require('node:fs'),assert=require('node:assert/strict');
(async()=>{
  const urls=JSON.parse(fs.readFileSync(process.env.REAL_SERVERS,'utf8'));
  const browser=await chromium.launch({channel:'msedge',headless:true});
  const page=await browser.newPage({viewport:{width:1700,height:1100}}),errors=[],outbound=[],forecastBodies=[];
  page.on('pageerror',error=>errors.push(error.message));
  page.on('dialog',dialog=>{console.log('DIALOG',dialog.type(),dialog.message());dialog.accept()});
  page.on('request',request=>{
    if(request.url().includes('/predict/forecast')&&request.method()==='POST'){
      console.log('FORECAST_REQUEST',request.url());forecastBodies.push(request.postDataJSON());
    }
  });
  await page.route('**/*',route=>{
    const url=new URL(route.request().url());
    if(['http:','https:'].includes(url.protocol)&&!['127.0.0.1','localhost'].includes(url.hostname)){
      outbound.push(url.href);return route.abort('internetdisconnected');
    }
    return route.continue();
  });
  await page.goto(urls.desktop+'&browser-test=1');
  await page.locator('#ribbonExternalBatch').click();
  const w=page.frameLocator('#moduleFrame');
  await w.locator('#batchEntry:not([hidden])').waitFor({timeout:60000});
  assert.match(await w.locator('#batchEntry h3').first().innerText(),/长时表格预测/);
  assert.equal(await w.locator('#dailyWeatherSection').isHidden(),true);
  assert.equal(await w.locator('#weatherSourceConsole').isHidden(),true);
  await w.locator('#externalManualTable').evaluate(element=>element.open=true);
  const row=w.locator('#externalBatchTableBody tr').first();
  await row.locator('[data-external-field="tree_id"]').fill('LONG-001');
  await row.locator('[data-external-field="species"]').fill('油松');
  await row.locator('[data-external-field="age"]').fill('18');
  await row.locator('[data-external-field="height"]').fill('8.5');
  await row.locator('[data-external-field="diameter"]').fill('21.2');
  await row.locator('[data-external-field="crown_width"]').fill('3.4');
  await row.locator('[data-external-field="horizontal_distance"]').fill('3.2');
  await row.locator('[data-external-field="vertical_distance"]').fill('2.1');
  await row.locator('[data-external-field="clearance_distance"]').fill('3.8');
  await w.locator('#loadExternalBatchTable').click();
  await w.locator('#runBatchPrediction:not([disabled])').waitFor();
  // Follow the page's visible order: after importing/validating the table, set the
  // long-term period and monthly weather immediately before prediction.
  await w.locator('#externalObservationDate').evaluate((input)=>{input.value='2026-10-11';input.dispatchEvent(new Event('change',{bubbles:true}))});
  await w.locator('#externalTargetMonths').evaluate((input)=>{input.value='3';input.dispatchEvent(new Event('change',{bubbles:true}))});
  const monthlyValues=[[2,6,50,12,5],[4,2,55,8,4],[6,-1,60,4,3]];
  await w.locator('#monthlyWeatherRows').evaluate((body,values)=>{
    [...body.querySelectorAll('tr')].forEach((row,index)=>{
      ['wind_force','temperature','humidity','rainfall','sunlight_hours'].forEach((field,fieldIndex)=>{
        const input=row.querySelector(`[data-monthly-field="${field}"]`);input.value=String(values[index][fieldIndex]);input.dispatchEvent(new Event('input',{bubbles:true}));
      });
    });
  },monthlyValues);
  assert.equal(await w.locator('#monthlyWeatherRows tr').count(),3);
  console.log('LONG_TERM_INPUTS',await w.locator('#externalObservationDate').inputValue(),await w.locator('#externalTargetMonths').inputValue());
  const responsePromise=page.waitForResponse(response=>response.url().includes('/predict/forecast'),{timeout:180000});
  await w.locator('#runBatchPrediction').click();
  const forecastResponse=await responsePromise;
  console.log('FORECAST_RESPONSE',forecastResponse.status());
  await w.locator('#runBatchPrediction').getByText(/重新预测/).waitFor({timeout:180000});
  const resultText=await w.locator('#batchResult').innerText();
  assert.match(resultText,/已完成 1 \/ 1 条树木预测/,resultText);
  assert.equal(forecastBodies.length,1);
  const body=forecastBodies[0];
  console.log('FORECAST_BODY',JSON.stringify(body));
  assert.deepEqual(body.future_monthly_weather.map(item=>item.wind_force),[2,4,6]);
  assert.deepEqual(body.long_horizons_months,[1,2,3]);
  assert.equal(body.tree_data.species,'油松');
  assert.equal(body.tree_data.age,18);
  assert.deepEqual(errors,[]);
  assert.deepEqual(outbound,[]);
  const result={long_term_title:true,monthly_rows:3,monthly_forces:[2,4,6],result:'1/1',offline:true};
  fs.writeFileSync(process.env.REAL_OUTPUT+'/long_term_monthly_checks.json',JSON.stringify(result,null,2));
  console.log(JSON.stringify(result));
  await browser.close();
})().catch(error=>{console.error(error);process.exitCode=1});
