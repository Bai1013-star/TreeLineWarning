const {chromium}=require(process.env.PLAYWRIGHT_MODULE);
const fs=require('node:fs'),assert=require('node:assert/strict');
(async()=>{
  const urls=JSON.parse(fs.readFileSync(process.env.REAL_SERVERS,'utf8'));
  const out=process.env.REAL_OUTPUT,errors=[],outgoing=[],checks=[];
  const browser=await chromium.launch({channel:'msedge',headless:true});
  const page=await browser.newPage({viewport:{width:1700,height:1100}});
  page.on('pageerror',e=>errors.push(e.message));
  page.on('dialog',d=>d.accept());
  await page.route('**/*',route=>{const u=new URL(route.request().url());if(['http:','https:'].includes(u.protocol)&&!['127.0.0.1','localhost'].includes(u.hostname)){outgoing.push(u.href);return route.abort();}return route.continue()});
  const w=()=>page.frameLocator('#moduleFrame');
  try{
    const context=await(await page.request.get(urls.warning+'integrated/context')).json();
    const measured=context.trees.filter(t=>t.point_cloud_diameter!=null);
    const usable=(measured.length>=2?measured:context.trees).slice(0,2);
    assert.equal(usable.length,2,'need two real trees with measured DBH');
    await page.goto(urls.desktop+'&browser-test=1');
    await page.locator('#openWarningModule').click();
    await w().locator('#integratedMode').click();
    await w().locator('#integratedTree option').nth(context.trees.length-1).waitFor({state:'attached',timeout:180000});
    await w().locator('#integratedTree').selectOption(usable[0].tree_key);
    await w().locator('#integratedTreeCount').fill('2');await w().locator('#integratedTreeCount').blur();
    await w().locator('[data-integrated-extra-tree]').selectOption(usable[1].tree_key);
    assert.equal(await w().locator('[data-selected-tree-measurement]').count(),2);
    assert.match(await w().locator('[data-selected-tree-measurement]').nth(0).innerText(),/树高（点云）[\s\S]*胸径（点云）[\s\S]*冠幅（点云）[\s\S]*水平 \/ 垂直距离[\s\S]*三维最小净空/);
    const cards=w().locator('[data-selected-tree-measurement]');
    await cards.nth(0).locator('[data-integrated-tree-species]').fill('杨树');
    await cards.nth(0).locator('[data-integrated-tree-age]').fill('16');
    await cards.nth(0).locator('[data-manual-dbh-key]').fill(String(usable[0].point_cloud_diameter??usable[0].diameter));
    await cards.nth(1).locator('[data-integrated-tree-species]').fill('油松');
    await cards.nth(1).locator('[data-integrated-tree-age]').fill('31');
    await cards.nth(0).locator('[data-estimate-integrated-age]').click();
    // A valid estimate may legitimately round to the same integer already in
    // the field (for example, 16 -> 16).  Verify the completed per-tree status
    // and a positive numeric value rather than requiring an artificial change.
    await cards.nth(0).locator('[data-integrated-age-status]').filter({hasText:'估算约'}).waitFor({timeout:30000});
    const estimatedAge=await cards.nth(0).locator('[data-integrated-tree-age]').inputValue();
    assert.ok(Number(estimatedAge)>0);
    await cards.nth(0).locator('[data-integrated-tree-age]').fill('16');
    await w().locator('#integratedObservationDate').fill('2026-10-10');
    await w().locator('#integratedTargetDate').fill('2027-10-10');
    await w().locator('#integratedManualWeather').click();
    await w().locator('#saveIntegratedWeather').click();
    await w().locator('#openDailyWeather').click();
    for(const days of ['1','3','5','15']){
      await w().locator('#dailyWeatherDays').selectOption(days);
      assert.equal(await w().locator('#dailyWeatherRows tr').count(),Number(days));
    }
    await w().locator('#dailyWeatherDays').selectOption('3');
    const rows=w().locator('#dailyWeatherRows tr');assert.equal(await rows.count(),3);
    for(let i=0;i<3;i++){
      await rows.nth(i).locator('[data-daily-field="wind_force"]').fill(String(2+i*3));
      await rows.nth(i).locator('[data-daily-field="temperature"]').fill(String(9+i));
    }
    assert.equal(await w().locator('#dailyWeatherDays').inputValue(),'3');
    assert.equal(await rows.nth(0).locator('[data-daily-field="wind_force"]').inputValue(),'2');
    await w().locator('#saveIntegratedWeather').click();
    const weatherRequest=await w().locator('body').evaluate(()=>DailyWeather.request('2026-10-10'));
    assert.equal(weatherRequest.future_weather.length,3,JSON.stringify(weatherRequest));
    await w().locator('#integratedGenerate').click();
    await w().locator('#integratedMultiResults [data-open-tree-result="0"]').waitFor({timeout:180000});
    await w().locator('#integratedMultiResults [data-open-tree-result="0"]').click();
    await w().locator('#resultPage').waitFor({state:'visible',timeout:180000});
    const data=await w().locator('body').evaluate(()=>treeForecastResults.map(e=>e.result));
    assert.equal(data.length,2);
    assert.deepEqual(data.map(result=>result.input_snapshot.species),['杨树','油松']);
    assert.deepEqual(data.map(result=>result.input_snapshot.age),[16,31]);
    for(const result of data){
      assert.equal(result.daily_weather_enabled,true);
      assert.equal(result.future_weather.length,3);
      assert.equal(result.short_term_forecast.length,3);
      assert.deepEqual(result.short_term_forecast.map(v=>v.target_date),['2026-10-11','2026-10-12','2026-10-13']);
      assert.equal(new Set(result.short_term_forecast.map(v=>v.predicted_wind_speed)).size,3);
    }
    const text=await w().locator('#reportOutput').innerText();
    assert.equal((text.match(/配电线路树线隐患智能预警工单/g)||[]).length,1);
    assert.match(text,/按档距汇总/);assert.match(text,/档距1/);assert.match(text,/共 2 棵树/);
    assert.match(text,/到达危险阈值/);assert.match(text,/预计接触风级/);
    await page.evaluate(()=>{window.exportCalls=[];window.pywebview={api:{save_export_file:async(...args)=>{window.exportCalls.push(args);return {ok:true,path:'D:/TreeLineWarning/oct9_acceptance/B/latest_all_workorders.html'}}}}});
    assert.equal(await w().locator('#workOrderSaveScope').inputValue(),'all');
    await w().locator('#saveWorkOrder').click();await page.waitForFunction(()=>window.exportCalls.length===1);
    const args=await page.evaluate(()=>window.exportCalls[0]);
    const html=Buffer.from(args[1].split(',').at(-1),'base64').toString('utf8');
    assert.equal((html.match(/配电线路树线隐患智能预警工单/g)||[]).length,1);
    assert.match(html,/按档距汇总/);assert.match(html,/档距1/);
    fs.writeFileSync(out+'/latest_all_workorders.html',html);
    await w().locator('#backToInputPage').click();
    await w().locator('#openDailyWeather').click();
    assert.equal(await w().locator('#dailyWeatherDays').inputValue(),'3');
    await w().locator('#closeDailyWeather').click();
    checks.push({daily_weather_dates:3,tree_count:2,per_tree_species_age:true,age_estimator:true,span_grouped_work_order:true,back_preserved:true,offline:true});
    assert.deepEqual(errors,[]);assert.deepEqual(outgoing,[]);
    await page.screenshot({path:out+'/latest_features.png',fullPage:true});
  }finally{
    fs.writeFileSync(out+'/latest_feature_checks.json',JSON.stringify({checks,errors,outgoing},null,2));
    await browser.close();
  }
  console.log(JSON.stringify(checks));
})().catch(e=>{console.error(e);process.exitCode=1});
