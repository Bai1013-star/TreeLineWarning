const {chromium}=require(process.env.PLAYWRIGHT_MODULE);
const fs=require('node:fs'),assert=require('node:assert/strict');
(async()=>{
  const urls=JSON.parse(fs.readFileSync(process.env.REAL_SERVERS,'utf8'));
  const browser=await chromium.launch({channel:'msedge',headless:true});
  const page=await browser.newPage({viewport:{width:1700,height:1100}}),errors=[],outbound=[];
  page.on('pageerror',error=>errors.push(error.message));
  await page.route('**/*',route=>{
    const url=new URL(route.request().url());
    if(['http:','https:'].includes(url.protocol)&&!['127.0.0.1','localhost'].includes(url.hostname)){
      outbound.push(url.href);return route.abort('internetdisconnected');
    }
    return route.continue();
  });
  await page.goto(urls.processing+'dbh-review');
  await page.waitForFunction(()=>window.TreeLineDbhReview?.state.trees.length>0,{},{timeout:180000});
  const existing=await page.evaluate(()=>window.TreeLineDbhReview.state.trees.find(tree=>Number(tree.dbh_cm)>0));
  assert.ok(existing,'真实项目应至少有一棵已自动测得胸径的树');
  await page.locator(`[data-tree-id="${existing.persistent_tree_id}"]`).click();
  assert.equal(await page.locator('#btn-prepare').isEnabled(),true,'已有胸径的树也必须允许重新复核');
  await page.locator('#btn-prepare').click();
  await page.waitForFunction(()=>window.TreeLineDbhReview.state.rendererReady,{},{timeout:300000});
  const candidateCount=await page.locator('#axis-candidates button').count();
  assert.ok(candidateCount>0,'应从原始点云最低点+1.30米生成候选树干轴');
  await page.locator('#btn-best-candidate').click();
  await page.locator('#identity-confirmed').check();
  assert.equal(await page.locator('#btn-confirm').isEnabled(),true,'人工确认后计算胸径按钮应可点击');
  await page.screenshot({path:process.env.REAL_OUTPUT+'/胸径复核_已有结果可重测.png',fullPage:true});
  await page.locator('#btn-reset-review').click();
  assert.deepEqual(errors,[]);assert.deepEqual(outbound,[]);
  const result={tree:existing.display_tree_id,existing_dbh_cm:existing.dbh_cm,
    method:existing.dbh_method,height_reference:existing.dbh_height_reference,
    candidate_count:candidateCount,prepare_enabled:true,confirm_enabled:true,offline:true};
  fs.writeFileSync(process.env.REAL_OUTPUT+'/dbh_recheck_checks.json',JSON.stringify(result,null,2));
  console.log(JSON.stringify(result));
  await browser.close();
})().catch(error=>{console.error(error);process.exitCode=1});
