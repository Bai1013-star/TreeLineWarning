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
  const unresolved=await page.evaluate(()=>window.TreeLineDbhReview.state.trees.find(tree=>tree.dbh_cm==null));
  assert.ok(unresolved,'需要至少一棵未获得胸径的独立验收树');
  await page.locator(`[data-tree-id="${unresolved.persistent_tree_id}"]`).click();
  await page.locator('#btn-prepare').click();
  await page.waitForFunction(()=>window.TreeLineDbhReview.state.rendererReady,{},{timeout:300000});
  const candidateCount=await page.locator('#axis-candidates button').count();
  assert.ok(candidateCount>0,'未生成可供人工判断的候选树干轴');
  const expected=await page.evaluate(()=>[...TreeLineDbhReview.state.review.candidates].sort((a,b)=>Number(b.candidate_score)-Number(a.candidate_score))[0].candidate_id);
  await page.locator('#btn-best-candidate').click();
  const selected=await page.evaluate(()=>TreeLineDbhReview.state.selectedCandidateId);
  assert.equal(selected,expected);
  assert.equal(await page.locator('#identity-confirmed').isChecked(),false,'采用候选后不得替用户自动勾选身份确认');
  assert.equal(await page.locator('#identity-confirmed').evaluate(input=>document.activeElement===input),true);
  await page.locator('#identity-confirmed').check();
  assert.equal(await page.locator('#btn-confirm').isEnabled(),true,'人工确认身份后应允许提交复核');
  assert.match(await page.locator('.review-howto').innerText(),/选择一棵树[\s\S]*准备点云复核[\s\S]*候选轴[\s\S]*确认人工身份并计算胸径/);
  await page.locator('#btn-reset-review').click();
  await page.locator('#btn-prepare:not([disabled])').waitFor();
  assert.deepEqual(errors,[]);assert.deepEqual(outbound,[]);
  const result={tree:unresolved.display_tree_id,candidate_count:candidateCount,best_candidate:selected,identity_requires_human:true,confirm_enabled_after_check:true,reset:true,offline:true};
  fs.writeFileSync(process.env.REAL_OUTPUT+'/dbh_guided_checks.json',JSON.stringify(result,null,2));
  console.log(JSON.stringify(result));
  await browser.close();
})().catch(error=>{console.error(error);process.exitCode=1});
