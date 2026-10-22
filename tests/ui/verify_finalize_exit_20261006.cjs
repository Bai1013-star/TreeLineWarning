// Actual finalize button + Back during background work, no cancelled science.
const {chromium}=require(process.env.PLAYWRIGHT_MODULE);
const fs=require('node:fs'),assert=require('node:assert/strict');
(async()=>{
 const urls=JSON.parse(fs.readFileSync(process.env.REAL_SERVERS,'utf8'));
 const out=process.env.REAL_OUTPUT,checks=[],errors=[],external=[];
 const browser=await chromium.launch({channel:'msedge',headless:true});
 const page=await browser.newPage({viewport:{width:1700,height:1080}});
 page.on('pageerror',e=>errors.push(e.message));page.on('dialog',d=>d.accept());
 await page.route('**/*',r=>{const u=new URL(r.request().url());if(['http:','https:'].includes(u.protocol)&&!['127.0.0.1','localhost'].includes(u.hostname)){external.push(u.origin);return r.abort();}return r.continue();});
 try{
  await page.goto(urls.desktop+'&browser-test=1');
  await page.waitForFunction(()=>!document.querySelector('#footerState').textContent.includes('正在连接'));
  await page.locator('#ribbonReview').click();
  const f=page.frameLocator('#viewer');
  await f.locator('#treetable tbody tr').first().waitFor({timeout:180000});
  const review=page.frames().find(f=>new URL(f.url()).pathname==='/step2');
  await review.waitForFunction(()=>{const s=window.TreeLineTreeReviewApp.state;return s.viewer&&!s.editingBusy&&!s.mutation;},{},{timeout:180000});
  const submitted=page.waitForResponse(r=>new URL(r.url()).pathname==='/api/confirm_trees');
  await f.locator('#btn-confirm').click();await submitted;
  assert.equal(await review.evaluate(()=>window.TreeLineTreeReviewApp.state.editingBusy),true);
  await f.locator('#btn-review-back').click();await page.locator('#viewerWrap').waitFor({state:'hidden'});
  await page.locator('#ribbonData').click();
  await page.frameLocator('#moduleFrame').locator('#dataPage').waitFor({state:'visible'});
  const start=Date.now();let last;
  while(Date.now()-start<360000){
    const response=await page.request.get(urls.processing+'api/desktop/workspace',{timeout:180000});
    last=await response.json();
    const state=last.project?.workflow?.finalize_state;
    if(state?.status==='FAILED')throw Error(state.error);
    if(state?.status==='DONE'&&last.finalized)break;
    await page.waitForTimeout(1000);
  }
  assert.equal(last.project.workflow.finalize_state.status,'DONE');assert.equal(last.finalized,true);
  await page.waitForTimeout(2000);
  assert.equal(await page.locator('#viewerWrap').isVisible(),false);
  assert.equal(new URL(review.url()).pathname,'/step2','late response must not redirect hidden review');
  assert.equal(await page.frameLocator('#moduleFrame').locator('#dataPage').isVisible(),true);
  checks.push({name:'点击定稿后在忙碌状态退出，后台仍完成，页面未被拉回',passed:true});
  await page.frameLocator('#moduleFrame').locator('#refreshProcessedData').click();
  await page.frameLocator('#moduleFrame').locator('#processedDataBody input[type="radio"]').first().waitFor({timeout:180000});
  checks.push({name:'返回查看数据后可刷新最终结果',passed:true});
  assert.deepEqual(errors,[]);assert.deepEqual(external,[]);
 }finally{fs.writeFileSync(out+'/finalize_exit_ui.json',JSON.stringify({checks,errors,external},null,2));await browser.close();}
})().catch(e=>{console.error(e);process.exitCode=1});
