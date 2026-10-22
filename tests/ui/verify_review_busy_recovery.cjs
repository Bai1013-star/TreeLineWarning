// Read-only verification against the user's currently running local service.
// Inject only HTTP read responses: never submit an editor/DBH/finalize action.
const {chromium} = require(process.env.PLAYWRIGHT_MODULE);
const fs = require('node:fs'), assert = require('node:assert/strict');
(async () => {
  const base = process.env.REAL_PROCESSING_URL;
  assert.ok(/^http:\/\/127\.0\.0\.1:\d+\/$/.test(base));
  const browser = await chromium.launch({channel:'msedge',headless:true});
  const checks = [], posts = [], outbound = [];
  const script = fs.readFileSync(process.env.REVIEW_APP_SOURCE,'utf8');
  const busy = {ok:false,status:'EDIT_RUNNING',error:{code:'EDIT_RUNNING',message:'EDIT_RUNNING: 正在读取胸径证据',details:'DBH_REVIEW_PREVIEW busy test'}};
  const scenario = async kind => {
    const page = await browser.newPage({viewport:{width:1500,height:1100}});
    const errors = [];
    let reads = 0;
    page.on('pageerror',e=>errors.push(e.message));
    page.on('request',r=>{if(r.method()==='POST')posts.push(new URL(r.url()).pathname);});
    await page.route('**/*',route=>{
      const url=new URL(route.request().url());
      if(url.hostname!=='127.0.0.1' && !['data:','blob:'].includes(url.protocol)){
        outbound.push(url.href);return route.abort();
      }
      return route.continue();
    });
    await page.route('**/tree-review-app.js',r=>r.fulfill({contentType:'application/javascript',body:script}));
    await page.route('**/api/editor_data?**',async route=>{
      reads++;
      if(kind==='busy' && reads<=2){
        const body=reads===1 ? busy : {ok:false,error:busy.error};
        return route.fulfill({status:409,contentType:'application/json',body:JSON.stringify(body)});
      }
      if(kind==='timeout')return route.fulfill({status:409,contentType:'application/json',body:JSON.stringify(busy)});
      if(kind==='failure' || kind==='logical_failure')return route.fulfill({
        status:kind==='failure'?409:200,contentType:'application/json',
        body:JSON.stringify({ok:false,error:{code:'MISSING_MEMBERS',message:'验收测试：原始成员证据缺失',details:'MISSING_MEMBERS: synthetic read failure preserved'}})
      });
      return route.continue();
    });
    if(kind==='timeout')await page.clock.install();
    await page.goto(base+'step2');
    if(kind==='busy' || kind==='timeout'){
      await page.waitForFunction(()=>document.querySelector('#editmsg').textContent.includes('正在等待后台'));
      assert.equal(await page.locator('#btn-merge').isDisabled(),true);
      assert.equal(await page.locator('#btn-confirm').isDisabled(),true);
    }
    if(kind==='busy'){
      await page.waitForFunction(()=>window.TreeLineTreeReviewApp?.state.viewer?.treeXYZ?.length>0,{},{timeout:120000});
      assert.equal(await page.locator('#treetable tbody tr').count(),6);
      assert.equal(reads,3);
      assert.equal(await page.locator('#btn-merge').isEnabled(),true);
      assert.ok((await page.locator('#editmsg').textContent()).includes('自动加载'));
      assert.ok((await page.locator('#viewer-renderer-status').textContent()).includes('WebGL'));
      await page.screenshot({path:process.env.REAL_OUTPUT+'/busy_recovered.png',fullPage:true});
    }else if(kind==='timeout'){
      await page.clock.fastForward(121000);
      await page.waitForFunction(()=>document.querySelector('#editmsg').textContent.includes('后台任务仍未完成'));
      const finalReads=reads;
      await page.clock.fastForward(5000);
      assert.equal(reads,finalReads); // bounded wait; no endless initialization
      assert.ok((await page.locator('#viewer-renderer-status').textContent()).includes('数据暂未加载'));
    }else{
      await page.waitForFunction(()=>document.querySelector('#editmsg').textContent.includes('原始成员证据缺失'));
      assert.ok((await page.locator('#error-detail-text').textContent()).includes('MISSING_MEMBERS'));
      assert.equal(reads,1); // permanent failures must not be retried
      assert.ok((await page.locator('#viewer-renderer-status').textContent()).includes('数据暂未加载'));
    }
    assert.deepEqual(errors,[]);
    const result={kind,reads,passed:true};checks.push(result);console.log(JSON.stringify(result));
    await page.close();
  };
  try{
    for(const kind of ['busy','failure','logical_failure','timeout'])await scenario(kind);
    assert.deepEqual(outbound,[]);
    assert.ok(posts.every(path=>path==='/api/viewer/diagnostic'),JSON.stringify(posts));
    checks.push({name:'无外网请求、未提交编辑/胸径/定稿操作',posts,passed:true});
    fs.writeFileSync(process.env.REAL_OUTPUT+'/busy_recovery_checks.json',JSON.stringify(checks,null,2));
  }finally{await browser.close();}
})().catch(e=>{console.error(e);process.exitCode=1;});
