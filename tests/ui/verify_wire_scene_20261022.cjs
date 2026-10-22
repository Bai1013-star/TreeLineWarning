const {chromium}=require(process.env.PLAYWRIGHT_MODULE);
const fs=require('node:fs'),assert=require('node:assert/strict');
(async()=>{
  const urls=JSON.parse(fs.readFileSync(process.env.REAL_SERVERS,'utf8'));
  const browser=await chromium.launch({channel:'msedge',headless:true});
  const page=await browser.newPage({viewport:{width:1700,height:1100}}),errors=[];
  page.on('pageerror',error=>errors.push(error.message));
  await page.goto(urls.processing+'step2');
  await page.waitForFunction(()=>window.TreeLineTreeReviewApp?.state.treeMap.size>0,{},{timeout:120000});
  await page.locator('#btn-view-side').click();
  const status=await page.locator('#viewer-renderer-status').innerText();
  assert.match(status,/3\s*根已拟合导线/);
  assert.equal(await page.evaluate(()=>TreeLineTreeReviewApp.state.editorDisplay.wires3d.segments.length),3);
  await page.screenshot({path:process.env.REAL_OUTPUT+'/真实点云_三根同高度导线.png',fullPage:true});
  assert.deepEqual(errors,[]);console.log(JSON.stringify({passed:true,status,script_errors:errors}));
  fs.writeFileSync(process.env.REAL_OUTPUT+'/真实导线场景验收.json',JSON.stringify({passed:true,status},null,2));
  await browser.close();
})().catch(error=>{console.error(error);process.exitCode=1});
