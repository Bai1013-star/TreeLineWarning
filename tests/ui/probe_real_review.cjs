const {chromium}=require(process.env.PLAYWRIGHT_MODULE);
const fs=require('node:fs');
(async()=>{
  const urls=JSON.parse(fs.readFileSync(process.env.REAL_SERVERS,'utf8'));
  const browser=await chromium.launch({channel:'msedge',headless:true});
  const page=await browser.newPage({viewport:{width:1600,height:1000}});
  const errors=[],requests=[];
  page.on('pageerror',e=>errors.push(e.message));
  page.on('request',r=>{if(r.url().includes('/api/editor_data')) requests.push(r.url());});
  await page.goto(urls.processing+'step2');
  await page.waitForTimeout(15000);
  console.log(JSON.stringify({errors,requests,splitCount:await page.locator('#btn-split').count(),
    status:await page.locator('#viewer-renderer-status').textContent(),
    body:(await page.locator('body').innerText()).slice(-1600)},null,2));
  await page.screenshot({path:process.env.REAL_SHOT,fullPage:true});
  await browser.close();
})().catch(e=>{console.error(e);process.exitCode=1});
