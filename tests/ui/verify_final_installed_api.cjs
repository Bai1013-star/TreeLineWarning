const {chromium}=require(process.env.PLAYWRIGHT_MODULE);
const fs=require('node:fs'),assert=require('node:assert/strict');
(async()=>{
  const urls=JSON.parse(fs.readFileSync(process.env.REAL_SERVERS,'utf8'));
  const browser=await chromium.launch({channel:'msedge',headless:true});
  const page=await browser.newPage();
  try {
    const response=await page.request.post(urls.warning+'weather/cma/auto',{
      data:{latitude:39.89,longitude:116.32},timeout:90000});
    const body=await response.json();
    assert.equal(response.status(),200);
    assert.notEqual(body.weather?.wind_direction,'9999');
    const result={checked_at:new Date().toISOString(),status:response.status(),body,
      scope:'真实官方接口；不是模拟天气；日照为估算值，不是观测日照'};
    fs.writeFileSync(process.env.REAL_OUTPUT+'/cma_final_installed.json',JSON.stringify(result,null,2));
    console.log(JSON.stringify(result));
  } finally {await browser.close();}
})().catch(e=>{console.error(e);process.exitCode=1});
