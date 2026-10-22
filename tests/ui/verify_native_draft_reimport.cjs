const {chromium}=require(process.env.PLAYWRIGHT_MODULE);
const fs=require('node:fs'),assert=require('node:assert/strict');
(async()=>{
 const browser=await chromium.launch({channel:'msedge',headless:true});
 const page=await browser.newPage();const errors=[];
 page.on('pageerror',e=>errors.push(e.message));
 await page.context().route('**/*',r=>{
   const u=new URL(r.request().url());
   return ['127.0.0.1','localhost'].includes(u.hostname)?r.continue():r.abort('internetdisconnected');
 });
 await page.goto('http://127.0.0.1:18765/?workspace=batch');
 await page.locator('#batchFile').waitFor({state:'visible'});
 await page.locator('#batchFile').setInputFiles(process.env.AUDIT_ROOT+'/native_draft.xlsx');
 await page.locator('#loadBatchData').click();
 await page.locator('#runBatchPrediction').waitFor({state:'visible',timeout:60000});
 const count=await page.locator('#batchPreview tbody tr').count();
 assert.equal(count,6);assert.deepEqual(errors,[]);
 const audit={passed:true,rows:count,source:'native_draft.xlsx',
   scope:'Native Windows save and cancel observed; actual saved workbook parsed in independent external table page',errors};
 fs.writeFileSync(process.env.AUDIT_ROOT+'/native_draft_reimport.json',JSON.stringify(audit,null,2));
 console.log(JSON.stringify(audit));await browser.close();
})().catch(e=>{console.error(e);process.exitCode=1});
