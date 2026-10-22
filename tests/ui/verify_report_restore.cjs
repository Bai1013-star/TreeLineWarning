const {chromium}=require(process.env.PLAYWRIGHT_MODULE);
const fs=require('node:fs'),assert=require('node:assert/strict');
(async()=>{
 const urls=JSON.parse(fs.readFileSync(process.env.REAL_SERVERS,'utf8'));
 const browser=await chromium.launch({channel:'msedge',headless:true});
 const page=await browser.newPage(),checks=[];
 try{
  await page.goto(urls.warning+'?workspace=single');
  await page.locator('#singleEntry').waitFor({state:'visible'});
  await page.locator('#species').fill('油松');await page.locator('#height').fill('10');
  await page.locator('#generateReport').click();await page.locator('#resultPage').waitFor({state:'visible',timeout:120000});
  const id=await page.evaluate(()=>activeWorkOrderId);assert.ok(id);
  assert.ok((await page.locator('#reportOutput').innerText()).includes('危险'));
  await page.evaluate(()=>{window.__auditPrintCalls=0;window.print=()=>window.__auditPrintCalls++});
  await page.locator('#printReport').click();assert.equal(await page.evaluate(()=>window.__auditPrintCalls),1);
  checks.push({name:'打印按钮调用系统打印入口',passed:true,scope:'验证调用；未连接物理打印机或保存 PDF'});
  await page.reload();
  await page.waitForFunction(()=>typeof openSavedOrCurrentReport==='function');
  assert.equal(await page.evaluate(()=>openSavedOrCurrentReport()),true);
  assert.equal(await page.evaluate(()=>activeWorkOrderId),id);
  assert.ok((await page.locator('#reportOutput').innerText()).includes('工单'));
  const bg=await page.locator('#reportOutput').evaluate(e=>getComputedStyle(e).backgroundColor);
  checks.push({name:'关闭页面重载后恢复同一完整工单',passed:true,id,background:bg});
  fs.writeFileSync(process.env.REAL_OUTPUT+'/report_restore_checks_20261001.json',JSON.stringify(checks,null,2));
  console.log(JSON.stringify(checks));
 }finally{await browser.close()}
})().catch(e=>{console.error(e);process.exitCode=1});
