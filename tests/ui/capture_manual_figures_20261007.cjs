const {chromium}=require(process.env.PLAYWRIGHT_MODULE);
(async()=>{
 const browser=await chromium.launch({channel:'msedge',headless:true});
 const page=await browser.newPage({viewport:{width:980,height:1100},deviceScaleFactor:2});
 await page.goto('http://127.0.0.1:18766/?workspace=single&embedded=1');
 await page.locator('#integratedMode').click();
 await page.locator('#integratedTree option').nth(2).waitFor({state:'attached'});
 await page.locator('#integratedTree').selectOption('fixture-1');
 await page.locator('[data-manual-dbh-key="fixture-1"]').locator('..').screenshot({path:process.env.REAL_OUTPUT+'/manual-detail.png'});
 await page.goto('http://127.0.0.1:18766/?workspace=batch&embedded=1');
 await page.locator('#batchImportGrid .batch-card').first().screenshot({path:process.env.REAL_OUTPUT+'/external-detail.png'});
 await browser.close();
})().catch(e=>{console.error(e);process.exit(1)});
