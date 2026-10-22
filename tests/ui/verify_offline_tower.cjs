const {chromium}=require(process.env.PLAYWRIGHT_MODULE);
const fs=require('node:fs'),assert=require('node:assert/strict');
(async()=>{
 const urls=JSON.parse(fs.readFileSync(process.env.REAL_SERVERS,'utf8'));
 const browser=await chromium.launch({channel:'msedge',headless:true,args:['--proxy-server=direct://','--proxy-bypass-list=*']});
 const page=await browser.newPage({viewport:{width:1600,height:1100}}),outbound=[],errors=[];
 await page.context().route('**/*',r=>{
   const url=new URL(r.request().url());
   if(['127.0.0.1','localhost','[::1]'].includes(url.hostname))return r.continue();
   outbound.push(url.href);return r.abort('internetdisconnected');
 });
 page.on('pageerror',e=>errors.push(e.message));
 try {
   await page.goto(urls.desktop+'&browser-test=1');
   assert.ok((await page.locator('.brand').innerText()).includes('2026/10/02'));
   // DOM page buttons invoke the real overview API; don't hide failures by
   // using a cached direct API response in place of the actual 3D canvas.
   await page.locator('#towerLoad').waitFor({timeout:30000});
   if(process.env.INVALID_SOURCE==='1'){
     await page.locator('#towerLoad').filter({hasText:'原始 LAS 文件不完整'}).waitFor({timeout:30000});
     const text=await page.locator('#towerLoad').innerText();
     assert.ok(text.includes('84382141')||text.includes('84,382,141'));
     assert.ok(text.includes('重新复制'));
     assert.ok(!text.includes('buffer size'));
     fs.writeFileSync(process.env.REAL_OUTPUT+'/tower_offline.json',JSON.stringify({
       status:'TRUNCATED_SOURCE_CORRECTLY_REJECTED',message:text,outbound,errors},null,2));
   }else{
     await page.locator('#towerLoad').filter({hasText:'真实 3D 点云已就绪'}).waitFor({timeout:180000});
     for(const id of ['topTowerView','sideTowerView','perspectiveTowerView','resetTowerView'])await page.locator('#'+id).click();
     for(const mode of ['rgb','intensity','height'])await page.locator('#displayMode').selectOption(mode);
     await page.locator('#pointSize').evaluate(e=>{e.value='4';e.dispatchEvent(new Event('input',{bubbles:true}));});
     const message=await page.locator('#towerLoad').innerText();
     fs.writeFileSync(process.env.REAL_OUTPUT+'/tower_offline.json',JSON.stringify({
       status:'REAL_TOWER_OVERVIEW_READY_OFFLINE',message,outbound,errors},null,2));
   }
   await page.screenshot({path:process.env.REAL_OUTPUT+'/tower_offline.png',fullPage:true});
   assert.deepEqual(outbound,[]);assert.deepEqual(errors,[]);
   console.log(await page.locator('#towerLoad').innerText());
 }finally{await browser.close();}
})().catch(e=>{console.error(e);process.exitCode=1});
