// Positive control: ordinary finalize still navigates if operator stays.
const {chromium}=require(process.env.PLAYWRIGHT_MODULE);
const fs=require('node:fs'),assert=require('node:assert/strict');
(async()=>{
 const urls=JSON.parse(fs.readFileSync(process.env.REAL_SERVERS,'utf8'));
 const browser=await chromium.launch({channel:'msedge',headless:true});
 const page=await browser.newPage({viewport:{width:1700,height:1080}});const errors=[],external=[];
 page.on('pageerror',e=>errors.push(e.message));page.on('dialog',d=>d.accept());
 await page.route('**/*',r=>{const u=new URL(r.request().url());if(['http:','https:'].includes(u.protocol)&&!['127.0.0.1','localhost'].includes(u.hostname)){external.push(u.origin);return r.abort();}return r.continue();});
 try{
  await page.goto(urls.desktop+'&browser-test=1');
  await page.waitForFunction(()=>!document.querySelector('#footerState').textContent.includes('正在连接'));
  await page.locator('#ribbonReview').click();const f=page.frameLocator('#viewer');
  await f.locator('#treetable tbody tr').first().waitFor({timeout:180000});
  const review=page.frames().find(f=>new URL(f.url()).pathname==='/step2');
  await review.waitForFunction(()=>{const s=window.TreeLineTreeReviewApp.state;return s.viewer&&!s.editingBusy&&!s.mutation;},{},{timeout:180000});
  await f.locator('#btn-confirm').click();
  await page.waitForFunction(()=>document.querySelector('#viewer').contentWindow.location.pathname==='/step3',{},{timeout:360000});
  const publication=await(await page.request.get(urls.processing+'api/desktop/workspace',{timeout:180000})).json();
  assert.equal(publication.finalized,true);assert.deepEqual(errors,[]);assert.deepEqual(external,[]);
  fs.writeFileSync(process.env.REAL_OUTPUT+'/finalize_stay_ui.json',JSON.stringify({passed:true,path:'complete while staying -> step3',errors,external},null,2));
 }finally{await browser.close();}
})().catch(e=>{console.error(e);process.exitCode=1});
