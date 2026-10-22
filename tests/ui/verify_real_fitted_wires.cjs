const {chromium}=require(process.env.PLAYWRIGHT_MODULE);
const fs=require('node:fs'),assert=require('node:assert/strict');
(async()=>{
 const urls=JSON.parse(fs.readFileSync(process.env.REAL_SERVERS,'utf8'));
 const browser=await chromium.launch({channel:'msedge',headless:true});
 const page=await browser.newPage({viewport:{width:1600,height:1100}}),errors=[];
 page.on('pageerror',error=>errors.push(error.message));
 await page.goto(urls.processing+'step2');
 const expectedTrees=(await(await page.request.get(urls.processing+'api/desktop/workspace')).json()).batch.final_tree_count;
 await page.waitForFunction(n=>window.TreeLineTreeReviewApp?.state.treeMap.size===n,expectedTrees,{timeout:180000});
 const data=await page.evaluate(()=>window.TreeLineTreeReviewApp.state.latestViewerData);
 assert.equal(data.wires3d.source,'PERSISTED_SPAN_WIRE_RESULT_W1');
 assert.ok(data.wires3d.segments.length>0);
 assert.ok(data.wires3d.segments.every(segment=>segment.xyz.length>4));
 assert.equal(data.wires3d.xyz.length,data.wires3d.segments.reduce((sum,segment)=>sum+segment.xyz.length,0));
 assert.ok((await page.locator('#review-status').innerText()).includes('本次已查看'));
 assert.ok((await page.locator('#treetable thead').innerText()).includes('分割参考高'));
 assert.deepEqual(errors,[]);
 await page.screenshot({path:process.env.REAL_OUTPUT+'/review_fitted_wires.png',fullPage:true});
 const result={passed:true,trees:expectedTrees,fittedWireSegments:data.wires3d.segments.length,
   fittedWirePoints:data.wires3d.xyz.length,source:data.wires3d.source,scriptErrors:errors};
 fs.writeFileSync(process.env.REAL_OUTPUT+'/fitted_wire_checks.json',JSON.stringify(result,null,2));
 console.log(JSON.stringify(result));await browser.close();
})().catch(error=>{console.error(error);process.exitCode=1});
