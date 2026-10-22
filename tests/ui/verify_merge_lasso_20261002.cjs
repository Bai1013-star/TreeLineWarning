// Real installed backend + real point-cloud membership. Only own audit projects.
const {chromium} = require(process.env.PLAYWRIGHT_MODULE);
const fs = require('node:fs'), assert = require('node:assert/strict');
(async () => {
  const urls = JSON.parse(fs.readFileSync(process.env.REAL_SERVERS, 'utf8'));
  const browser = await chromium.launch({channel: 'msedge', headless: true});
  const page = await browser.newPage({viewport: {width: 1800, height: 1500}});
  const checks = [], errors = [], outbound = [];
  let acceptDialog = true;
  page.on('dialog', d => acceptDialog ? d.accept() : d.dismiss());
  page.on('pageerror', e => errors.push(e.message));
  await page.route('**/*', route => {
    const url = new URL(route.request().url());
    if (['127.0.0.1', 'localhost', '[::1]'].includes(url.hostname) || ['data:', 'blob:'].includes(url.protocol)) return route.continue();
    outbound.push(url.href); return route.abort();
  });
  const stats = () => page.evaluate(() => {
    const s = window.TreeLineTreeReviewApp.state;
    return {trees: s.treeMap.size, points: [...s.treeMap.values()].reduce((n,t) => n+t.point_count,0), journal: s.committedJournalCount};
  });
  const idle = () => page.waitForFunction(() => {
    const s = window.TreeLineTreeReviewApp?.state;
    return s && !s.editingBusy && !s.mutation;
  }, {}, {timeout: 240000});
  const count = async n => {
    await page.waitForFunction(n => {
      const s = window.TreeLineTreeReviewApp.state;
      return !s.editingBusy && !s.mutation && (s.treeMap.size === n || document.querySelector('#editmsg').textContent.startsWith('失败'));
    }, n, {timeout: 240000});
    assert.equal((await stats()).trees, n, await page.locator('#editmsg').textContent());
  };
  const note = (name, detail) => {checks.push({name, detail}); console.log(JSON.stringify({name, detail}));};
  const rows = page.locator('#treetable tbody tr');
  try {
    await page.goto(urls.processing + 'step2');
    await page.waitForFunction(() => window.TreeLineTreeReviewApp?.state.viewer?.treeXYZ?.length > 0, {}, {timeout: 180000});
    await idle();
    // Recover only this isolated audit project's edits from interrupted runs.
    // This also exercises persisted undo after opening a new viewer.
    let recoveryCount = 0;
    while ((await stats()).journal > 0) {
      assert.ok(recoveryCount++ < 20);
      await page.locator('#btn-undo').click(); await idle();
    }
    const original = await stats();
    assert.ok(original.trees >= 2);
    // The same button now actually executes after two entities are selected.
    await rows.nth(0).click(); await rows.nth(1).click({modifiers: ['Control']});
    await page.locator('#btn-merge').click(); await count(original.trees - 1);
    assert.equal((await stats()).points, original.points);
    await page.reload(); await count(original.trees - 1);
    assert.equal((await stats()).points, original.points);
    await page.locator('#btn-undo').click(); await count(original.trees);
    assert.equal((await stats()).points, original.points);
    note('先 Ctrl 选两个再点合并：实际提交、刷新保存、撤销不丢点', original);
    // Cancelled confirmation must not change membership or journal.
    await rows.nth(0).click(); await rows.nth(1).click({modifiers: ['Control']});
    acceptDialog = false;
    await page.locator('#btn-merge').click(); await idle();
    assert.deepEqual(await stats(), original); acceptDialog = true;
    note('取消合并确认：点数、编号和编辑日志不变', true);
    // Repeated mode clicks with fewer than two selections must not toggle it off.
    await page.locator('#btn-selection-clear').click();
    await page.locator('#btn-merge').click(); await page.locator('#btn-merge').click();
    assert.equal(await page.locator('#btn-merge').getAttribute('aria-pressed'), 'true');
    assert.equal(await page.locator('#btn-edit-execute').isDisabled(), true);
    await rows.nth(0).click(); await page.keyboard.down('n');
    await rows.nth(1).click(); await page.keyboard.up('n');
    await page.locator('#btn-merge').click(); await count(original.trees - 1);
    assert.equal((await stats()).points, original.points);
    await page.locator('#btn-undo').click(); await count(original.trees);
    note('先开合并模式、按住 N 加选、再点击合并、撤销', true);
    await page.keyboard.press('Escape');
    await page.locator('#toggle-labels').uncheck();
    await page.locator('#btn-view-top').click();
    await rows.nth(0).click();
    const selectedId = Number(await rows.nth(0).getAttribute('data-tid'));
    const target = await page.evaluate(selectedId => {
      const v = window.TreeLineTreeReviewApp.state.viewer, r = v.canvasRect();
      for (let i=0;i<v.treeXYZ.length;i++) {
        if (!v.treeVisible[i] || Number(v.treeIds[i]) === selectedId) continue;
        const p = new v.THREE.Vector3(...v.treeXYZ[i].map((x,j)=>x-v.origin[j])).project(v.camera);
        const x = r.left+(p.x+1)*r.width/2, y = r.top+(1-p.y)*r.height/2;
        if (p.z < -1 || p.z > 1 || x < r.left+35 || x > r.right-35 || y < r.top+90 || y > r.bottom-35) continue;
        if (document.elementFromPoint(x,y) === v.canvas) return {x, y, treeId: Number(v.treeIds[i])};
      }
      throw Error('No unselected visible cloud point');
    }, selectedId);
    const draw = async (x1,y1,x2,y2) => {
      await page.locator('#btn-trim-lasso').click();
      await page.mouse.move(x1,y1); await page.mouse.down();
      await page.mouse.move(x2,y1,{steps:5}); await page.mouse.move(x2,y2,{steps:5});
      await page.mouse.move(x1,y2,{steps:5}); await page.mouse.move(x1,y1,{steps:5}); await page.mouse.up();
      await page.waitForFunction(()=>window.TreeLineTreeReviewApp.state.activePreview, {}, {timeout:240000});
      await idle();
      return page.evaluate(()=>window.TreeLineTreeReviewApp.state.activePreview);
    };
    const preview = await draw(target.x-22,target.y-22,target.x+22,target.y+22);
    assert.ok(preview.tree_ids.includes(target.treeId));
    assert.equal((await stats()).points, original.points);
    await page.locator('#btn-trim-confirm').click(); await idle();
    assert.equal((await stats()).points, original.points-preview.point_count);
    await page.reload(); await idle();
    await page.waitForFunction(()=>window.TreeLineTreeReviewApp.state.treeMap.size>0);
    assert.equal((await stats()).points, original.points-preview.point_count);
    await page.locator('#btn-undo').click(); await count(original.trees);
    assert.equal((await stats()).points, original.points);
    note('实际鼠标套索：跨未选对象、精确删除、刷新保存、撤销恢复', {selectedId,targetId:target.treeId,removed:preview.point_count});
    // Broad lasso can remove complete editable objects, not just <95% per tree.
    const r = await page.evaluate(()=>window.TreeLineTreeReviewApp.state.viewer.canvasRect().toJSON());
    const broad = await draw(r.left+2,r.top+2,r.right-2,r.bottom-2);
    await page.locator('#btn-trim-confirm').click(); await idle();
    assert.equal((await stats()).points, original.points-broad.point_count);
    await page.locator('#btn-undo').click(); await count(original.trees);
    assert.equal((await stats()).points, original.points);
    note('大范围套索：完整对象也能删除，圈外保留，撤销恢复', {removed:broad.point_count,fullObjects:broad.fully_removed_tree_ids});
    assert.deepEqual(errors,[]); assert.deepEqual(outbound,[]);
    await page.locator('#toggle-labels').check();
    await page.screenshot({path:process.env.REAL_OUTPUT+'/merge_lasso_verified.png',fullPage:true});
    note('无页面脚本错误、无外部网络请求', true);
    fs.writeFileSync(process.env.REAL_OUTPUT+'/merge_lasso_checks.json',JSON.stringify(checks,null,2));
    await page.locator('#btn-confirm').click(); await page.waitForURL('**/step3',{timeout:600000});
    note('验收项目恢复原成员并重新定稿', await page.url());
    fs.writeFileSync(process.env.REAL_OUTPUT+'/merge_lasso_checks.json',JSON.stringify(checks,null,2));
  } finally {await browser.close();}
})().catch(e => {console.error(e); process.exitCode=1;});
