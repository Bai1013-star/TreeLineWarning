const {chromium} = require(process.env.PLAYWRIGHT_MODULE);
const fs = require('node:fs'), assert = require('node:assert/strict');
(async () => {
  const urls = JSON.parse(fs.readFileSync(process.env.REAL_SERVERS, 'utf8'));
  const browser = await chromium.launch({channel:'msedge', headless:true});
  const page = await browser.newPage({viewport:{width:1800,height:1300}});
  const errors = [], checks = [];
  page.on('pageerror', e => errors.push(e.message));
  page.on('dialog', d => d.accept());
  const note = (name, detail) => { checks.push({name,detail}); console.log(JSON.stringify({name,detail})); };
  const selected = () => page.evaluate(() => [...window.TreeLineTreeReviewApp.state.selected]);
  const waitCount = async n => {
    await page.waitForFunction(n => { const s = window.TreeLineTreeReviewApp.state; return !s.editingBusy && !s.mutation && (s.treeMap.size===n || document.querySelector('#editmsg').textContent.startsWith('失败')); }, n, {timeout:180000});
    assert.equal(await page.evaluate(()=>window.TreeLineTreeReviewApp.state.treeMap.size),n,await page.locator('#editmsg').textContent());
  };
  const sum = () => page.evaluate(() => [...window.TreeLineTreeReviewApp.state.treeMap.values()].reduce((n,t) => n+t.point_count,0));
  try {
    await page.goto(urls.processing + 'step2');
    await page.waitForFunction(() => window.TreeLineTreeReviewApp?.state.treeMap.size>0, {}, {timeout:180000});
    const originalTrees = await rowsCount();
    await waitCount(originalTrees);
    const initialCount = await sum();
    const initialJournal = await page.evaluate(() => window.TreeLineTreeReviewApp.state.committedJournalCount);
    const rows = page.locator('#treetable tbody tr');
    const firstId = Number(await rows.nth(0).getAttribute('data-tid'));
    const secondId = Number(await rows.nth(1).getAttribute('data-tid'));
    assert.equal(await page.locator('#multi-select-key option').count(),26);
    await rows.nth(0).click(); await rows.nth(1).click();
    assert.deepEqual(await selected(),[secondId]);
    await rows.nth(0).click({modifiers:['Control']});
    assert.equal((await selected()).length,2);
    await rows.nth(0).click({modifiers:['Control']});
    assert.deepEqual(await selected(),[secondId]);
    note('普通单选 / Ctrl 加选和取消',true);
    await page.locator('#btn-selection-clear').click();
    await rows.nth(0).click(); await page.keyboard.down('n'); await rows.nth(1).click(); await page.keyboard.up('n');
    assert.equal((await selected()).length,2);
    await page.locator('#multi-select-key').selectOption('Q');
    await page.locator('#btn-selection-clear').click();
    await rows.nth(0).click(); await page.keyboard.down('q'); await rows.nth(1).click(); await page.keyboard.up('q');
    assert.equal((await selected()).length,2);
    await page.reload(); await waitCount(originalTrees);
    assert.equal(await page.locator('#multi-select-key').inputValue(),'Q');
    note('按住 N / 自定义 Q / 重载保留按键',true);
    await page.locator('#btn-selection-clear').click();
    for (const id of ['btn-merge','btn-split','btn-trim-component']) {
      await page.locator('#'+id).click();
      assert.equal(await page.locator('#'+id).getAttribute('aria-pressed'),'true');
      assert.equal(await page.evaluate(() => window.TreeLineTreeReviewApp.state.committedJournalCount),initialJournal);
      await page.keyboard.press('Escape');
      assert.equal(await page.locator('#'+id).getAttribute('aria-pressed'),'false');
    }
    note('三种模式只选择不提交 / Esc取消',true);
    const editIds=await page.evaluate(()=>[...window.TreeLineTreeReviewApp.state.treeMap.values()].sort((a,b)=>b.point_count-a.point_count).slice(0,2).map(t=>Number(t.tree_id)));
    const centers = await page.evaluate(ids => {
      const v = window.TreeLineTreeReviewApp.state.viewer;
      return ids.map(id => {
        const points=v.treeXYZ.filter((p,i)=>Number(v.treeIds[i])===id);
        const center=[0,1,2].map(axis=>points.reduce((n,p)=>n+p[axis],0)/points.length);
        return points.reduce((best,p)=>Math.hypot(...p.map((x,j)=>x-center[j]))<Math.hypot(...best.map((x,j)=>x-center[j]))?p:best);
      });
    },editIds);
    await page.locator('#btn-merge').click();
    await page.locator(`#treetable tr[data-tid="${editIds[0]}"]`).click();
    await page.locator(`#treetable tr[data-tid="${editIds[1]}"]`).click({modifiers:['Control']});
    assert.equal(await sum(),initialCount);
    // Enter while the mode button is focused executes, instead of toggling off.
    await page.locator('#btn-merge').focus(); await page.keyboard.press('Enter'); await waitCount(originalTrees - 1);
    assert.equal(await sum(),initialCount);
    note('真实点云 Ctrl 两编号合并 / Enter 执行 / 不丢点',true);
    await page.locator('#btn-split').click();
    await page.locator('#toggle-labels').uncheck();
    await page.locator('#btn-view-top').click();
    await page.evaluate(()=>{window.TreeLineTreeReviewApp.state.viewer.controls.enableDamping=false;});
    const screenPoint = async center => page.evaluate(center => {
      const v=window.TreeLineTreeReviewApp.state.viewer, rect=v.canvasRect();
      const candidates=v.treeXYZ.map((p,i)=>({p,i,d:Math.hypot(p[0]-center[0],p[1]-center[1])})).filter(c=>v.treeVisible[c.i]).sort((a,b)=>a.d-b.d);
      for (const c of candidates.slice(0,300)) {
        const p=new v.THREE.Vector3(...c.p.map((x,j)=>x-v.origin[j])).project(v.camera);
        const x=rect.left+(p.x+1)*rect.width/2,y=rect.top+(1-p.y)*rect.height/2;
        if (p.z<-1 || p.z>1 || x<rect.left || x>rect.right || y<rect.top || y>rect.bottom) continue;
        if (document.elementFromPoint(x,y)!==v.canvas) continue;
        const hit=v.hitAt(x,y);
        if (hit && Number(hit.treeId)===Number(v.treeIds[c.i])) return {x,y,hit};
      }
      throw new Error('没有找到可点击的树木点');
    },center);
    const first=await screenPoint(centers[0]); await page.mouse.click(first.x,first.y);
    note('第一个分开位置',{picks:await page.evaluate(()=>window.TreeLineTreeReviewApp.state.editPicks),mode:await page.evaluate(()=>window.TreeLineTreeReviewApp.state.operationMode),message:await page.locator('#editmsg').textContent()});
    const second=await screenPoint(centers[1]); await page.keyboard.down('Control'); await page.mouse.click(second.x,second.y); await page.keyboard.up('Control');
    note('两个点击位置诊断',{first,second,picks:await page.evaluate(()=>window.TreeLineTreeReviewApp.state.editPicks),message:await page.locator('#editmsg').textContent()});
    assert.equal(await page.evaluate(()=>window.TreeLineTreeReviewApp.state.editPicks.length),2);
    assert.equal(await page.locator('#btn-edit-execute').isEnabled(),true);
    await page.locator('#btn-edit-execute').click(); await waitCount(originalTrees);
    assert.equal(await sum(),initialCount);
    note('真实点云同一编号两中心手动分开 / 所有点保留',true);
    await page.locator('#btn-undo').click(); await waitCount(originalTrees - 1);
    await page.locator('#btn-undo').click(); await waitCount(originalTrees);
    assert.equal(await sum(),initialCount);
    note('分开与合并撤销恢复完整成员',true);
    // Component picking collects positions locally. No preview or deletion occurs yet.
    await page.locator('#btn-trim-component').click();
    const one=await screenPoint(centers[0]); await page.mouse.click(one.x,one.y);
    const two=await screenPoint(centers[1]); await page.keyboard.down('Control'); await page.mouse.click(two.x,two.y); await page.keyboard.up('Control');
    assert.equal(await page.evaluate(()=>window.TreeLineTreeReviewApp.state.editPicks.length),2);
    assert.equal(await page.evaluate(()=>Boolean(window.TreeLineTreeReviewApp.state.activePreview)),false);
    assert.equal(await sum(),initialCount);
    await page.keyboard.press('Escape');
    assert.equal(await page.evaluate(()=>window.TreeLineTreeReviewApp.state.editPicks.length),0);
    note('团块 Ctrl 多位置选择只作标记 / Esc清除',true);
    assert.deepEqual(errors,[]);
    await page.locator('#toggle-labels').check();
    await page.locator('#multi-select-key').selectOption('N');
    await page.screenshot({path:process.env.REAL_OUTPUT+'/edit_modes_20261001.png',fullPage:true});
    fs.writeFileSync(process.env.REAL_OUTPUT+'/edit_modes_checks_20261001.json',JSON.stringify(checks,null,2));
    await page.locator('#btn-confirm').click();
    await page.waitForURL('**/step3',{timeout:600000});
    note('独立验收项目恢复后重新定稿',true);
  } finally { await browser.close(); }
  async function rowsCount(){return page.locator('#treetable tbody tr').count();}
})().catch(e => {console.error(e);process.exitCode=1;});
