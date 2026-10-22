const {chromium} = require(process.env.PLAYWRIGHT_MODULE);
const fs = require('node:fs'), assert = require('node:assert/strict');
(async () => {
  const urls = JSON.parse(fs.readFileSync(process.env.REAL_SERVERS, 'utf8'));
  const browser = await chromium.launch({channel: 'msedge', headless: true});
  const page = await browser.newPage({viewport: {width: 1680, height: 1050}});
  const errors = [], checks = [], outgoing = [];
  page.on('pageerror', e => errors.push(e.message));
  await page.route('**/*', route => {
    const url = new URL(route.request().url());
    if (['http:', 'https:'].includes(url.protocol) && !['127.0.0.1', 'localhost'].includes(url.hostname)) {
      outgoing.push(url.href); return route.abort();
    }
    return route.continue();
  });
  const frame = () => page.frames().find(f => new URL(f.url() || 'about:blank').pathname === '/step2');
  let originalThreshold;
  try {
    await page.goto(urls.desktop + '&browser-test=1');
    await page.waitForFunction(() => document.querySelector('#resultBox')?.textContent.includes('已有'), {}, {timeout: 180000});
    await page.locator('#ribbonReview').click();
    await page.frameLocator('#viewer').locator('#treetable tbody tr').first().waitFor({timeout: 180000});
    originalThreshold = Number(await page.locator('#riskThreshold').inputValue());
    const review = frame();
    const info = () => review.evaluate(() => window.TreeLineTreeReviewApp.state.viewer.dangerLayerInfo());
    const originalPoints = await review.evaluate(() => [...window.TreeLineTreeReviewApp.state.treeMap.values()].reduce((n, t) => n + t.point_count, 0));
    await review.locator('#btn-toggle-danger').click();
    const initial = await info();
    assert.equal(initial.danger_authority, 'EXACT_CURRENT_THRESHOLD_FULL_RESOLUTION');
    assert.equal(initial.threshold_m, originalThreshold);
    const boundaries = await review.evaluate(() => window.TreeLineTreeReviewApp.state.viewer._thresholdBoundaries.length);
    assert.equal(boundaries, 3, 'the three measured B conductors were not all displayed');
    checks.push({name: '真实点云危险点、导线阈值边界', initial, boundaries, originalPoints});
    const apply = async value => {
      await page.locator('#riskThreshold').fill(String(value));
      await page.locator('#saveRisk').click();
      await review.waitForFunction(v => window.TreeLineTreeReviewApp.state.viewer.dangerLayerInfo().threshold_m === v, value, {timeout: 180000});
      assert.equal(frame(), review, 'threshold update navigated or replaced the review frame');
      const current = await info();
      assert.ok(await page.locator('#viewerWrap').isVisible());
      assert.ok(current.visible, 'threshold update hid the requested danger layer');
      const points = await review.evaluate(() => [...window.TreeLineTreeReviewApp.state.treeMap.values()].reduce((n, t) => n + t.point_count, 0));
      assert.equal(points, originalPoints);
      return current;
    };
    const small = await apply(.1), large = await apply(1.5), restored = await apply(originalThreshold);
    assert.ok(small.danger_point_count <= initial.danger_point_count);
    assert.ok(large.danger_point_count >= initial.danger_point_count);
    assert.equal(restored.danger_point_count, initial.danger_point_count);
    checks.push({name: '阈值调小、调大、恢复：危险点联动且树木点数不变', small, large, restored});
    for (const id of ['btn-view-reset', 'btn-view-top', 'btn-view-side']) await review.locator('#' + id).click();
    for (const value of ['WARNING', 'DANGER', 'REVIEW_REQUIRED', 'ALL']) await review.locator('#clearance-risk-filter').selectOption(value);
    for (const id of ['btn-split', 'btn-trim-component', 'btn-trim-lasso']) {
      await review.locator('#' + id).click(); await review.locator('body').press('Escape');
    }
    await review.locator('#btn-review-back').click();
    await page.locator('#viewerWrap').waitFor({state: 'hidden'});
    await page.locator('#ribbonReview').click();
    await page.frameLocator('#viewer').locator('#treetable tbody tr').first().waitFor({timeout: 180000});
    checks.push({name: '危险点筛选、视角、编辑模式进入取消、退出并重新打开', passed: true});
    assert.deepEqual(errors, []); assert.deepEqual(outgoing, []);
    checks.push({name: '页面错误和外网请求', errors, outgoing});
    await page.screenshot({path: process.env.REAL_OUTPUT + '/threshold_review.png', fullPage: true});
  } catch (error) {
    const diagnostic = {error: String(error), errors, outgoing,
      frames: await Promise.all(page.frames().map(async f => ({url: f.url(), body: (await f.locator('body').innerText().catch(() => '')).slice(-7000)})))};
    fs.writeFileSync(process.env.REAL_OUTPUT + '/threshold_failure.json', JSON.stringify(diagnostic, null, 2));
    await page.screenshot({path: process.env.REAL_OUTPUT + '/threshold_failure.png', fullPage: true});
    throw error;
  } finally {
    if (originalThreshold !== undefined) {
      const response = await page.request.post(urls.processing + 'api/desktop/risk-policy', {data: {warning_clearance_threshold_m: originalThreshold}});
      assert.ok(response.ok(), 'restore acceptance threshold failed');
    }
    fs.writeFileSync(process.env.REAL_OUTPUT + '/threshold_checks.json', JSON.stringify(checks, null, 2));
    await browser.close();
  }
  console.log(JSON.stringify(checks));
})().catch(e => {console.error(e); process.exitCode = 1;});
