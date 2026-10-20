import assert from 'node:assert/strict';
const { chromium } = await import(process.env.PLAYWRIGHT_MODULE ?? "playwright");
const browser = await chromium.launch({ headless: true });
const url = process.env.FRONTEND_TEST_URL ?? 'http://127.0.0.1:5173';
const performance = { trade_count:0,open_trade_count:0,win_rate:0,profit_factor:null,total_pnl:0,max_drawdown:{amount:0,pct:0},sharpe_ratio:0,sortino_ratio:0,average_r_multiple:0,financing:0 };
try {
  for (const width of [1440, 390]) {
    const page = await browser.newPage({ viewport: { width, height: 900 } });
    const errors = [];
    page.on('pageerror', error => errors.push(error.message));
    await page.routeWebSocket('**/ws/live**', () => {});
    await page.route('**/api/**', async route => {
      const path = new URL(route.request().url()).pathname;
      const bodies = {
        '/api/status': {state:'running',reason:'Demo testing',updated_at:new Date().toISOString()},
        '/api/config': {instruments:['EUR_USD'],dashboard_badge:'Demo only',strategy:{entry_timeframe:'15m',structure_timeframe:'1h',htf_timeframe:'4h',tp_timeframe:'5m'}},
        '/api/performance': performance,
        '/api/news': {state:'fresh',stale:false,events:[],blackouts:{},event_count:0},
        '/api/monitoring': {stale_prices:{},clock_health:{},recent_alerts:[],broker_connected:true,unknown_orders:0},
      };
      await route.fulfill({json:bodies[path] ?? []});
    });
    await page.goto(url);
    await page.getByLabel('API key', {exact:true}).fill('local-smoke-only');
    await page.getByRole('button',{name:'Connect securely'}).click();
    const strip = page.getByLabel('Active strategy timeframes');
    await strip.waitFor();
    await page.waitForFunction(() => document.querySelector('.timeframeStrip')?.textContent?.includes('15m'));
    const text = await strip.innerText();
    for (const expected of ['Entry 15m','Confirmation 1h','Primary bias 4h','TP reference 5m']) assert.ok(text.includes(expected), text);
    await page.getByRole('button',{name:'Switch to light theme'}).click();
    assert.ok(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth));
    if (width === 1440) await page.screenshot({path:'/tmp/fx-dashboard-smoke.png',fullPage:true});
    assert.deepEqual(errors, []);
    console.log(`Strategy roles and responsive dashboard passed at width ${width}`);
    await page.close();
  }
} finally { await browser.close(); }
