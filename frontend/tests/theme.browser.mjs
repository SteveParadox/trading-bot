import assert from "node:assert/strict";
import { chromium } from "playwright";

const url = process.env.FRONTEND_TEST_URL ?? "http://127.0.0.1:5173";

function brightness(value) {
  const rgb = value.match(/[\d.]+/g)?.slice(0, 3).map(Number);
  assert.ok(rgb?.length === 3, `Expected an RGB CSS color, received ${value}`);
  return rgb.map((c) => {
    const v = c / 255;
    return v <= 0.04045 ? v / 12.92 : ((v + 0.055) / 1.055) ** 2.4;
  }).reduce((sum, c, index) => sum + c * [0.2126, 0.7152, 0.0722][index], 0);
}

function contrast(a, b) {
  const x = brightness(a);
  const y = brightness(b);
  return (Math.max(x, y) + 0.05) / (Math.min(x, y) + 0.05);
}

const browser = await chromium.launch({ headless: true });
try {
  for (const viewport of [{ width: 1280, height: 800 }, { width: 375, height: 812 }]) {
    const context = await browser.newContext({ viewport, reducedMotion: "reduce" });
    try {
      const page = await context.newPage();
      const pageErrors = [];
      page.on("pageerror", (error) => pageErrors.push(error.message));
      await page.goto(url, { waitUntil: "networkidle" });
      assert.equal(await page.locator("html").getAttribute("data-theme"), "dark");

      const toWhite = page.getByRole("button", { name: "Switch to light theme" });
      assert.equal(await toWhite.isVisible(), true, "Theme toggle must be visible");
      await toWhite.click();
      assert.equal(await page.locator("html").getAttribute("data-theme"), "light");
      assert.equal(await page.evaluate(() => getComputedStyle(document.body).backgroundColor), "rgb(255, 255, 255)");
      assert.equal(await page.evaluate(() => localStorage.getItem("northstar-ui-theme")), "light");
      assert.equal(await page.locator('meta[name="theme-color"]').getAttribute("content"), "#ffffff");

      const connect = page.getByRole("button", { name: "Connect securely" });
      await page.getByLabel("API key").fill("only-for-appearance-testing");
      await connect.hover();
      const colors = await connect.evaluate((element) => ({
        foreground: getComputedStyle(element).color,
        background: getComputedStyle(element).backgroundColor,
      }));
      assert.ok(contrast(colors.foreground, colors.background) >= 4.5,
        `Connection hover contrast too low: ${JSON.stringify(colors)}`);

      await page.reload({ waitUntil: "networkidle" });
      assert.equal(await page.locator("html").getAttribute("data-theme"), "light");
      const toBlack = page.getByRole("button", { name: "Switch to dark theme" });
      await toBlack.click();
      assert.equal(await page.locator("html").getAttribute("data-theme"), "dark");
      assert.deepEqual(pageErrors, [], `Browser errors at width ${viewport.width}`);
      console.log(`Theme browser smoke passed at width ${viewport.width}`);
    } finally {
      await context.close();
    }
  }
} finally {
  await browser.close();
}
