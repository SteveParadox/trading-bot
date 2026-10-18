import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { test } from "node:test";
import { fileURLToPath } from "node:url";
import { runInNewContext } from "node:vm";

const frontend = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const boot = readFileSync(resolve(frontend, "public/theme-init.js"), "utf8");
const html = readFileSync(resolve(frontend, "index.html"), "utf8");
const styles = readFileSync(resolve(frontend, "src/styles.css"), "utf8");

function loadTheme(value, storageDenied = false) {
  const meta = { content: "#090b0d", setAttribute(name, value) { this[name] = value; } };
  const document = {
    documentElement: { dataset: {} },
    querySelector(selector) {
      assert.equal(selector, 'meta[name="theme-color"]');
      return meta;
    },
  };
  const window = {
    localStorage: {
      getItem(key) {
        assert.equal(key, "northstar-ui-theme");
        if (storageDenied) throw new Error("Storage denied");
        return value;
      },
    },
  };
  runInNewContext(boot, { document, window });
  return { theme: document.documentElement.dataset.theme, themeColor: meta.content };
}

test("defaults to black and restores only an explicitly saved white theme", () => {
  assert.deepEqual(loadTheme(null), { theme: "dark", themeColor: "#090b0d" });
  assert.deepEqual(loadTheme("light"), { theme: "light", themeColor: "#ffffff" });
  assert.deepEqual(loadTheme("dark"), { theme: "dark", themeColor: "#090b0d" });
  assert.deepEqual(loadTheme("unexpected-value"), { theme: "dark", themeColor: "#090b0d" });
});

test("continues to use black when browser storage is unavailable", () => {
  assert.deepEqual(loadTheme(null, true), { theme: "dark", themeColor: "#090b0d" });
});

test("loads a same-origin script synchronously before the React entry point", () => {
  assert.match(html, /<script src="\/theme-init\.js"><\/script>/);
  assert.doesNotMatch(html, /<script>\s*try/);
  assert.ok(html.indexOf('src="/theme-init.js"') < html.indexOf('src="/src/main.tsx"'));
});

test("provides theme-aware foreground, background and hover colors", () => {
  assert.match(styles, /:root\[data-theme="light"\]/);
  assert.match(styles, /:root\[data-theme="dark"\]/);
  assert.match(styles, /\.connectButton:hover:not\(:disabled\)[\s\S]+?background: var\(--teal-dark\);/);
  assert.match(styles, /\.themeToggle:focus-visible/);
  assert.match(styles, /@media \(max-width: 460px\)/);
});
