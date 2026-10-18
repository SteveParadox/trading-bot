/* Apply persisted appearance before the app stylesheet and React render. */
/* Separate same-origin file: compatible with CSP script-src 'self'. */
(function initializeTheme() {
  var theme = "dark";
  try {
    if (window.localStorage.getItem("northstar-ui-theme") === "light") {
      theme = "light";
    }
  } catch (_) {
    // Private browsing / disabled storage: keep the default black theme.
  }
  document.documentElement.dataset.theme = theme;
  var themeColor = document.querySelector('meta[name="theme-color"]');
  if (themeColor) {
    themeColor.setAttribute("content", theme === "light" ? "#ffffff" : "#090b0d");
  }
})();
