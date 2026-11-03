/* 在 CSS 加载前应用偏好，避免深色模式打开新章节时闪白。 */
(() => {
  const key = 'ai-book-theme', valid = new Set(['auto', 'light', 'dark']);
  const system = matchMedia('(prefers-color-scheme: dark)');
  let preference = 'auto';
  try {const stored = localStorage.getItem(key); if (valid.has(stored)) preference = stored;} catch { /* 隐私模式仍可切换当前页面。 */ }
  function apply() {
    const theme = preference === 'auto' ? (system.matches ? 'dark' : 'light') : preference;
    document.documentElement.dataset.theme = theme;
    document.documentElement.dataset.themePreference = preference;
    document.querySelector('meta[name="theme-color"]')?.setAttribute('content', theme === 'dark' ? '#131b18' : '#f6f7f2');
    const select = document.querySelector('#theme-select');
    if (select) {
      select.value = preference;
      select.parentElement.title = '配色主题：' + select.selectedOptions[0].textContent;
      document.querySelector('#theme-icon').textContent = preference === 'auto' ? '◐' : theme === 'dark' ? '☾' : '☀';
    }
  }
  apply();
  system.addEventListener('change', () => {if (preference === 'auto') apply();});
  addEventListener('storage', event => {if (event.key === key) {preference = valid.has(event.newValue) ? event.newValue : 'auto'; apply();}});
  addEventListener('DOMContentLoaded', () => {
    apply();
    document.querySelector('#theme-select').addEventListener('change', event => {
      preference = event.target.value;
      try {localStorage.setItem(key, preference);} catch { /* 不阻断主题切换。 */ }
      apply();
    });
  });
})();
