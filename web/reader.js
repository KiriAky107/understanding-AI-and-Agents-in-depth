import {marked} from './vendor/marked.esm.js';
import {protectMath, escapeHTML as esc} from './math.js?v=20261104.4';

const base = new URL('./', location.href);
const repo = 'https://github.com/KiriAky107/understanding-AI-and-Agents-in-depth';
const app = document.querySelector('#app');
const sourceCache = new Map();
let catalog, searchIndex, queryVersion = 0;
const urlFor = (id, hash = '') => '?chapter=' + encodeURIComponent(id) + (hash ? '#' + encodeURIComponent(hash) : '');
const byId = id => catalog.chapters.find(c => c.id === id);

async function source(chapter) {
  if (!sourceCache.has(chapter.id)) {
    sourceCache.set(chapter.id, fetch(new URL(chapter.file, base)).then(response => {
      if (!response.ok) throw Error('正文加载失败（' + response.status + '）');
      return response.text();
    }).catch(error => {sourceCache.delete(chapter.id); throw error;}));
  }
  return sourceCache.get(chapter.id);
}

// 标题标识保持稳定；显式题号锚点（如 RT-07）保留原大小写。
function assignHeadings(root) {
  const used = new Set([...root.querySelectorAll('[id]')].map(e => e.id));
  return [...root.querySelectorAll('h1,h2,h3,h4,h5,h6')].map(heading => {
    if (!heading.id) {
      const stem = heading.textContent.trim().toLowerCase().replace(/[^\p{L}\p{N}\s_-]/gu, '').replace(/\s/g, '-') || 'section';
      let id = stem, n = 0;
      while (used.has(id)) id = stem + '-' + (++n);
      heading.id = id; used.add(id);
    }
    return {id: heading.id, title: heading.textContent.trim(), level: Number(heading.tagName.slice(1))};
  });
}

function parseMarkdown(text) {
  const protectedMath = protectMath(text);
  const template = document.createElement('template');
  template.innerHTML = marked.parse(protectedMath.output).replace(/BOOKMATHTOKEN(\d+)END/g, (_, i) => protectedMath.math[Number(i)]);
  for (const p of template.content.querySelectorAll('p')) if (p.textContent.trim() === '[TOC]') p.remove();
  const headings = assignHeadings(template.content);
  return {root: template.content, headings};
}

function routeLink(raw, chapter) {
  if (!raw || raw.startsWith('#') || /^(mailto:|tel:)/i.test(raw)) return raw;
  let target;
  try { target = new URL(raw, new URL(chapter.file, base)); } catch { return raw; }
  let file;
  const githubPrefix = repo + '/blob/main/';
  if (target.href.startsWith(githubPrefix)) file = decodeURIComponent(target.pathname.split('/blob/main/')[1]);
  else if (target.origin === base.origin && target.pathname.startsWith(base.pathname)) file = decodeURIComponent(target.pathname.slice(base.pathname.length));
  if (file) {
    const destination = catalog.chapters.find(c => c.file === file);
    if (destination) return urlFor(destination.id, decodeURIComponent(target.hash.slice(1)));
    if (file.startsWith('现代AI示例/') || file.startsWith('LICENSE')) return repo + '/blob/main/' + file + target.hash;
  }
  return target.href;
}

function navLinks(active) {
  return catalog.chapters.map(c => '<a href="' + urlFor(c.id) + '"' + (c.id === active ? ' aria-current="page"' : '') + '><span class="nav-number">' + esc(c.label) + '</span><span>' + esc(c.shortTitle) + '</span></a>').join('');
}

function home() {
  document.title = '深入理解AI和Agent · 从线性代数开始的自学笔记';
  const chapters = catalog.chapters.map(c => '<a class="chapter-row" href="' + urlFor(c.id) + '"><span class="chapter-num">' + esc(c.label) + '</span><div><h3>' + esc(c.shortTitle) + '</h3><p>' + esc(c.description) + '</p></div><span class="chapter-arrow" aria-hidden="true">↗</span></a>').join('');
  app.innerHTML = `<main id="main" class="home">
    <section class="hero" aria-labelledby="book-title"><div>
      <div class="eyebrow">A SELF-STUDY NOTEBOOK · KIRIAKY107</div>
      <h1 id="book-title">深入理解<span>AI 和 Agent</span></h1>
      <p class="subtitle">从线性代数开始的自学笔记</p>
      <p class="hero-description">从向量与矩阵的几何变化出发，沿着手算、图解与 Python 实验，走进神经网络、大语言模型与智能体工程。</p>
      <div class="topic-tags" aria-label="主题标签">${catalog.tags.map(tag => '<button type="button" data-search="' + esc(tag) + '">' + esc(tag) + '</button>').join('')}</div>
      <div class="hero-buttons"><a class="button primary" href="?chapter=00">开始阅读 <span aria-hidden="true">→</span></a><a class="button" href="#chapters">浏览目录 <span aria-hidden="true">↓</span></a></div>
      <p class="hero-note">在线阅读全文 · 配套代码与练习题解 · 可下载 PDF</p>
    </div><div class="cover-figure" aria-label="由向量变换走向神经网络与智能体的示意">
      <svg viewBox="0 0 360 245" role="img" aria-label="方格中的两条基向量经过矩阵变换，进入网络与反馈循环">
        <defs><pattern id="grid" width="24" height="24" patternUnits="userSpaceOnUse"><path d="M24 0H0V24" fill="none" stroke="#d5dce3" stroke-width=".7"/></pattern><marker id="arrow" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="5" markerHeight="5" orient="auto-start-reverse"><path d="M0 0L10 5L0 10Z" fill="#2563a5"/></marker></defs>
        <rect x="0" y="0" width="360" height="245" fill="url(#grid)"/>
        <path d="M24 137H169M58 168V30" stroke="#a7b6c7" stroke-width="1"/>
        <path d="M58 137L130 137L154 65L82 65Z" fill="#aad2f5" fill-opacity=".35" stroke="#7a9cbe" stroke-width="1.3"/>
        <path d="M58 137H130M58 137L82 65" stroke="#2563a5" stroke-width="2.3" marker-end="url(#arrow)"/>
        <circle cx="58" cy="137" r="3" fill="#2563a5"/>
        <text x="133" y="154" fill="#536c87" font-family="Georgia,serif" font-size="15">e₁</text><text x="65" y="58" fill="#536c87" font-family="Georgia,serif" font-size="15">e₂</text>
        <path d="M176 96H210" stroke="#2563a5" marker-end="url(#arrow)"/>
        <g stroke="#8ba6c4" stroke-width="1.1"><path d="M232 59L282 83L232 114L282 143L232 166M232 59L282 143M232 166L282 83M282 83L329 113L282 143" fill="none"/></g>
        <g fill="#fbfcfe" stroke="#2563a5" stroke-width="1.6"><circle cx="232" cy="59" r="6"/><circle cx="232" cy="114" r="6"/><circle cx="232" cy="166" r="6"/><circle cx="282" cy="83" r="6"/><circle cx="282" cy="143" r="6"/><circle cx="329" cy="113" r="6"/></g>
        <path d="M329 134V207H232V183" stroke="#2563a5" stroke-width="1.2" fill="none" stroke-dasharray="3 4" marker-end="url(#arrow)"/>
        <text x="30" y="216" fill="#536c87" font-family="ui-monospace,monospace" font-size="13">y = Wx</text><text x="254" y="229" fill="#536c87" font-family="ui-monospace,monospace" font-size="10">act · observe</text>
      </svg><div class="cover-caption"><span>01 → 11</span><strong>表示 / 学习 / 生成 / 行动</strong></div>
    </div></section>
    <div class="book-meta"><span><strong>11</strong> 篇正文</span><span><strong>175</strong> 幅配图</span><span><strong>83</strong> 道练习题解</span><span><strong>466</strong> 页 PDF</span></div>
    <section aria-labelledby="routes-title"><div class="section-intro"><h2 id="routes-title">选一条阅读路线</h2><a href="?chapter=00" class="muted">完整学习路线 ↗</a></div>
      <div class="routes"><div class="route"><small>01 / 数学与学习</small><h3>先把底层算清楚</h3><p>从资料的数值表示，走到误差反传。把符号、几何图像和程序里的数组对上。</p><a href="?chapter=01">第一至三篇 →</a></div><div class="route"><small>02 / 模型与生成</small><h3>看清模型怎样工作</h3><p>逐位置追踪注意力、训练与生成，再进入微调部署以及视觉、语音和多模态。</p><a href="?chapter=04">第四至六篇 →</a></div><div class="route"><small>03 / 智能体工程</small><h3>让任务真正跑完</h3><p>把检索、工具和协议接成循环，再追问任务状态如何保存、恢复与验收。</p><a href="?chapter=07">第七至十篇 →</a></div></div>
    </section><section id="chapters" aria-labelledby="chapters-title"><div class="section-intro"><div><h2 id="chapters-title">全书目录</h2><p>序言 → 十一篇正文 → 后记，另附完整练习题解。</p></div><span class="eyebrow">CONTENTS</span></div><div class="chapter-list">${chapters}</div></section>
  </main>`;
  for (const button of app.querySelectorAll('[data-search]')) button.addEventListener('click', () => openSearch(button.dataset.search));
  document.documentElement.dataset.ready = 'home';
}

let mathReady;
async function typeset(article) {
  if (!mathReady) {
    window.MathJax = {tex: {inlineMath: [['\\(', '\\)']], displayMath: [['\\[', '\\]']]}, startup: {typeset: false}, options: {enableMenu: false}, svg: {fontCache: 'global'}};
    mathReady = new Promise((resolve, reject) => {
      const script = document.createElement('script'); script.src = new URL('web/vendor/mathjax.js', base).href;
      script.onload = resolve;
      script.onerror = () => {script.remove(); reject(Error('公式组件加载失败'));};
      document.head.append(script);
    }).then(() => window.MathJax.startup.promise).catch(error => {mathReady = null; throw error;});
  }
  await mathReady;
  await window.MathJax.typesetPromise([article]);
}

function prepareCode(root) {
  for (const pre of root.querySelectorAll('pre')) {
    const code = pre.querySelector('code');
    const text = code?.textContent || '';
    const requested = /(?:^|\s)language-(\S+)/.exec(code?.className || '')?.[1] || 'text';
    pre.dataset.language = requested;
    const label = document.createElement('span'); label.className = 'code-language'; label.textContent = requested.toUpperCase(); label.setAttribute('aria-hidden', 'true');
    const copy = document.createElement('button'); copy.type = 'button'; copy.className = 'copy-code'; copy.textContent = '复制代码';
    copy.addEventListener('click', async () => {
      try {await navigator.clipboard.writeText(text); copy.textContent = '已复制';}
      catch {copy.textContent = '请选中代码复制';}
      setTimeout(() => copy.textContent = '复制代码', 1800);
    });
    pre.append(label, copy);
  }
}

async function highlightCode(root) {
  const blocks = [...root.querySelectorAll('pre:not(.shiki)')];
  if (!blocks.length) return;
  const {highlighter} = await import('./vendor/shiki.js?v=20261104.4');
  const shiki = await highlighter();
  for (const original of blocks) {
    const text = original.querySelector('code')?.textContent || '';
    const requested = original.dataset.language;
    const language = shiki.getLoadedLanguages().includes(requested) ? requested : 'text';
    const template = document.createElement('template');
    template.innerHTML = shiki.codeToHtml(text, {lang: language, themes: {light: 'github-light', dark: 'github-dark'}});
    const pre = template.content.firstElementChild;
    // 只改着色，不改变代码字符；复制控件在高亮失败时也能使用。
    if (pre.querySelector('code').textContent !== text) throw Error('代码高亮改变了文本');
    pre.dataset.language = language;
    pre.append(...original.querySelectorAll('.code-language, .copy-code'));
    original.replaceWith(pre);
  }
}

async function enhance(stage, action, message) {
  document.documentElement.dataset[stage] = 'loading';
  document.getElementById('notice-' + stage)?.remove();
  try {
    await action();
    document.documentElement.dataset[stage] = 'ready';
  } catch (error) {
    document.documentElement.dataset[stage] = 'error';
    const note = document.createElement('p');
    note.id = 'notice-' + stage; note.className = 'render-notice'; note.setAttribute('role', 'status');
    note.textContent = message;
    if (stage === 'math') {
      const retry = document.createElement('button'); retry.type = 'button'; retry.textContent = '重试公式排版';
      retry.addEventListener('click', () => enhance(stage, action, message)); note.append(' ', retry);
    }
    app.querySelector('.reader-meta').after(note);
    console.error(stage + ':', error);
  }
}

function scrollToHash() {
  if (!location.hash) return;
  try {document.getElementById(decodeURIComponent(location.hash.slice(1)))?.scrollIntoView();} catch { /* 保留正文阅读，不因无效 URL 中断。 */ }
}

async function reader(chapter) {
  document.title = chapter.shortTitle + ' · 深入理解AI和Agent';
  const text = await source(chapter), parsed = parseMarkdown(text);
  const sectionHeadings = parsed.headings.filter(h => h.level === 2 || h.level === 3);
  const toc = sectionHeadings.map(h => '<a data-level="' + h.level + '" href="#' + encodeURIComponent(h.id) + '">' + esc(h.title) + '</a>').join('');
  const i = catalog.chapters.indexOf(chapter), previous = catalog.chapters[i - 1], next = catalog.chapters[i + 1];
  const pager = (c, label) => c ? '<a href="' + urlFor(c.id) + '"><small>' + label + '</small><span>' + esc(c.shortTitle) + '</span></a>' : '<span></span>';
  app.innerHTML = `<div class="reading-progress" aria-hidden="true"></div><div class="reader-layout">
    <nav class="chapter-nav" aria-label="分章导航"><p class="nav-title">全书目录</p><a class="nav-home" href="./">← 返回首页</a>${navLinks(chapter.id)}</nav>
    <main id="main" class="article-wrap"><details class="mobile-chapters"><summary>全书目录 · ${esc(chapter.label)}</summary><nav aria-label="移动端分章导航">${navLinks(chapter.id)}</nav></details>
      <div class="reader-meta"><span>自学笔记 / ${esc(chapter.label)} / ${esc(chapter.group)}</span><a href="${repo}/blob/main/${encodeURI(chapter.file)}" target="_blank" rel="noopener">查看原稿 ↗</a></div>
      <details class="mobile-toc"><summary>本篇目录 · ${sectionHeadings.length} 个标题</summary><nav>${toc}</nav></details>
      <article class="article"></article><nav class="chapter-pager" aria-label="篇章翻页">${pager(previous, '← 上一篇')}${pager(next, '下一篇 →')}</nav>
    </main><nav class="page-toc" aria-label="本篇目录"><p class="nav-title">本篇目录</p>${toc}</nav></div><button class="back-top" type="button" aria-label="回到顶部">↑</button>`;
  const article = app.querySelector('article');
  for (const a of parsed.root.querySelectorAll('a[href]')) {
    a.setAttribute('href', routeLink(a.getAttribute('href'), chapter));
    if (/^https?:/.test(a.getAttribute('href')) && new URL(a.href).origin !== base.origin) { a.target = '_blank'; a.rel = 'noopener'; }
  }
  for (const img of parsed.root.querySelectorAll('img')) {
    img.src = new URL(img.getAttribute('src'), new URL(chapter.file, base)).href;
    img.loading = 'lazy'; img.decoding = 'async'; img.tabIndex = 0;
    const figure = document.createElement('figure'), caption = document.createElement('figcaption');
    caption.textContent = img.alt; img.replaceWith(figure); figure.append(img, caption);
    img.addEventListener('click', () => openImage(img));
    img.addEventListener('keydown', e => {if (e.key === 'Enter') openImage(img);});
  }
  for (const table of parsed.root.querySelectorAll('table')) {
    const wrapper = document.createElement('div'); wrapper.className = 'table-scroll';
    wrapper.tabIndex = 0; wrapper.setAttribute('role', 'region'); wrapper.setAttribute('aria-label', '可横向滚动的表格');
    table.replaceWith(wrapper); wrapper.append(table);
  }
  // 先呈现正文，再分别增强；任何一个组件失败都不能让整篇空白。
  article.append(parsed.root);
  prepareCode(article);
  app.querySelector('.back-top').addEventListener('click', () => scrollTo({top: 0, behavior: 'instant'}));
  for (const a of app.querySelectorAll('.mobile-toc a')) a.addEventListener('click', () => app.querySelector('.mobile-toc').open = false);
  const progress = app.querySelector('.reading-progress');
  const updateProgress = () => {progress.style.width = Math.min(100, 100 * scrollY / Math.max(1, document.documentElement.scrollHeight - innerHeight)) + '%';};
  addEventListener('scroll', updateProgress, {passive: true}); updateProgress();
  const observer = new IntersectionObserver(entries => {
    const visible = entries.find(e => e.isIntersecting);
    if (!visible) return;
    for (const a of app.querySelectorAll('.page-toc a')) a.classList.toggle('active', decodeURIComponent(a.hash.slice(1)) === visible.target.id);
  }, {rootMargin: '-90px 0px -65% 0px'});
  for (const heading of article.querySelectorAll('h2,h3')) observer.observe(heading);
  await Promise.all([
    enhance('math', () => typeset(article), '公式组件暂未载入，当前保留公式源码。'),
    enhance('highlight', () => highlightCode(article), '代码高亮暂未载入，代码仍可阅读和复制。'),
  ]);
  // SVG 和 PNG 加载完成后再跳锚点，避免图片撑开导致跳转位置漂移。
  if (location.hash) {
    await Promise.all([...article.querySelectorAll('img')].map(img => {img.loading = 'eager'; return img.decode().catch(() => {});}));
    scrollToHash();
  }
  document.documentElement.dataset.ready = chapter.id;
}

function openImage(img) {
  const dialog = document.querySelector('#image-dialog');
  document.querySelector('#image-large').src = img.src;
  document.querySelector('#image-large').alt = img.alt;
  document.querySelector('#image-caption').textContent = img.alt;
  dialog.showModal();
}

async function makeSearchIndex() {
  const sections = await Promise.all(catalog.chapters.map(async chapter => {
    const {root} = parseMarkdown(await source(chapter));
    const results = []; let current = {chapter, id: '', title: chapter.shortTitle, text: ''};
    for (const node of root.children) {
      if (/^H[1-6]$/.test(node.tagName)) {
        if (current.text.trim()) results.push(current);
        current = {chapter, id: node.id, title: node.textContent, text: ''};
      } else current.text += ' ' + node.textContent.replace(/\s+/g, ' ');
    }
    if (current.text.trim()) results.push(current);
    return results;
  }));
  return sections.flat();
}

function highlight(text, terms) {
  const pattern = new RegExp('(' + terms.map(t => t.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')).join('|') + ')', 'gi');
  return text.split(pattern).map((part, i) => i % 2 ? '<mark>' + esc(part) + '</mark>' : esc(part)).join('');
}

async function search() {
  const version = ++queryVersion, query = document.querySelector('#search-input').value.trim();
  const status = document.querySelector('#search-status'), output = document.querySelector('#search-results');
  output.replaceChildren();
  if (!query) {status.textContent = '搜索包含正文与练习题解。'; return;}
  status.textContent = '正在检索…';
  try {
    if (!searchIndex) searchIndex = makeSearchIndex().catch(error => {searchIndex = null; throw error;});
    const sections = await searchIndex;
    if (version !== queryVersion) return;
    const terms = query.toLowerCase().split(/\s+/).filter(Boolean);
    const matches = sections.map(s => ({...s, score: terms.reduce((sum, t) => sum + (s.title.toLowerCase().includes(t) ? 5 : 0), 0)}))
      .filter(s => terms.every(t => (s.title + ' ' + s.text).toLowerCase().includes(t))).sort((a, b) => b.score - a.score);
    status.textContent = matches.length ? '找到 ' + matches.length + ' 个相关小节' + (matches.length > 40 ? '，先显示前 40 项。' : '。') : '没有找到相关内容，试试更短的关键词或题号。';
    output.innerHTML = matches.slice(0, 40).map(s => {
      const pos = Math.max(0, s.text.toLowerCase().indexOf(terms[0])), start = Math.max(0, pos - 35);
      const snippet = (start ? '…' : '') + s.text.slice(start, start + 145) + (s.text.length > start + 145 ? '…' : '');
      return '<a class="search-result" href="' + urlFor(s.chapter.id, s.id) + '"><small>' + esc(s.chapter.shortTitle) + '</small><strong>' + highlight(s.title, terms) + '</strong><p>' + highlight(snippet, terms) + '</p></a>';
    }).join('');
  } catch {status.textContent = '正文暂时无法载入，请检查网络后重试。';}
}

function openSearch(query) {
  const dialog = document.querySelector('#search-dialog');
  if (!dialog.open) dialog.showModal();
  if (query !== undefined) document.querySelector('#search-input').value = query;
  document.querySelector('#search-input').focus(); search();
}

document.querySelector('#search-open').addEventListener('click', () => openSearch());
let debounce;
document.querySelector('#search-input').addEventListener('input', () => {clearTimeout(debounce); debounce = setTimeout(search, 180);});
addEventListener('keydown', event => {if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === 'k') {event.preventDefault(); openSearch();}});
for (const dialog of document.querySelectorAll('dialog')) dialog.addEventListener('click', event => {if (event.target === dialog) {const r = dialog.getBoundingClientRect(); if (event.clientX < r.left || event.clientX > r.right || event.clientY < r.top || event.clientY > r.bottom) dialog.close();}});
addEventListener('hashchange', scrollToHash);

try {
  const response = await fetch(new URL('web/catalog.json', base));
  if (!response.ok) throw Error('目录加载失败');
  catalog = await response.json();
  const id = new URLSearchParams(location.search).get('chapter');
  if (id === null) home();
  else {
    const chapter = byId(id);
    if (!chapter) throw Error('没有找到这一篇，请返回首页选择章节。');
    await reader(chapter);
  }
} catch (error) {
  if (app.querySelector('article')) {
    const note = document.createElement('p'); note.className = 'render-notice'; note.textContent = '阅读组件加载中断，已载入的正文仍可阅读。'; app.querySelector('.reader-meta').after(note);
  } else app.innerHTML = '<main id="main" class="loading"><h1>暂时无法打开</h1><p>' + esc(error.message) + '</p><a href="./">返回首页</a></main>';
  console.error(error); document.documentElement.dataset.ready = 'error';
}
