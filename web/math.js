/* 保留代码围栏与行内代码；在 Markdown 解析前保护数学标记与反斜杠。 */
export const escapeHTML = s => s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');

export function protectMath(source) {
  const segments = [];
  let opened = null, buffer = '', currentCode = false;
  for (const line of source.match(/[^\n]*\n|[^\n]+$/g) || []) {
    const m = /^ {0,3}([\x60~]{3,})(.*)/.exec(line);
    if (!opened && m && new Set(m[1]).size === 1) {
      if (buffer) segments.push({code: currentCode, text: buffer});
      buffer = line; currentCode = true; opened = m[1];
    } else if (opened) {
      buffer += line;
      if (m && new Set(m[1]).size === 1 && m[1][0] === opened[0] && m[1].length >= opened.length && !m[2].trim()) {
        segments.push({code: true, text: buffer}); buffer = ''; currentCode = false; opened = null;
      }
    } else buffer += line;
  }
  if (opened) throw Error('代码围栏未闭合');
  if (buffer) segments.push({code: currentCode, text: buffer});
  const math = [];
  const pattern = /(\x60+)[\s\S]*?\1|(?<!\\)\$\$([\s\S]*?)(?<!\\)\$\$|(?<!\\)\$((?:\\.|[^$\\\n])+)\$/g;
  const output = segments.map(seg => seg.code ? seg.text : seg.text.replace(pattern, (raw, code, display, inline) => {
    if (code) return raw;
    const index = math.length, block = display !== undefined;
    math.push('<span' + (block ? ' class="math-display"' : '') + '>' +
      (block ? '\\[' : '\\(') + escapeHTML(block ? display : inline) + (block ? '\\]' : '\\)') + '</span>');
    return 'BOOKMATHTOKEN' + index + 'END';
  })).join('');
  return {output, math};
}
