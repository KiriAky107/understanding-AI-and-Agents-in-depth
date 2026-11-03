"""先发现 Skill 元数据，命中后才读取正文与模板。"""
from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parents[1]/'extensions'/'reading-plugin'/'skills'


def discover():
    result = []
    for path in ROOT.glob('*/SKILL.md'):
        # YAML 位于两个 --- 之间，正文继续保持 Markdown。
        parts = path.read_text(encoding='utf-8').split('---', 2)
        meta = yaml.safe_load(parts[1])
        result.append({'name':meta['name'],'description':meta['description']})
    return result


def load(name):
    # 按已发现名称选择路径，不接受任意路径输入。
    allowed = {item['name'] for item in discover()}
    if name not in allowed:
        raise ValueError('没有可加载的 Skill')
    directory = ROOT/name
    return {'body':(directory/'SKILL.md').read_text(encoding='utf-8').split('---',2)[2].strip(),
            'template':(directory/'references'/'template.txt').read_text(encoding='utf-8')}
