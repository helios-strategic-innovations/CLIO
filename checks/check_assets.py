"""Static packaging checks; no browser or network execution."""
from html.parser import HTMLParser
from pathlib import Path
import json
import re

root = Path(__file__).resolve().parents[1]
class Page(HTMLParser):
    def __init__(self):
        super().__init__(); self.ids=set(); self.refs=[]; self.buttons=[]; self.panels=[]
    def handle_starttag(self, tag, attrs):
        a=dict(attrs)
        if 'id' in a:
            assert a['id'] not in self.ids
            self.ids.add(a['id'])
        for k in ('href','src'):
            if k in a: self.refs.append((tag,a[k],a))
        if tag=='button': self.buttons.append(a)
        if 'data-panel' in a: self.panels.append(a['data-panel'])
        assert tag not in ('iframe','object','embed')
page=Page();page.feed((root/'index.html').read_text(encoding='utf-8-sig'))
assert page.panels == ['records','synthetic','about']
for tag,ref,a in page.refs:
    if ref == 'https://t.me/DN_Moltbot':
        assert tag=='a' and a.get('target')=='_blank' and 'noopener' in a.get('rel','').split() and 'noreferrer' in a.get('rel','').split()
    elif ref.startswith('#'):
        assert ref[1:] in page.ids
    else:
        assert not re.match(r'(?i)(https?:|//|javascript:|data:|file:)',ref),ref
        assert (root/ref).is_file(),ref
assert all(b.get('type')=='button' for b in page.buttons)
assert not re.search(r'@import|url\s*\(', (root/'styles.css').read_text(encoding='utf-8-sig'), re.I)
raw=(root/'records_data.js').read_text(encoding='utf-8-sig').split('const CLIO_RECORDED = ',1)[1].rstrip().removesuffix(';')
assert json.loads(raw)==json.loads((root/'META_EVIDENCE.json').read_text(encoding='utf-8-sig'))
cases=(root/'case_data.js').read_text(encoding='utf-8-sig').split('const CLIO_REPLAY = ',1)[1].rstrip().removesuffix(';')
assert len(json.loads(cases)['cases'])==4
print('PASS: 3 panels, local assets/references, sole explicit Telegram link, 6 exact sanitized records, 4 synthetic cases, no remote assets')
