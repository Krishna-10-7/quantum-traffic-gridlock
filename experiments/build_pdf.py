"""Render REPORT.md (+ figures) and AI_DISCLOSURE.md to PDF with a local headless Chrome/Edge.

    python experiments/build_pdf.py          # -> REPORT.pdf, AI_DISCLOSURE.pdf
"""
import base64
import shutil
import subprocess
import sys
from pathlib import Path

import markdown

ROOT = Path(__file__).resolve().parents[1]
CSS = """
@page { size: A4; margin: 11mm 13mm; }
body { font-family: 'Segoe UI', Arial, sans-serif; font-size: 8.2pt; line-height: 1.27; color: #111; }
h1 { font-size: 13.5pt; margin: 0 0 4px; } h2 { font-size: 10.2pt; margin: 9px 0 3px; border-bottom: 1px solid #bbb; }
p, li { margin: 2px 0; } ul { margin: 2px 0 2px 16px; padding: 0; }
table { border-collapse: collapse; margin: 2px 0 4px; font-size: 7.4pt; }
th, td { border: 1px solid #bbb; padding: 1px 5px; } th { background: #eee; }
code { font-size: 7.8pt; background: #f3f3f3; padding: 0 2px; }
img { width: 88%; margin: 1px 0; }
.fig { font-size: 7.6pt; color: #333; margin: 0 0 4px; }
"""
BROWSERS = [r'C:\Program Files\Google\Chrome\Application\chrome.exe',
            r'C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe',
            shutil.which('chromium'), shutil.which('google-chrome'), shutil.which('chrome')]


def img(path, caption):
    data = base64.b64encode(Path(path).read_bytes()).decode()
    return f'<img src="data:image/png;base64,{data}"><div class="fig">{caption}</div>'


def render(md_path, pdf_path, figures=()):
    html_body = markdown.markdown(Path(md_path).read_text(encoding='utf-8'), extensions=['tables'])
    for marker, path, caption in figures:
        if Path(path).exists():
            html_body = html_body.replace(marker, img(path, caption) + marker, 1)
    html = f'<!doctype html><html><head><meta charset="utf-8"><style>{CSS}</style></head><body>{html_body}</body></html>'
    tmp = Path(pdf_path).with_suffix('.html')
    tmp.write_text(html, encoding='utf-8')
    exe = next(b for b in BROWSERS if b and Path(b).exists())
    subprocess.run([exe, '--headless=new', '--disable-gpu', '--no-pdf-header-footer',
                    f'--print-to-pdf={Path(pdf_path).resolve()}', tmp.resolve().as_uri()], check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    tmp.unlink()
    print('wrote', pdf_path)


def main():
    render(ROOT / 'REPORT.md', ROOT / 'REPORT.pdf', figures=[
        ('<h2>3. QAOA design</h2>', ROOT / 'figures' / 'fig_R.png',
         'Figure 1. Scenario R: (a) QUBO matrix (symmetrised); (b) QUBO energy vs official cost for all 256 schedules; '
         '(c) 4,096 raw QAOA shots by official-cost rank (log scale).'),
    ])
    render(ROOT / 'AI_DISCLOSURE.md', ROOT / 'AI_DISCLOSURE.pdf')


if __name__ == '__main__':
    sys.exit(main())
