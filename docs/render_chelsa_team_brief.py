"""Render chelsa-team-brief.md as a print-ready PDF using headless Chrome."""

from pathlib import Path
import shutil
import subprocess
import sys

from markdown_it import MarkdownIt


HERE = Path(__file__).resolve().parent
SOURCE = HERE / "chelsa-team-brief.md"
HTML = HERE / "chelsa-team-brief.html"
PDF = HERE / "chelsa-team-brief.pdf"


def find_chrome():
    candidates = (
        "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
        "/Applications/Chromium.app/Contents/MacOS/Chromium",
    )
    for candidate in candidates:
        if Path(candidate).exists():
            return candidate
    for name in ("chromium", "google-chrome-stable", "google-chrome"):
        if found := shutil.which(name):
            return found
    return None


CSS = r"""
@page { size: A4; margin: 16mm 17mm 17mm; }
:root { --navy:#17385f; --blue:#3174a8; --pale:#eef5f9; --ink:#17202a;
  --muted:#596775; --rule:#d7e0e7; --good:#26734d; --warn:#a25b18; }
* { box-sizing:border-box; }
html { -webkit-print-color-adjust:exact; print-color-adjust:exact; }
body { margin:0; color:var(--ink); font:10.25pt/1.52 Georgia,"Iowan Old Style",serif;
  orphans:3; widows:3; }
.masthead { display:flex; justify-content:space-between; align-items:baseline;
  border-bottom:2px solid var(--navy); padding-bottom:6px; margin-bottom:22px;
  font:700 9pt/1.2 Arial,sans-serif; color:var(--navy); letter-spacing:.07em;
  text-transform:uppercase; }
.masthead span:last-child { color:#63809d; font-size:8pt; }
h1,h2,h3 { font-family:Arial,"Helvetica Neue",sans-serif; color:var(--navy);
  break-after:avoid; page-break-after:avoid; }
h1 { font-size:24pt; line-height:1.12; letter-spacing:-.02em; margin:0 0 5px; }
h1 + h3 { color:var(--muted); font-size:12pt; font-weight:400; line-height:1.35;
  margin:0 0 10px; }
h1 + h3 + p { margin:0 0 20px; color:#688098; font:8.5pt Arial,sans-serif;
  text-transform:uppercase; letter-spacing:.06em; }
h2 { margin:24px 0 8px; padding-bottom:4px; border-bottom:1.5px solid var(--blue);
  font-size:15pt; }
h3 { margin:18px 0 5px; font-size:11.5pt; }
p { margin:7px 0; }
a { color:#1d6298; text-decoration:none; border-bottom:.4px solid #8fb1ca; }
blockquote { margin:14px 0 18px; padding:11px 14px; background:var(--pale);
  border-left:4px solid var(--blue); border-radius:0 5px 5px 0; break-inside:avoid; }
blockquote p { margin:0; font-size:10.6pt; }
ul,ol { margin:7px 0 8px 20px; padding:0; }
li { margin:4px 0; }
table { width:100%; border-collapse:collapse; margin:10px 0 14px;
  font:8.7pt/1.38 Arial,sans-serif; break-inside:auto; }
thead { display:table-header-group; }
tr { break-inside:avoid; }
th { color:white; background:var(--navy); text-align:left; padding:6px 7px;
  font-size:8pt; text-transform:uppercase; letter-spacing:.03em; }
td { padding:6px 7px; border-bottom:1px solid var(--rule); vertical-align:top; }
tbody tr:nth-child(even) { background:#f6f9fb; }
th[align=right],td[align=right] { text-align:right; font-variant-numeric:tabular-nums; }
code { font:8.7pt "SFMono-Regular",Consolas,monospace; color:#263e57;
  background:#f0f3f6; padding:1px 3px; border-radius:3px; }
.flow { display:grid; grid-template-columns:1.05fr auto 1.45fr auto 1.15fr auto 1.05fr;
  align-items:stretch; gap:7px; margin:14px 0 17px; break-inside:avoid;
  font-family:Arial,sans-serif; }
.flow div { background:var(--pale); border:1px solid #bfd1df; border-radius:5px;
  padding:10px 7px; text-align:center; color:var(--navy); font-size:8.6pt; }
.flow span { align-self:center; color:var(--blue); font:bold 14pt Arial,sans-serif; }
.flow small { display:block; margin-top:4px; color:var(--muted); font-size:7.4pt; }
img { display:block; max-width:100%; max-height:150mm; margin:11px auto 7px;
  break-inside:avoid; }
hr { border:0; border-top:1px solid var(--rule); margin:22px 0 12px; }
hr + h3 { margin-top:0; }
body > .wrap > p:last-child { color:var(--muted); font-size:8.7pt; }
"""


def main():
    if not SOURCE.exists():
        sys.exit(f"Source not found: {SOURCE}")
    md = MarkdownIt("commonmark", {"html": True}).enable("table")
    body = md.render(SOURCE.read_text())
    masthead = ("<div class='masthead'><span>DeepScale technical brief</span>"
                "<span>CHELSA precipitation downscaling</span></div>")
    html = ("<!doctype html><html><head><meta charset='utf-8'>"
            "<title>CHELSA precipitation downscaling in DeepScale</title>"
            f"<style>{CSS}</style></head><body><div class='wrap'>{masthead}{body}"
            "</div></body></html>")
    HTML.write_text(html)
    print(f"wrote {HTML}")
    chrome = find_chrome()
    if chrome is None:
        sys.exit("Chrome or Chromium is required to render the PDF")
    subprocess.run(
        [chrome, "--headless", "--disable-gpu", "--no-pdf-header-footer",
         f"--print-to-pdf={PDF}", HTML.as_uri()],
        check=True, capture_output=True, timeout=180,
    )
    print(f"wrote {PDF} ({PDF.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
