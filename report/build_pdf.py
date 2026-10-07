"""Build report/report.pdf from report/report.md (already rendered by tools/number_trace.py).

Markdown -> HTML (markdown-it) -> PDF (headless Chrome or Edge). Prints the page count and exits 1 above 6 pages.
Usage: python -X utf8 report/build_pdf.py
"""
import pathlib
import re
import subprocess
import sys

from markdown_it import MarkdownIt

HERE = pathlib.Path(__file__).resolve().parent
BROWSERS = [
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
]
MAX_PAGES = 6
CSS = """
@page { size: A4; margin: 13mm 14mm 13mm 14mm; }
body { font-family: "Segoe UI", Arial, sans-serif; font-size: 8.6pt; line-height: 1.27; color: #111; }
h1 { font-size: 15pt; margin: 0 0 2pt 0; }
h2 { font-size: 10.8pt; margin: 8pt 0 2pt 0; border-bottom: 0.6pt solid #888; padding-bottom: 1pt; }
h3 { font-size: 9.2pt; margin: 5pt 0 1pt 0; }
p { margin: 2.5pt 0; text-align: justify; }
ul, ol { margin: 2pt 0 2pt 14pt; padding: 0; }
li { margin: 0.5pt 0; }
table { border-collapse: collapse; margin: 3pt 0; font-size: 7.6pt; width: 100%; }
th, td { border: 0.5pt solid #999; padding: 1pt 3pt; text-align: left; vertical-align: top; }
th { background: #eee; }
img { max-width: 100%; }
.figrow { display: flex; gap: 6pt; align-items: flex-start; }
.figrow > div { flex: 1; }
.cap { font-size: 7.4pt; color: #333; margin-top: 0; }
code { font-family: Consolas, monospace; font-size: 7.8pt; }
.eq { text-align: center; font-family: "Cambria Math", Cambria, serif; font-size: 9.2pt; margin: 2pt 0; }
"""


def count_pages(pdf: pathlib.Path) -> int:
    """Count page objects in a PDF file without a PDF library."""
    return len(re.findall(rb"/Type\s*/Page(?![s])", pdf.read_bytes()))


def main() -> int:
    md = (HERE / "report.md").read_text(encoding="utf-8")
    body = MarkdownIt("commonmark", {"html": True}).enable("table").render(md)
    html = f"<!doctype html><html><head><meta charset='utf-8'><style>{CSS}</style></head><body>{body}</body></html>"
    html_path = HERE / "report.html"
    html_path.write_text(html, encoding="utf-8", newline="\n")
    pdf_path = HERE / "report.pdf"
    browser = next((b for b in BROWSERS if pathlib.Path(b).exists()), None)
    if browser is None:
        print("build_pdf: no Chrome/Edge found")
        return 1
    subprocess.run([browser, "--headless=new", "--disable-gpu", "--no-pdf-header-footer",
                    f"--print-to-pdf={pdf_path}", html_path.as_uri()], check=True, capture_output=True, timeout=180)
    pages = count_pages(pdf_path)
    print(f"build_pdf: {pdf_path.name} pages={pages} (limit {MAX_PAGES})")
    return 0 if 0 < pages <= MAX_PAGES else 1


if __name__ == "__main__":
    sys.exit(main())
