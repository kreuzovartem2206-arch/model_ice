"""Сборка автономной HTML-карты из pilot_results.json.

    python -m pevek_router.viewer out/pilot_results.json out/pevek_router.html
"""
import sys
from pathlib import Path


def build(json_path, html_path):
    tpl = (Path(__file__).parent / "viewer_template.html").read_text(encoding="utf-8")
    data = Path(json_path).read_text(encoding="utf-8").replace("</", "<\\/")
    Path(html_path).write_text(tpl.replace("/*__DATA__*/", data), encoding="utf-8")
    return html_path


if __name__ == "__main__":
    build(sys.argv[1], sys.argv[2])
