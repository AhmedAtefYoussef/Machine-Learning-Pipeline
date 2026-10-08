#!/usr/bin/env python3
"""Presentation metadata for the executed notebook; the cells' sources and outputs are not touched.

The setup cell (the code cell that starts with `#@title Setup`) is marked as a collapsed form cell so that Colab and
Jupyter show its title but not the long block of base64 text, and the notebook gets the Colab table of contents.
Usage: polish_nb.py rush_hour.ipynb
"""
import json
import sys


def polish(path: str) -> int:
    with open(path, encoding="utf-8") as handle:
        notebook = json.load(handle)
    setup = [c for c in notebook["cells"] if c["cell_type"] == "code" and "".join(c["source"]).startswith("#@title Setup")]
    if not setup:
        print(f"polish_nb: no setup cell (#@title Setup) in {path}")
        return 1
    setup[0]["metadata"].update({"cellView": "form", "jupyter": {"source_hidden": True}})
    notebook["metadata"]["colab"] = {"provenance": [], "toc_visible": True}
    notebook["metadata"].setdefault("language_info", {})["name"] = "python"
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        json.dump(notebook, handle, indent=1, ensure_ascii=False)
        handle.write(chr(10))
    print(f"polish_nb: setup cell collapsed, Colab metadata added -> {path}")
    return 0


if __name__ == "__main__":
    sys.exit(polish(sys.argv[1]))
