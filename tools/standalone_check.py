#!/usr/bin/env python3
"""Prove the notebook is self-contained: run it in an empty folder that holds only the two data files, as on Colab.

Copies the training and hidden-test CSVs (flat, as an upload would place them) into a temporary folder, executes a
copy of rush_hour.ipynb there with a fresh kernel, and reports errors, the files the notebook created, and whether
the submission it wrote equals the repository's. Nothing in the repository is modified.
Usage: standalone_check.py [--notebook rush_hour.ipynb] [--config config.yaml]
"""
import argparse
import pathlib
import shutil
import sys
import tempfile

import nbclient
import nbformat
import yaml

ap = argparse.ArgumentParser()
ap.add_argument("--notebook", default="rush_hour.ipynb")
ap.add_argument("--config", default="config.yaml")
a = ap.parse_args()
paths = yaml.safe_load(pathlib.Path(a.config).read_text(encoding="utf-8"))["paths"]
node = nbformat.read(a.notebook, as_version=4)
for cell in node.cells:
    if cell.cell_type == "code":
        cell.outputs, cell.execution_count = [], None
with tempfile.TemporaryDirectory() as folder:
    for key in ("train", "test"):
        shutil.copy(paths[key], folder)                       # flat copy: <folder>/<file>.csv
    nbclient.NotebookClient(node, timeout=3600, kernel_name="python3",
                            resources={"metadata": {"path": folder}}).execute()
    created = sorted(p.name for p in pathlib.Path(folder).iterdir())
    written = pathlib.Path(folder) / "sample_submission.csv"
    same = written.exists() and written.read_bytes() == pathlib.Path("sample_submission.csv").read_bytes()
errors = sum(1 for c in node.cells if c.cell_type == "code" for o in c.outputs if o.output_type == "error")
print(f"standalone_check: errors={errors}; folder now holds {created}; submission identical to repository: {same}")
sys.exit(1 if errors or not same else 0)
