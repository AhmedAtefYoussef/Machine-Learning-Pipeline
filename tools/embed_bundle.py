#!/usr/bin/env python3
"""Put the project files the notebook needs INSIDE the notebook, so it runs where only the .ipynb and the two data
files are available (Google Colab).

The setup cell of nb/p0_setup.py contains three markers. This tool replaces them in the rendered copy:
  @@PROJECT_BUNDLE@@  base64 text of a zip with src/, config.yaml, the stored artifacts and the submission template
  @@SHA_TRAIN@@       sha256 of the training file (line endings normalised) the stored artifacts were computed from
  @@SHA_TEST@@        the same for the hidden-test file
The data files themselves are NOT bundled: the user supplies them.
Usage: embed_bundle.py build/nb/p0_setup.py [--config config.yaml]
"""
import argparse
import base64
import hashlib
import io
import pathlib
import sys
import zipfile

import yaml

BUNDLE_GLOBS = ["config.yaml", "src/*.py", "src/phases/*.py", "artifacts/p?.json", "artifacts/bonus_control.json"]
LINE = 120   # characters per line of base64 text


def normalised_sha256(path: pathlib.Path) -> str:
    """sha256 of a text file with Windows line endings turned into Unix ones (the same data on any system)."""
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def bundle_text(root: pathlib.Path, extra: list[pathlib.Path]) -> tuple[str, int]:
    """Zip the project files (fixed timestamps: same files, same bundle) and return wrapped base64 text."""
    files = sorted({p for pattern in BUNDLE_GLOBS for p in root.glob(pattern)} | set(extra))
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for path in files:
            info = zipfile.ZipInfo(path.relative_to(root).as_posix(), date_time=(2026, 10, 7, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(info, path.read_bytes())
    text = base64.b64encode(buffer.getvalue()).decode("ascii")
    return "\n".join(text[i:i + LINE] for i in range(0, len(text), LINE)), len(files)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("target")
    ap.add_argument("--config", default="config.yaml")
    a = ap.parse_args()
    root = pathlib.Path(a.config).resolve().parent
    paths = yaml.safe_load(pathlib.Path(a.config).read_text(encoding="utf-8"))["paths"]
    template = root / paths["sample_submission"]
    blob, n_files = bundle_text(root, [template] if template.exists() else [])
    target = pathlib.Path(a.target)
    text = target.read_text(encoding="utf-8")
    for marker, value in (("@@PROJECT_BUNDLE@@", blob), ("@@SHA_TRAIN@@", normalised_sha256(root / paths["train"])),
                          ("@@SHA_TEST@@", normalised_sha256(root / paths["test"]))):
        if text.count(marker) != 1:
            print(f"embed_bundle: expected exactly one {marker} in {target}, found {text.count(marker)}")
            return 1
        text = text.replace(marker, value)
    target.write_text(text, encoding="utf-8", newline="\n")
    print(f"embed_bundle: {n_files} files, {len(blob) // 1024} KiB of text -> {target}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
