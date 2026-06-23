#!/bin/bash
set -e

rm -f evaluation_script.zip challenge_config.zip

python - <<'PY'
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile


def skip_common(path):
    parts = set(path.parts)
    return path.name == ".DS_Store" or "__pycache__" in parts or ".git" in parts


def add_file(zf, path, arcname=None):
    path = Path(path)
    if path.is_file() and not skip_common(path):
        zf.write(path, arcname or path)


def add_tree(zf, root):
    root = Path(root)
    for path in sorted(root.rglob("*")):
        add_file(zf, path)


with ZipFile("evaluation_script.zip", "w", ZIP_DEFLATED) as zf:
    for path in ["evaluation_script/__init__.py", "evaluation_script/main.py"]:
        add_file(zf, path, Path(path).relative_to("evaluation_script"))

with ZipFile("challenge_config.zip", "w", ZIP_DEFLATED) as zf:
    for path in [
        "challenge_config.yaml",
        "evaluation_script.zip",
        "logo.jpg",
        "submission.json",
        "f034078f-3fc8-4a9e-a6c7-8dfeb04b9102.json",
        "936aa32e-760f-4ada-8672-ee4bfeeb5d05.json",
    ]:
        add_file(zf, path)
    for root in ["annotations", "templates"]:
        add_tree(zf, root)
PY
