"""Dependency-free checks suitable for GitHub-hosted runners without an NPU."""

import ast
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
REQUIRED_FILES = (
    "LICENSE",
    "NOTICE",
    "CITATION.cff",
    "ascend_rsfa/flash_attention_forward.py",
    "baseline/06-fused-attention.py",
    "benchmarks/run_correctness_test.py",
)


def main() -> None:
    for relative_path in REQUIRED_FILES:
        path = ROOT / relative_path
        if not path.is_file():
            raise SystemExit(f"missing required file: {relative_path}")

    python_files = sorted(ROOT.rglob("*.py"))
    for path in python_files:
        if any(part.startswith(".") for part in path.relative_to(ROOT).parts):
            continue
        ast.parse(path.read_text(encoding="utf-8"), filename=str(path))

    for path in sorted(ROOT.rglob("*.json")):
        if any(part.startswith(".") for part in path.relative_to(ROOT).parts):
            continue
        json.loads(path.read_text(encoding="utf-8"))

    print(f"validated {len(python_files)} Python files and repository JSON data")


if __name__ == "__main__":
    main()
