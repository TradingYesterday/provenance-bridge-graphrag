#!/usr/bin/env python3
"""Print a diagnostic summary. Refuses to treat it as a QA score."""

import json
import sys
from pathlib import Path

def main() -> int:
    path = Path(sys.argv[1] if len(sys.argv) > 1 else "outputs/diagnostics/summary.json")
    if not path.exists():
        print(f"No summary at {path}. Run scripts/run_diagnostics.py first.", file=sys.stderr)
        return 2
    summary = json.loads(path.read_text(encoding="utf-8"))
    print(json.dumps({"kind": summary.get("kind"), "passed": summary.get("passed"), "failed": summary.get("failed")}, ensure_ascii=False))
    return 0 if summary.get("failed", 1) == 0 else 1

if __name__ == "__main__":
    raise SystemExit(main())
