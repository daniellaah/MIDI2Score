from __future__ import annotations

import subprocess
import time
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
TSV_PATH = REPO_ROOT / "exp" / "apr12-lmx-batching.tsv"
SYNC_SCRIPT = REPO_ROOT / "scripts" / "update_length_bucketing_doc.py"


def main() -> None:
    last_mtime_ns = -1
    while True:
        if TSV_PATH.exists():
            current_mtime_ns = TSV_PATH.stat().st_mtime_ns
            if current_mtime_ns != last_mtime_ns:
                subprocess.run(
                    ["uv", "run", "python", str(SYNC_SCRIPT)],
                    cwd=REPO_ROOT,
                    check=True,
                )
                last_mtime_ns = current_mtime_ns
        time.sleep(10)


if __name__ == "__main__":
    main()
