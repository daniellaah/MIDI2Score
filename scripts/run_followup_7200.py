from __future__ import annotations

import json
import subprocess
import time
from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
BASELINE_CONFIG_PATH = REPO_ROOT / "configs" / "pretrain_rd_best.yaml"
TMP_CONFIG_PATH = REPO_ROOT / "configs" / "tmp" / "length_bucketing_mt16384_pad128_7200s.yaml"
TSV_PATH = REPO_ROOT / "exp" / "apr12-lmx-batching.tsv"
RUNS_ROOT = REPO_ROOT / "artifacts" / "runs"
NOTE = "length_bucketing + max_tokens 16384 + pad128 [7200s targeted canonical]"


def main() -> None:
    wait_for_current_long_run()
    if note_already_recorded():
        print(f"[followup-7200] skip existing: {NOTE}", flush=True)
        return

    write_config()
    commit = git_short_commit()
    print(f"[followup-7200] start: {NOTE}", flush=True)
    started_at = time.monotonic()
    completed = subprocess.run(
        [
            "uv",
            "run",
            "python",
            "run_pretrain.py",
            "--config",
            str(TMP_CONFIG_PATH),
            "--runs-root",
            str(RUNS_ROOT),
            "--note",
            NOTE,
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=7200 + 1800,
        check=True,
    )
    elapsed = time.monotonic() - started_at
    run_dir = parse_run_dir(completed.stdout)
    summary = json.loads((Path(run_dir) / "summary.json").read_text(encoding="utf-8"))
    result = summary["result"]
    append_tsv(
        commit=commit,
        valid_loss=float(result["best_validation_loss"] or 0.0),
        tokens_per_second=float(result["average_tokens_per_second"]),
        description=f"{NOTE} run={relative_run_dir(run_dir)} wall={elapsed:.1f}s",
    )
    print(
        f"[followup-7200] done: val={float(result['best_validation_loss'] or 0.0):.6f} "
        f"toks/sec={float(result['average_tokens_per_second']):.1f}",
        flush=True,
    )


def wait_for_current_long_run() -> None:
    target = "length_bucketing + max_tokens 14336 + pad32 [7200s targeted canonical]"
    while True:
        running = subprocess.run(
            ["ps", "-ax"],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        if target not in running:
            return
        time.sleep(15)


def note_already_recorded() -> bool:
    if not TSV_PATH.exists():
        return False
    return NOTE in TSV_PATH.read_text(encoding="utf-8")


def write_config() -> None:
    config = deepcopy(yaml.safe_load(BASELINE_CONFIG_PATH.read_text(encoding="utf-8")))
    config["training"]["batch_size"] = 64
    config["training"]["eval_batch_size"] = 16
    config["training"]["num_steps"] = 1_000_000
    config["training"]["max_duration_seconds"] = 7200
    config["training"]["log_every"] = 50
    config["training"]["eval_every"] = 500
    config["training"]["num_eval_batches"] = None
    config["data"]["bucket_padding_noise"] = 0.0
    config["data"]["max_tokens_per_batch"] = 16384
    config["data"]["pad_to_length_multiple"] = 128
    TMP_CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    TMP_CONFIG_PATH.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")


def git_short_commit() -> str:
    return subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def parse_run_dir(stdout: str) -> str:
    for line in stdout.splitlines():
        if line.startswith("run_dir="):
            return line.removeprefix("run_dir=").strip()
    raise RuntimeError("run_dir not found in run_pretrain output")


def relative_run_dir(run_dir: str) -> str:
    return str(Path(run_dir).resolve().relative_to(REPO_ROOT.resolve()))


def append_tsv(*, commit: str, valid_loss: float, tokens_per_second: float, description: str) -> None:
    with TSV_PATH.open("a", encoding="utf-8") as handle:
        handle.write(f"{commit}\t{valid_loss:.6f}\t{tokens_per_second:.1f}\tkeep\t{description}\n")


if __name__ == "__main__":
    main()
