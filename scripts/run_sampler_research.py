from __future__ import annotations

import json
import subprocess
import sys
import time
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
TMP_CONFIG_DIR = REPO_ROOT / "configs" / "tmp"
TSV_PATH = REPO_ROOT / "exp" / "apr12-lmx-batching.tsv"
BASELINE_CONFIG_PATH = REPO_ROOT / "configs" / "pretrain_rd_best.yaml"
RUNNER = ["uv", "run", "python", "run_pretrain.py"]
RUNS_ROOT = REPO_ROOT / "artifacts" / "runs"


@dataclass
class ExperimentResult:
    description: str
    valid_loss: float
    tokens_per_second: float
    status: str
    run_dir: str | None


def main() -> None:
    commit = git_short_commit()
    seen_descriptions = load_seen_descriptions(TSV_PATH)
    print(f"[sampler-research] starting on commit {commit}", flush=True)

    targeted_experiments = build_targeted_sampler_experiments()
    for experiment_spec in targeted_experiments:
        description = experiment_spec["description"]
        if description in seen_descriptions:
            print(f"[sampler-research] skip existing: {description}", flush=True)
            continue

        config_path = write_temp_config(experiment_spec)
        print(f"[sampler-research] start: {description}", flush=True)
        started_at = time.monotonic()
        result = run_experiment(config_path, description, experiment_spec["budget_seconds"])
        elapsed = time.monotonic() - started_at
        append_tsv(
            TSV_PATH,
            commit=commit,
            valid_loss=result.valid_loss,
            tokens_per_second=result.tokens_per_second,
            status=result.status,
            description=f"{description} run={result.run_dir or 'n/a'} wall={elapsed:.1f}s",
        )
        print(
            "[sampler-research] done: "
            f"{description} status={result.status} val={result.valid_loss:.6f} "
            f"toks/sec={result.tokens_per_second:.1f}",
            flush=True,
        )
        seen_descriptions.add(description)

    targeted_results = load_targeted_sampler_results(TSV_PATH, targeted_experiments)
    fastest = pick_fastest_targeted_sampler(targeted_results)
    if fastest is None:
        return

    rerun_description = (
        f"{fastest.description.replace('[600s sampler-speed]', '').strip()} [7200s targeted canonical]"
    )
    if rerun_description in seen_descriptions:
        print(f"[sampler-research] skip existing: {rerun_description}", flush=True)
        return

    rerun_spec = experiment(
        name=f"{sanitize_name(rerun_description)}_7200s",
        description=rerun_description,
        budget_seconds=7200,
        batch_size=64,
        data_updates=settings_from_description(fastest.description),
    )
    config_path = write_temp_config(rerun_spec)
    print(f"[sampler-research] fastest targeted 600s config: {fastest.description}", flush=True)
    print(f"[sampler-research] start: {rerun_description}", flush=True)
    started_at = time.monotonic()
    result = run_experiment(config_path, rerun_description, rerun_spec["budget_seconds"])
    elapsed = time.monotonic() - started_at
    append_tsv(
        TSV_PATH,
        commit=commit,
        valid_loss=result.valid_loss,
        tokens_per_second=result.tokens_per_second,
        status=result.status,
        description=f"{rerun_description} run={result.run_dir or 'n/a'} wall={elapsed:.1f}s",
    )
    print(
        "[sampler-research] done: "
        f"{rerun_description} status={result.status} val={result.valid_loss:.6f} "
        f"toks/sec={result.tokens_per_second:.1f}",
        flush=True,
    )


def build_targeted_sampler_experiments() -> list[dict[str, Any]]:
    experiments: list[dict[str, Any]] = []
    for max_tokens in [14336, 16384, 18432]:
        for pad_multiple in [32, 64, 128]:
            experiments.append(
                experiment(
                    name=f"target_lb_mt{max_tokens}_pad{pad_multiple}_600s",
                    description=f"length_bucketing + max_tokens {max_tokens} + pad{pad_multiple} [600s sampler-speed]",
                    budget_seconds=600,
                    batch_size=64,
                    data_updates={
                        "bucket_padding_noise": 0.0,
                        "max_tokens_per_batch": max_tokens,
                        "pad_to_length_multiple": pad_multiple,
                    },
                )
            )
    return experiments


def experiment(
    *,
    name: str,
    description: str,
    budget_seconds: int,
    batch_size: int,
    data_updates: dict[str, Any],
) -> dict[str, Any]:
    return {
        "name": name,
        "description": description,
        "budget_seconds": budget_seconds,
        "batch_size": batch_size,
        "data_updates": data_updates,
    }


def write_temp_config(experiment_spec: dict[str, Any]) -> Path:
    TMP_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    config = load_yaml(BASELINE_CONFIG_PATH)
    config["training"]["batch_size"] = experiment_spec["batch_size"]
    config["training"]["eval_batch_size"] = 16
    config["training"]["num_steps"] = 1_000_000
    config["training"]["max_duration_seconds"] = experiment_spec["budget_seconds"]
    config["training"]["log_every"] = 50
    config["training"]["eval_every"] = 500
    config["training"]["num_eval_batches"] = None
    for key, value in experiment_spec["data_updates"].items():
        config["data"][key] = value

    config_path = TMP_CONFIG_DIR / f"{experiment_spec['name']}.yaml"
    with config_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False)
    return config_path


def run_experiment(config_path: Path, description: str, budget_seconds: int) -> ExperimentResult:
    try:
        completed = subprocess.run(
            [
                *RUNNER,
                "--config",
                str(config_path),
                "--runs-root",
                str(RUNS_ROOT),
                "--note",
                description,
            ],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=budget_seconds + 900,
            check=True,
        )
    except subprocess.TimeoutExpired:
        return ExperimentResult(description=description, valid_loss=0.0, tokens_per_second=0.0, status="crash", run_dir=None)
    except subprocess.CalledProcessError as error:
        return ExperimentResult(
            description=description,
            valid_loss=0.0,
            tokens_per_second=0.0,
            status="crash",
            run_dir=parse_run_dir(error.stdout),
        )

    run_dir = parse_run_dir(completed.stdout)
    if run_dir is None:
        raise RuntimeError(f"Failed to parse run_dir from stdout for {description!r}.")
    summary = json.loads((Path(run_dir) / "summary.json").read_text(encoding="utf-8"))
    result = summary["result"]
    return ExperimentResult(
        description=description,
        valid_loss=float(result["best_validation_loss"] or 0.0),
        tokens_per_second=float(result["average_tokens_per_second"]),
        status="keep",
        run_dir=run_dir,
    )


def parse_run_dir(stdout: str) -> str | None:
    for line in stdout.splitlines():
        if line.startswith("run_dir="):
            return line.removeprefix("run_dir=").strip()
    return None


def load_yaml(path: Path) -> dict[str, Any]:
    return deepcopy(yaml.safe_load(path.read_text(encoding="utf-8")))


def git_short_commit() -> str:
    return subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def append_tsv(
    path: Path,
    *,
    commit: str,
    valid_loss: float,
    tokens_per_second: float,
    status: str,
    description: str,
) -> None:
    if not path.exists():
        path.write_text("commit\tvalid_loss\ttoks/sec\tstatus\tdescription\n", encoding="utf-8")
    with path.open("a", encoding="utf-8") as handle:
        handle.write(
            f"{commit}\t{valid_loss:.6f}\t{tokens_per_second:.1f}\t{status}\t{description}\n"
        )


def load_seen_descriptions(path: Path) -> set[str]:
    if not path.exists():
        return set()
    descriptions = set()
    for line in path.read_text(encoding="utf-8").splitlines()[1:]:
        if not line.strip():
            continue
        parts = line.split("\t", maxsplit=4)
        if len(parts) != 5:
            continue
        descriptions.add(parts[4].split(" run=")[0])
    return descriptions


def load_targeted_sampler_results(path: Path, targeted_experiments: list[dict[str, Any]]) -> list[ExperimentResult]:
    targeted_descriptions = {experiment["description"] for experiment in targeted_experiments}
    results: list[ExperimentResult] = []
    if not path.exists():
        return results
    for line in path.read_text(encoding="utf-8").splitlines()[1:]:
        if not line.strip():
            continue
        commit, valid_loss, toks_per_sec, status, description = line.split("\t", maxsplit=4)
        clean_description = description.split(" run=")[0]
        if clean_description not in targeted_descriptions:
            continue
        results.append(
            ExperimentResult(
                description=clean_description,
                valid_loss=float(valid_loss),
                tokens_per_second=float(toks_per_sec),
                status=status,
                run_dir=None,
            )
        )
    return results


def pick_fastest_targeted_sampler(results: list[ExperimentResult]) -> ExperimentResult | None:
    successful = [result for result in results if result.status != "crash"]
    if not successful:
        return None
    return max(successful, key=lambda result: result.tokens_per_second)


def settings_from_description(description: str) -> dict[str, Any]:
    max_tokens = int(description.split("max_tokens ")[1].split(" ")[0])
    pad_multiple = int(description.split("pad")[1].split(" ")[0])
    return {
        "bucket_padding_noise": 0.0,
        "max_tokens_per_batch": max_tokens,
        "pad_to_length_multiple": pad_multiple,
    }


def sanitize_name(description: str) -> str:
    return "".join(character.lower() if character.isalnum() else "_" for character in description).strip("_")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("[sampler-research] interrupted", file=sys.stderr)
        raise
