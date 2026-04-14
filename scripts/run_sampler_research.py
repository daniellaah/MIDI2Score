from __future__ import annotations

import json
import re
import subprocess
import sys
import time
from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
TMP_CONFIG_DIR = REPO_ROOT / "configs" / "tmp"
TSV_PATH = REPO_ROOT / "exp" / "apr12-lmx-batching.tsv"
BASELINE_CONFIG_PATH = REPO_ROOT / "configs" / "pretrain_rd_best.yaml"
RUNNER = ["uv", "run", "python", "run_pretrain.py"]
RUNS_ROOT = REPO_ROOT / "artifacts" / "runs"


def main() -> None:
    commit = git_short_commit()
    seen_descriptions = load_seen_descriptions(TSV_PATH)
    print(f"[sampler-research] starting on commit {commit}", flush=True)

    stage_600 = build_stage_600_experiments()
    completed_600: list[ExperimentResult] = []
    for experiment in stage_600:
        result = maybe_run_experiment(experiment, commit=commit, seen_descriptions=seen_descriptions)
        if result is not None:
            seen_descriptions.add(result.description)
            completed_600.append(result)

    recorded_600 = load_results(TSV_PATH)
    fastest_sampler = pick_fastest_sampler(recorded_600)
    if fastest_sampler is None:
        raise RuntimeError("Failed to determine the fastest sampler configuration from 600s runs.")

    print(
        "[sampler-research] fastest 600s sampler config: "
        f"{fastest_sampler.description} toks/sec={fastest_sampler.tokens_per_second:.1f}",
        flush=True,
    )

    stage_7200 = build_stage_7200_experiments(fastest_sampler)
    for experiment in stage_7200:
        result = maybe_run_experiment(experiment, commit=commit, seen_descriptions=seen_descriptions)
        if result is not None:
            seen_descriptions.add(result.description)

    for experiment in build_endless_sampler_experiments():
        result = maybe_run_experiment(experiment, commit=commit, seen_descriptions=seen_descriptions)
        if result is not None:
            seen_descriptions.add(result.description)


class ExperimentResult:
    def __init__(
        self,
        *,
        description: str,
        valid_loss: float,
        tokens_per_second: float,
        status: str,
        run_dir: str | None,
    ) -> None:
        self.description = description
        self.valid_loss = valid_loss
        self.tokens_per_second = tokens_per_second
        self.status = status
        self.run_dir = run_dir


def build_stage_600_experiments() -> list[dict[str, Any]]:
    return [
        experiment(
            name="baseline_fixed_600s",
            description="fixed batch baseline [600s sampler-speed]",
            budget_seconds=600,
            batch_size=16,
            data_updates={
                "length_bucketing": False,
                "bucket_padding_noise": 0.0,
                "max_tokens_per_batch": None,
                "required_batch_size_multiple": 1,
                "pad_to_length_multiple": 1,
            },
        ),
        experiment(
            name="lb_only_600s",
            description="length_bucketing only [600s sampler-speed]",
            budget_seconds=600,
            batch_size=16,
            data_updates={
                "length_bucketing": True,
                "bucket_padding_noise": 0.0,
                "max_tokens_per_batch": None,
                "required_batch_size_multiple": 1,
                "pad_to_length_multiple": 1,
            },
        ),
        experiment(
            name="mt12288_only_600s",
            description="max_tokens_per_batch 12288 only [600s sampler-speed]",
            budget_seconds=600,
            batch_size=64,
            data_updates={
                "length_bucketing": False,
                "bucket_padding_noise": 0.0,
                "max_tokens_per_batch": 12288,
                "required_batch_size_multiple": 1,
                "pad_to_length_multiple": 1,
            },
        ),
        experiment(
            name="mt16384_only_600s",
            description="max_tokens_per_batch 16384 only [600s sampler-speed]",
            budget_seconds=600,
            batch_size=64,
            data_updates={
                "length_bucketing": False,
                "bucket_padding_noise": 0.0,
                "max_tokens_per_batch": 16384,
                "required_batch_size_multiple": 1,
                "pad_to_length_multiple": 1,
            },
        ),
        experiment(
            name="mt20480_only_600s",
            description="max_tokens_per_batch 20480 only [600s sampler-speed]",
            budget_seconds=600,
            batch_size=64,
            data_updates={
                "length_bucketing": False,
                "bucket_padding_noise": 0.0,
                "max_tokens_per_batch": 20480,
                "required_batch_size_multiple": 1,
                "pad_to_length_multiple": 1,
            },
        ),
        experiment(
            name="lb_mt12288_600s",
            description="length_bucketing + max_tokens_per_batch 12288 [600s sampler-speed]",
            budget_seconds=600,
            batch_size=64,
            data_updates={
                "length_bucketing": True,
                "bucket_padding_noise": 0.0,
                "max_tokens_per_batch": 12288,
                "required_batch_size_multiple": 1,
                "pad_to_length_multiple": 1,
            },
        ),
        experiment(
            name="lb_mt16384_600s",
            description="length_bucketing + max_tokens_per_batch 16384 [600s sampler-speed]",
            budget_seconds=600,
            batch_size=64,
            data_updates={
                "length_bucketing": True,
                "bucket_padding_noise": 0.0,
                "max_tokens_per_batch": 16384,
                "required_batch_size_multiple": 1,
                "pad_to_length_multiple": 1,
            },
        ),
        experiment(
            name="lb_mt20480_600s",
            description="length_bucketing + max_tokens_per_batch 20480 [600s sampler-speed]",
            budget_seconds=600,
            batch_size=64,
            data_updates={
                "length_bucketing": True,
                "bucket_padding_noise": 0.0,
                "max_tokens_per_batch": 20480,
                "required_batch_size_multiple": 1,
                "pad_to_length_multiple": 1,
            },
        ),
        experiment(
            name="lb_mt16384_pad32_600s",
            description="length_bucketing + max_tokens 16384 + pad32 [600s sampler-speed]",
            budget_seconds=600,
            batch_size=64,
            data_updates={
                "length_bucketing": True,
                "bucket_padding_noise": 0.0,
                "max_tokens_per_batch": 16384,
                "required_batch_size_multiple": 1,
                "pad_to_length_multiple": 32,
            },
        ),
        experiment(
            name="lb_mt16384_pad64_600s",
            description="length_bucketing + max_tokens 16384 + pad64 [600s sampler-speed]",
            budget_seconds=600,
            batch_size=64,
            data_updates={
                "length_bucketing": True,
                "bucket_padding_noise": 0.0,
                "max_tokens_per_batch": 16384,
                "required_batch_size_multiple": 1,
                "pad_to_length_multiple": 64,
            },
        ),
        experiment(
            name="lb_mt16384_noise005_600s",
            description="length_bucketing + max_tokens 16384 + noise0.05 [600s sampler-speed]",
            budget_seconds=600,
            batch_size=64,
            data_updates={
                "length_bucketing": True,
                "bucket_padding_noise": 0.05,
                "max_tokens_per_batch": 16384,
                "required_batch_size_multiple": 1,
                "pad_to_length_multiple": 1,
            },
        ),
        experiment(
            name="lb_mt16384_noise01_600s",
            description="length_bucketing + max_tokens 16384 + noise0.10 [600s sampler-speed]",
            budget_seconds=600,
            batch_size=64,
            data_updates={
                "length_bucketing": True,
                "bucket_padding_noise": 0.10,
                "max_tokens_per_batch": 16384,
                "required_batch_size_multiple": 1,
                "pad_to_length_multiple": 1,
            },
        ),
        experiment(
            name="lb_mt16384_noise02_600s",
            description="length_bucketing + max_tokens 16384 + noise0.20 [600s sampler-speed]",
            budget_seconds=600,
            batch_size=64,
            data_updates={
                "length_bucketing": True,
                "bucket_padding_noise": 0.20,
                "max_tokens_per_batch": 16384,
                "required_batch_size_multiple": 1,
                "pad_to_length_multiple": 1,
            },
        ),
        experiment(
            name="lb_mt16384_req4_600s",
            description="length_bucketing + max_tokens 16384 + required4 [600s sampler-speed]",
            budget_seconds=600,
            batch_size=64,
            data_updates={
                "length_bucketing": True,
                "bucket_padding_noise": 0.0,
                "max_tokens_per_batch": 16384,
                "required_batch_size_multiple": 4,
                "pad_to_length_multiple": 1,
            },
        ),
        experiment(
            name="lb_mt16384_req8_600s",
            description="length_bucketing + max_tokens 16384 + required8 [600s sampler-speed]",
            budget_seconds=600,
            batch_size=64,
            data_updates={
                "length_bucketing": True,
                "bucket_padding_noise": 0.0,
                "max_tokens_per_batch": 16384,
                "required_batch_size_multiple": 8,
                "pad_to_length_multiple": 1,
            },
        ),
        experiment(
            name="lb_mt16384_pad64_noise01_600s",
            description="length_bucketing + max_tokens 16384 + pad64 + noise0.10 [600s sampler-speed]",
            budget_seconds=600,
            batch_size=64,
            data_updates={
                "length_bucketing": True,
                "bucket_padding_noise": 0.10,
                "max_tokens_per_batch": 16384,
                "required_batch_size_multiple": 1,
                "pad_to_length_multiple": 64,
            },
        ),
        experiment(
            name="lb_mt16384_pad64_req4_600s",
            description="length_bucketing + max_tokens 16384 + pad64 + required4 [600s sampler-speed]",
            budget_seconds=600,
            batch_size=64,
            data_updates={
                "length_bucketing": True,
                "bucket_padding_noise": 0.0,
                "max_tokens_per_batch": 16384,
                "required_batch_size_multiple": 4,
                "pad_to_length_multiple": 64,
            },
        ),
    ]


def build_stage_7200_experiments(fastest_sampler: ExperimentResult) -> list[dict[str, Any]]:
    experiments = [
        experiment(
            name="baseline_fixed_7200s",
            description="fixed batch baseline [7200s canonical]",
            budget_seconds=7200,
            batch_size=16,
            data_updates={
                "length_bucketing": False,
                "bucket_padding_noise": 0.0,
                "max_tokens_per_batch": None,
                "required_batch_size_multiple": 1,
                "pad_to_length_multiple": 1,
            },
        )
    ]
    sampler_settings = settings_from_description(fastest_sampler.description)
    experiments.append(
        experiment(
            name=f"{sanitize_name(fastest_sampler.description)}_7200s",
            description=f"{fastest_sampler.description.replace('[600s sampler-speed]', '').strip()} [7200s canonical]",
            budget_seconds=7200,
            batch_size=sampler_settings["batch_size"],
            data_updates=sampler_settings["data_updates"],
        )
    )
    return experiments


def build_endless_sampler_experiments():
    max_token_values = [10240, 12288, 14336, 16384, 18432, 20480, 24576]
    pad_values = [1, 16, 32, 64, 128]
    noise_values = [0.0, 0.05, 0.1, 0.2]
    required_values = [1, 2, 4, 8]

    for max_tokens in max_token_values:
        for pad_multiple in pad_values:
            yield experiment(
                name=f"grid_lb_mt{max_tokens}_pad{pad_multiple}_1800s",
                description=f"grid length_bucketing + max_tokens {max_tokens} + pad{pad_multiple} [1800s sampler-grid]",
                budget_seconds=1800,
                batch_size=64,
                data_updates={
                    "length_bucketing": True,
                    "bucket_padding_noise": 0.0,
                    "max_tokens_per_batch": max_tokens,
                    "required_batch_size_multiple": 1,
                    "pad_to_length_multiple": pad_multiple,
                },
            )
            for noise in noise_values:
                if noise == 0.0:
                    continue
                yield experiment(
                    name=f"grid_lb_mt{max_tokens}_pad{pad_multiple}_noise{format_decimal(noise)}_1800s",
                    description=(
                        "grid length_bucketing + max_tokens "
                        f"{max_tokens} + pad{pad_multiple} + noise{noise:.2f} [1800s sampler-grid]"
                    ),
                    budget_seconds=1800,
                    batch_size=64,
                    data_updates={
                        "length_bucketing": True,
                        "bucket_padding_noise": noise,
                        "max_tokens_per_batch": max_tokens,
                        "required_batch_size_multiple": 1,
                        "pad_to_length_multiple": pad_multiple,
                    },
                )
            for required_multiple in required_values:
                if required_multiple == 1:
                    continue
                yield experiment(
                    name=f"grid_lb_mt{max_tokens}_pad{pad_multiple}_req{required_multiple}_1800s",
                    description=(
                        "grid length_bucketing + max_tokens "
                        f"{max_tokens} + pad{pad_multiple} + required{required_multiple} [1800s sampler-grid]"
                    ),
                    budget_seconds=1800,
                    batch_size=64,
                    data_updates={
                        "length_bucketing": True,
                        "bucket_padding_noise": 0.0,
                        "max_tokens_per_batch": max_tokens,
                        "required_batch_size_multiple": required_multiple,
                        "pad_to_length_multiple": pad_multiple,
                    },
                )


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


def maybe_run_experiment(
    experiment_spec: dict[str, Any],
    *,
    commit: str,
    seen_descriptions: set[str],
) -> ExperimentResult | None:
    description = experiment_spec["description"]
    if description in seen_descriptions:
        print(f"[sampler-research] skip existing: {description}", flush=True)
        return None

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
    return result


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
            timeout=budget_seconds + 1800,
            check=True,
        )
    except subprocess.TimeoutExpired:
        return ExperimentResult(
            description=description,
            valid_loss=0.0,
            tokens_per_second=0.0,
            status="crash",
            run_dir=None,
        )
    except subprocess.CalledProcessError as error:
        run_dir = parse_run_dir(error.stdout)
        return ExperimentResult(
            description=description,
            valid_loss=0.0,
            tokens_per_second=0.0,
            status="crash",
            run_dir=run_dir,
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
    match = re.search(r"run_dir=(.+)", stdout)
    if match is None:
        return None
    return match.group(1).strip()


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


def load_results(path: Path) -> list[ExperimentResult]:
    if not path.exists():
        return []
    results: list[ExperimentResult] = []
    for line in path.read_text(encoding="utf-8").splitlines()[1:]:
        if not line.strip():
            continue
        commit, valid_loss, toks_per_sec, status, description = line.split("\t", maxsplit=4)
        result = ExperimentResult(
            description=description.split(" run=")[0],
            valid_loss=float(valid_loss),
            tokens_per_second=float(toks_per_sec),
            status=status,
            run_dir=None,
        )
        results.append(result)
    return results


def pick_fastest_sampler(results: list[ExperimentResult]) -> ExperimentResult | None:
    sampler_results = [
        result
        for result in results
        if "[600s sampler-speed]" in result.description and "fixed batch baseline" not in result.description
    ]
    if not sampler_results:
        return None
    return max(sampler_results, key=lambda result: result.tokens_per_second)


def settings_from_description(description: str) -> dict[str, Any]:
    settings = {
        "batch_size": 64,
        "data_updates": {
            "length_bucketing": "length_bucketing" in description,
            "bucket_padding_noise": 0.0,
            "max_tokens_per_batch": None,
            "required_batch_size_multiple": 1,
            "pad_to_length_multiple": 1,
        },
    }
    max_tokens_match = re.search(r"max_tokens(?:_per_batch)? (\d+)", description)
    if max_tokens_match is not None:
        settings["data_updates"]["max_tokens_per_batch"] = int(max_tokens_match.group(1))
    noise_match = re.search(r"noise([0-9.]+)", description)
    if noise_match is not None:
        settings["data_updates"]["bucket_padding_noise"] = float(noise_match.group(1))
    pad_match = re.search(r"pad(\d+)", description)
    if pad_match is not None:
        settings["data_updates"]["pad_to_length_multiple"] = int(pad_match.group(1))
    required_match = re.search(r"required(\d+)", description)
    if required_match is not None:
        settings["data_updates"]["required_batch_size_multiple"] = int(required_match.group(1))
    return settings


def sanitize_name(description: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", description.lower()).strip("_")


def format_decimal(value: float) -> str:
    return str(value).replace(".", "")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("[sampler-research] interrupted", file=sys.stderr)
        raise
