from pathlib import Path
import json
import subprocess

import torch
from datasets import Dataset, DatasetDict

from pretrain.decoder import DecoderLanguageModelConfig, TransformerDecoderLM


def test_pretrain_script_runs_from_yaml_config(tmp_path: Path) -> None:
    config_path = tmp_path / "pretrain.yaml"
    runs_root = tmp_path / "runs"
    config_path.write_text(
        "\n".join(
            [
                "model:",
                "  vocab_size: 5000",
                "  d_model: 32",
                "  nhead: 4",
                "  num_layers: 2",
                "  dim_feedforward: 64",
                "  dropout: 0.1",
                "  activation: relu",
                "  pad_token_id: 0",
                "  bos_token_id: 1",
                "  eos_token_id: 2",
                "  max_length: 1024",
                "data:",
                "  dataset_path: data/bar_aware_chunk/training_bar_chunk_encoded_overlap2_full_dataset",
                "  split: training",
                "  max_length: 1024",
                "  num_workers: 0",
                "training:",
                "  seed: 0",
                "  batch_size: 4",
                "  learning_rate: 0.001",
                "  num_steps: 2",
                "  log_every: 1",
                "  eval_every: 1",
                "  num_eval_batches: 1",
                "  device: cpu",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            "uv",
            "run",
            "python",
            "run_pretrain.py",
            "--config",
            str(config_path),
            "--runs-root",
            str(runs_root),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "finished " in result.stdout
    assert "pretraining steps on cpu" in result.stdout
    assert "best validation loss" in result.stdout
    run_dirs = [path for path in runs_root.iterdir() if path.is_dir()]
    assert len(run_dirs) == 1
    run_dir = run_dirs[0]
    assert (run_dir / "latest.pt").exists()
    assert (run_dir / "best.pt").exists()
    assert (run_dir / "summary.json").exists()
    assert (run_dir / "config.yaml").exists()
    assert (run_dir / "train.csv").exists()
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["result"]["final_step"] > 0
    assert "average_step_time_seconds" in summary["result"]
    assert "average_tokens_per_second" in summary["result"]
    assert "mps_peak_memory_bytes" in summary["result"]
    assert "target_validation_loss" in summary["result"]
    assert "time_to_target_validation_loss_seconds" in summary["result"]


def test_pretrain_script_eval_mode_runs_from_checkpoint(tmp_path: Path) -> None:
    dataset_path = tmp_path / "dataset"
    dataset_dict = DatasetDict(
        {
            "training": Dataset.from_list([{"input_ids": [1, 4, 5, 2]}]),
            "validation": Dataset.from_list([{"input_ids": [1, 4, 5, 2]}]),
            "test": Dataset.from_list([{"input_ids": [1, 4, 5, 2]}]),
        }
    )
    dataset_dict.save_to_disk(str(dataset_path))

    config_path = tmp_path / "pretrain.yaml"
    checkpoint_path = tmp_path / "checkpoint.pt"
    output_path = tmp_path / "metrics.json"
    config_path.write_text(
        "\n".join(
            [
                "model:",
                "  vocab_size: 32",
                "  d_model: 16",
                "  nhead: 4",
                "  num_layers: 2",
                "  dim_feedforward: 32",
                "  dropout: 0.1",
                "  activation: relu",
                "  pad_token_id: 0",
                "  bos_token_id: 1",
                "  eos_token_id: 2",
                "  max_length: 1024",
                "data:",
                f"  dataset_path: {dataset_path}",
                "  split: training",
                "  max_length: 1024",
                "  num_workers: 0",
                "training:",
                "  seed: 0",
                "  batch_size: 4",
                "  eval_batch_size: 2",
                "  learning_rate: 0.001",
                "  num_steps: 2",
                "  log_every: 1",
                "  eval_every: 1",
                "  num_eval_batches: 1",
                "  device: cpu",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    model_config = DecoderLanguageModelConfig(
        vocab_size=32,
        d_model=16,
        nhead=4,
        num_layers=2,
        dim_feedforward=32,
        dropout=0.1,
        activation="relu",
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        max_length=1024,
    )
    model = TransformerDecoderLM(model_config)
    torch.save(
        {
            "model_state": model.state_dict(),
            "model_config": model_config.to_dict(),
        },
        checkpoint_path,
    )

    result = subprocess.run(
        [
            "uv",
            "run",
            "python",
            "run_pretrain.py",
            "--eval-mode",
            "--config",
            str(config_path),
            "--checkpoint",
            str(checkpoint_path),
            "--split",
            "validation",
            "--device",
            "cpu",
            "--num-batches",
            "1",
            "--output",
            str(output_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "split=validation device=cpu" in result.stdout
    assert "loss=" in result.stdout
    assert output_path.exists()
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["split"] == "validation"
    assert payload["device"] == "cpu"
    assert "loss" in payload["metrics"]


def test_pretrain_script_eval_mode_honors_test_split(tmp_path: Path) -> None:
    dataset_path = tmp_path / "dataset"
    dataset_dict = DatasetDict(
        {
            "training": Dataset.from_list([{"input_ids": [1, 4, 5, 2]}]),
            "validation": Dataset.from_list([{"input_ids": [1, 4, 5, 2]}]),
            "test": Dataset.from_list([{"input_ids": [1, 6, 7, 2]}]),
        }
    )
    dataset_dict.save_to_disk(str(dataset_path))

    config_path = tmp_path / "pretrain.yaml"
    checkpoint_path = tmp_path / "checkpoint.pt"
    output_path = tmp_path / "metrics.json"
    config_path.write_text(
        "\n".join(
            [
                "model:",
                "  vocab_size: 32",
                "  d_model: 16",
                "  nhead: 4",
                "  num_layers: 2",
                "  dim_feedforward: 32",
                "  dropout: 0.1",
                "  activation: relu",
                "  pad_token_id: 0",
                "  bos_token_id: 1",
                "  eos_token_id: 2",
                "  max_length: 1024",
                "data:",
                f"  dataset_path: {dataset_path}",
                "  split: training",
                "  max_length: 1024",
                "  num_workers: 0",
                "training:",
                "  seed: 0",
                "  batch_size: 4",
                "  eval_batch_size: 2",
                "  learning_rate: 0.001",
                "  num_steps: 2",
                "  log_every: 1",
                "  eval_every: 1",
                "  num_eval_batches: 1",
                "  device: cpu",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    model_config = DecoderLanguageModelConfig(
        vocab_size=32,
        d_model=16,
        nhead=4,
        num_layers=2,
        dim_feedforward=32,
        dropout=0.1,
        activation="relu",
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        max_length=1024,
    )
    model = TransformerDecoderLM(model_config)
    torch.save(
        {
            "model_state": model.state_dict(),
            "model_config": model_config.to_dict(),
        },
        checkpoint_path,
    )

    result = subprocess.run(
        [
            "uv",
            "run",
            "python",
            "run_pretrain.py",
            "--eval-mode",
            "--config",
            str(config_path),
            "--checkpoint",
            str(checkpoint_path),
            "--split",
            "test",
            "--device",
            "cpu",
            "--num-batches",
            "1",
            "--output",
            str(output_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "split=test device=cpu" in result.stdout
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["split"] == "test"
    assert payload["metrics"]["evaluated_tokens"] == 3


def test_pretrain_script_eval_mode_uses_checkpoint_model_config(tmp_path: Path) -> None:
    dataset_path = tmp_path / "dataset"
    dataset_dict = DatasetDict(
        {
            "training": Dataset.from_list([{"input_ids": [1, 4, 5, 2]}]),
            "validation": Dataset.from_list([{"input_ids": [1, 4, 5, 2]}]),
            "test": Dataset.from_list([{"input_ids": [1, 4, 5, 2]}]),
        }
    )
    dataset_dict.save_to_disk(str(dataset_path))

    config_path = tmp_path / "pretrain.yaml"
    checkpoint_path = tmp_path / "checkpoint.pt"
    config_path.write_text(
        "\n".join(
            [
                "model:",
                "  vocab_size: 64",
                "  d_model: 32",
                "  nhead: 4",
                "  num_layers: 3",
                "  dim_feedforward: 96",
                "  dropout: 0.1",
                "  activation: relu",
                "  pad_token_id: 0",
                "  bos_token_id: 1",
                "  eos_token_id: 2",
                "  max_length: 1024",
                "data:",
                f"  dataset_path: {dataset_path}",
                "  split: training",
                "  max_length: 1024",
                "  num_workers: 0",
                "training:",
                "  seed: 0",
                "  batch_size: 4",
                "  eval_batch_size: 2",
                "  learning_rate: 0.001",
                "  num_steps: 2",
                "  log_every: 1",
                "  eval_every: 1",
                "  num_eval_batches: 1",
                "  device: cpu",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    checkpoint_model_config = DecoderLanguageModelConfig(
        vocab_size=32,
        d_model=16,
        nhead=4,
        num_layers=2,
        dim_feedforward=32,
        dropout=0.1,
        activation="relu",
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        max_length=1024,
    )
    model = TransformerDecoderLM(checkpoint_model_config)
    torch.save(
        {
            "model_state": model.state_dict(),
            "model_config": checkpoint_model_config.to_dict(),
        },
        checkpoint_path,
    )

    result = subprocess.run(
        [
            "uv",
            "run",
            "python",
            "run_pretrain.py",
            "--eval-mode",
            "--config",
            str(config_path),
            "--checkpoint",
            str(checkpoint_path),
            "--split",
            "validation",
            "--device",
            "cpu",
            "--num-batches",
            "1",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "split=validation device=cpu" in result.stdout
    assert "loss=" in result.stdout


def test_pretrain_script_preserves_resume_checkpoint_path(tmp_path: Path) -> None:
    dataset_path = tmp_path / "dataset"
    runs_root = tmp_path / "runs"
    checkpoint_path = tmp_path / "resume.pt"
    dataset_dict = DatasetDict(
        {
            "training": Dataset.from_list([{"input_ids": [1, 4, 5, 2]}]),
            "validation": Dataset.from_list([{"input_ids": [1, 4, 5, 2]}]),
            "test": Dataset.from_list([{"input_ids": [1, 4, 5, 2]}]),
        }
    )
    dataset_dict.save_to_disk(str(dataset_path))

    model_config = DecoderLanguageModelConfig(
        vocab_size=32,
        d_model=16,
        nhead=4,
        num_layers=2,
        dim_feedforward=32,
        dropout=0.1,
        activation="relu",
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        max_length=1024,
    )
    model = TransformerDecoderLM(model_config)
    torch.save(
        {
            "model_state": model.state_dict(),
            "model_config": model_config.to_dict(),
            "step": 1,
            "best_validation_loss": 1.0,
        },
        checkpoint_path,
    )

    config_path = tmp_path / "pretrain.yaml"
    config_path.write_text(
        "\n".join(
            [
                "model:",
                "  vocab_size: 32",
                "  d_model: 16",
                "  nhead: 4",
                "  num_layers: 2",
                "  dim_feedforward: 32",
                "  dropout: 0.1",
                "  activation: relu",
                "  pad_token_id: 0",
                "  bos_token_id: 1",
                "  eos_token_id: 2",
                "  max_length: 1024",
                "data:",
                f"  dataset_path: {dataset_path}",
                "  split: training",
                "  max_length: 1024",
                "  num_workers: 0",
                "training:",
                "  seed: 0",
                "  batch_size: 4",
                "  eval_batch_size: 2",
                "  learning_rate: 0.001",
                "  num_steps: 2",
                "  log_every: 1",
                "  eval_every: 1",
                "  num_eval_batches: 1",
                "  device: cpu",
                f"  resume_checkpoint_path: {checkpoint_path}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            "uv",
            "run",
            "python",
            "run_pretrain.py",
            "--config",
            str(config_path),
            "--runs-root",
            str(runs_root),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "finished 1 pretraining steps on cpu" in result.stdout
    run_dirs = [path for path in runs_root.iterdir() if path.is_dir()]
    assert len(run_dirs) == 1
    summary = json.loads((run_dirs[0] / "summary.json").read_text(encoding="utf-8"))
    assert summary["result"]["resumed_from_checkpoint"] == str(checkpoint_path)
