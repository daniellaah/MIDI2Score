from pathlib import Path
import json
import subprocess


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
                "  max_length: 64",
                "data:",
                "  dataset_path: data/huggingface",
                "  split: training",
                "  max_length: 64",
                "  sliding_window_stride: 32",
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

    assert "finished 2 pretraining steps on cpu" in result.stdout
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
    assert "average_step_time_seconds" in summary["result"]
    assert "average_tokens_per_second" in summary["result"]
    assert "mps_peak_memory_bytes" in summary["result"]
    assert "target_validation_loss" in summary["result"]
    assert "time_to_target_validation_loss_seconds" in summary["result"]
