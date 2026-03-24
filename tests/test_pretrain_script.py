from pathlib import Path
import subprocess


def test_pretrain_script_runs_from_yaml_config(tmp_path: Path) -> None:
    checkpoint_path = tmp_path / "decoder.pt"
    config_path = tmp_path / "pretrain.yaml"
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
                "  pad_token_id: 0",
                "  bos_token_id: 1",
                "  eos_token_id: 2",
                "  tokenizer_path: data/tokenizer_rd.json",
                "  random_crop: false",
                "  crop_seed: 0",
                "  num_workers: 0",
                "training:",
                "  batch_size: 4",
                "  learning_rate: 0.001",
                "  num_steps: 2",
                "  log_every: 1",
                "  eval_every: 1",
                "  num_eval_batches: 1",
                "  device: cpu",
                f"  save_checkpoint_path: {checkpoint_path}",
                f"  save_best_checkpoint_path: {tmp_path / 'decoder-best.pt'}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    result = subprocess.run(
        ["uv", "run", "python", "pretrain.py", "--config", str(config_path)],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "finished 2 pretraining steps on cpu" in result.stdout
    assert "best validation loss" in result.stdout
    assert checkpoint_path.exists()
