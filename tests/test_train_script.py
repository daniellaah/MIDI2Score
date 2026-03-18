from pathlib import Path
import subprocess


def test_train_script_runs_from_yaml_config() -> None:
    result = subprocess.run(
        ["uv", "run", "python", "train.py", "--config", str(Path("configs/baseline.yaml"))],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "finished 5 steps on" in result.stdout
