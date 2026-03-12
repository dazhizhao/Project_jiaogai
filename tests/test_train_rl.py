from __future__ import annotations

from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest


def test_train_rl_smoke_creates_expected_outputs(tmp_path: Path):
    pytest.importorskip("gymnasium")
    pytest.importorskip("stable_baselines3")

    output_dir = tmp_path / "results"
    command = [
        sys.executable,
        "scripts/train_rl.py",
        "--total-timesteps",
        "32",
        "--eval-episodes",
        "2",
        "--run-name",
        "smoke",
        "--output-dir",
        str(output_dir),
    ]
    subprocess.run(command, cwd=Path(__file__).resolve().parents[1], check=True)

    run_dir = output_dir / "rl_link_alloc" / "smoke"
    assert (run_dir / "model_final.zip").exists()
    assert (run_dir / "evaluation.json").exists()
    assert (run_dir / "best_lengths.json").exists()
    assert (run_dir / "train_config.json").exists()
    assert (run_dir / "best_workspace_samples.npz").exists()
    assert (run_dir / "best_workspace.png").exists()
    assert (run_dir / "best_workspace.mp4").exists()
    assert (run_dir / "best_workspace.mp4").stat().st_size > 0

    samples = np.load(run_dir / "best_workspace_samples.npz")
    assert set(samples.files) >= {"points", "lengths", "occupied_ratio", "xy_bounds"}
