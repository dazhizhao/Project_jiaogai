from __future__ import annotations

import argparse
import copy
from dataclasses import asdict, dataclass
from pathlib import Path
import json
import sys
from typing import Any

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

try:
    import yaml
except ImportError:  # pragma: no cover - exercised only without PyYAML
    yaml = None

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from env.bridge_robot_env import EnvConfig, RobotState
from scripts.output_utils import build_run_dir, ensure_artifacts_dir
from env.torque_control_env import TorqueControlEnv
from scripts.run_env import save_rollout_npz
from visualization.plots import save_training_curves
from visualization.render import render_environment_state
from visualization.video import export_rollout_video


@dataclass(frozen=True)
class TrainConfig:
    algo: str
    policy: str
    total_timesteps: int
    seed: int
    device: str
    learning_starts: int
    buffer_size: int
    batch_size: int
    train_freq: int
    gradient_steps: int
    learning_rate: float
    gamma: float
    tau: float
    eval_episodes: int
    run_name: str
    output_dir: str

    @classmethod
    def load(cls, config_path: str | Path | None = None) -> "TrainConfig":
        path = (
            Path(config_path)
            if config_path
            else ROOT / "configs" / "train_rl.yaml"
        )
        text = path.read_text(encoding="utf-8")
        raw = yaml.safe_load(text) if yaml is not None else json.loads(text)
        return cls(**raw)


@dataclass(frozen=True)
class EpisodeSummary:
    seed: int
    reward: float
    success: bool
    final_distance: float
    episode_length: int
    mean_torque_norm: float
    mean_motion_penalty: float
    mean_smoothness_penalty: float
    history: list[dict[str, Any]]
    final_state: RobotState


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train SAC for the torque control environment.")
    parser.add_argument("--env-config", default=None, help="Path to bridge robot env config.")
    parser.add_argument("--train-config", default=None, help="Path to RL train config.")
    parser.add_argument("--total-timesteps", type=int, default=None, help="Override total timesteps.")
    parser.add_argument("--seed", type=int, default=None, help="Override training seed.")
    parser.add_argument("--run-name", default=None, help="Override output run directory name.")
    parser.add_argument("--output-dir", default=None, help="Override output root directory.")
    parser.add_argument(
        "--eval-episodes",
        type=int,
        default=None,
        help="Override deterministic evaluation episode count.",
    )
    return parser


def load_sac():
    from stable_baselines3 import SAC
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.logger import configure
    from stable_baselines3.common.monitor import Monitor

    return SAC, Monitor, configure, BaseCallback


def build_progress_callback(base_callback_cls, total_timesteps: int):
    class TqdmProgressCallback(base_callback_cls):
        def __init__(self) -> None:
            super().__init__()
            self._progress_bar: tqdm | None = None
            self._last_timestep = 0

        def _on_training_start(self) -> None:
            self._progress_bar = tqdm(
                total=total_timesteps,
                desc="train_rl",
                dynamic_ncols=True,
                leave=True,
            )

        def _on_step(self) -> bool:
            if self._progress_bar is None:
                return True
            current_timestep = min(int(self.num_timesteps), total_timesteps)
            delta = current_timestep - self._last_timestep
            if delta > 0:
                self._progress_bar.update(delta)
                self._last_timestep = current_timestep
            return True

        def _on_training_end(self) -> None:
            if self._progress_bar is None:
                return
            remaining = total_timesteps - self._last_timestep
            if remaining > 0:
                self._progress_bar.update(remaining)
            self._progress_bar.close()
            self._progress_bar = None

    return TqdmProgressCallback()


def ensure_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def clone_history(history: list[dict[str, Any]]) -> list[dict[str, Any]]:
    cloned: list[dict[str, Any]] = []
    for item in history:
        snapshot: dict[str, Any] = {}
        for key, value in item.items():
            snapshot[key] = value.copy() if isinstance(value, np.ndarray) else value
        cloned.append(snapshot)
    return cloned


def run_deterministic_episode(model, env: TorqueControlEnv, seed: int) -> EpisodeSummary:
    observation, _ = env.reset(seed=seed)
    episode_reward = 0.0
    torque_norms: list[float] = []
    motion_penalties: list[float] = []
    smoothness_penalties: list[float] = []
    terminated = False
    truncated = False
    info: dict[str, Any] = {}

    while not (terminated or truncated):
        action, _ = model.predict(observation, deterministic=True)
        observation, reward, terminated, truncated, info = env.step(action)
        episode_reward += float(reward)
        reward_terms = info["reward_terms"]
        torque_norms.append(float(np.linalg.norm(np.asarray(info["applied_action"], dtype=float))))
        motion_penalties.append(float(reward_terms["motion_penalty"]))
        smoothness_penalties.append(float(reward_terms["smoothness_penalty"]))

    base_env = env.base_env
    if base_env.state is None:
        raise RuntimeError("BridgeRobotEnv state is unexpectedly unavailable after evaluation.")

    return EpisodeSummary(
        seed=seed,
        reward=episode_reward,
        success=bool(info["success"]),
        final_distance=float(base_env.state.distance_to_target),
        episode_length=int(base_env.state.step_count),
        mean_torque_norm=float(np.mean(torque_norms)) if torque_norms else 0.0,
        mean_motion_penalty=float(np.mean(motion_penalties)) if motion_penalties else 0.0,
        mean_smoothness_penalty=float(np.mean(smoothness_penalties)) if smoothness_penalties else 0.0,
        history=clone_history(base_env.history),
        final_state=copy.deepcopy(base_env.state),
    )


def evaluate_policy(model, env: TorqueControlEnv, eval_episodes: int, base_seed: int) -> dict[str, Any]:
    episodes: list[EpisodeSummary] = []
    for episode_idx in range(eval_episodes):
        episodes.append(run_deterministic_episode(model, env, seed=base_seed + episode_idx))

    successful_episodes = [episode for episode in episodes if episode.success]
    best_episode = max(
        successful_episodes if successful_episodes else episodes,
        key=lambda episode: episode.reward,
    )

    return {
        "success_rate": float(np.mean([episode.success for episode in episodes])),
        "mean_reward": float(np.mean([episode.reward for episode in episodes])),
        "mean_final_distance": float(np.mean([episode.final_distance for episode in episodes])),
        "mean_episode_length": float(np.mean([episode.episode_length for episode in episodes])),
        "mean_torque_norm": float(np.mean([episode.mean_torque_norm for episode in episodes])),
        "mean_motion_penalty": float(np.mean([episode.mean_motion_penalty for episode in episodes])),
        "mean_smoothness_penalty": float(np.mean([episode.mean_smoothness_penalty for episode in episodes])),
        "episodes": int(eval_episodes),
        "best_episode": best_episode,
    }


def export_best_episode(
    artifacts_dir: Path,
    env_config: EnvConfig,
    best_episode: EpisodeSummary,
) -> dict[str, str]:
    history = best_episode.history
    final_state = best_episode.final_state

    rollout_path = save_rollout_npz(
        output_path=artifacts_dir / "best_rollout.npz",
        steps=[int(item["step"]) for item in history],
        rewards=[float(item["reward"]) for item in history],
        distances=[float(item["distance_to_target"]) for item in history],
        terminated=[bool(item["terminated"]) for item in history],
        truncated=[bool(item["truncated"]) for item in history],
        end_effector_pos=[np.asarray(item["end_effector_pos"], dtype=float) for item in history],
        target_pos=[np.asarray(item["target_pos"], dtype=float) for item in history],
        joint_torques=[np.asarray(item["joint_torques"], dtype=float) for item in history],
        joint_power=[np.asarray(item["joint_power"], dtype=float) for item in history],
        applied_action=[np.asarray(item["applied_action"], dtype=float) for item in history],
        seed=best_episode.seed,
        policy="sac_deterministic",
    )
    pose_path = artifacts_dir / "best_pose.png"
    pose_figure = render_environment_state(
        state=final_state,
        history=history,
        config=env_config.render,
        save_path=pose_path,
        show=False,
    )
    plt.close(pose_figure)
    video_path = export_rollout_video(
        history=history,
        link_lengths=env_config.robot.link_lengths,
        target_pos=final_state.target_pos,
        output_path=artifacts_dir / "best_rollout.mp4",
        fps=15,
    )
    return {
        "best_rollout": str(rollout_path),
        "best_pose": str(pose_path),
        "rollout_video": str(video_path),
    }


def main() -> None:
    args = build_parser().parse_args()
    env_config = EnvConfig.load(args.env_config)
    train_config = TrainConfig.load(args.train_config)

    if train_config.algo.lower() != "sac":
        raise SystemExit("Only SAC is supported in this training entrypoint.")

    total_timesteps = args.total_timesteps if args.total_timesteps is not None else train_config.total_timesteps
    seed = args.seed if args.seed is not None else train_config.seed
    run_name = args.run_name if args.run_name is not None else train_config.run_name
    eval_episodes = args.eval_episodes if args.eval_episodes is not None else train_config.eval_episodes
    output_root = Path(args.output_dir if args.output_dir is not None else train_config.output_dir)
    run_dir, resolved_run_name = build_run_dir(output_root, "rl_torque_control", run_name)
    artifacts_dir = ensure_artifacts_dir(run_dir)

    SAC, Monitor, configure, BaseCallback = load_sac()

    train_env = Monitor(TorqueControlEnv(config=env_config), filename=str(run_dir / "monitor.csv"))
    eval_env = TorqueControlEnv(config=env_config)

    model = SAC(
        policy=train_config.policy,
        env=train_env,
        learning_starts=train_config.learning_starts,
        buffer_size=train_config.buffer_size,
        batch_size=train_config.batch_size,
        train_freq=train_config.train_freq,
        gradient_steps=train_config.gradient_steps,
        learning_rate=train_config.learning_rate,
        gamma=train_config.gamma,
        tau=train_config.tau,
        seed=seed,
        device=train_config.device,
        verbose=0,
    )
    model.set_logger(configure(str(run_dir), ["csv"]))
    print(f"training torque control: run={resolved_run_name} timesteps={total_timesteps} seed={seed}")
    progress_callback = build_progress_callback(BaseCallback, total_timesteps=total_timesteps)
    model.learn(total_timesteps=total_timesteps, progress_bar=False, callback=progress_callback)

    model_path = run_dir / "model_final.zip"
    model.save(str(model_path))
    training_curves_path = save_training_curves(
        output_path=artifacts_dir / "training_curves.png",
        monitor_csv_path=run_dir / "monitor.csv",
        progress_csv_path=run_dir / "progress.csv",
    )

    evaluation = evaluate_policy(model, eval_env, eval_episodes=eval_episodes, base_seed=seed + 1000)
    artifact_paths = export_best_episode(
        artifacts_dir=artifacts_dir,
        env_config=env_config,
        best_episode=evaluation["best_episode"],
    )
    artifact_paths["training_curves"] = str(training_curves_path)
    summary_payload = {
        "env_config": asdict(env_config),
        "train_config": asdict(train_config),
        "resolved": {
            "total_timesteps": total_timesteps,
            "seed": seed,
            "eval_episodes": eval_episodes,
            "run_name": run_name,
            "resolved_run_name": resolved_run_name,
            "run_dir": str(run_dir),
        },
        "evaluation": {
            "success_rate": evaluation["success_rate"],
            "mean_reward": evaluation["mean_reward"],
            "mean_final_distance": evaluation["mean_final_distance"],
            "mean_episode_length": evaluation["mean_episode_length"],
            "mean_torque_norm": evaluation["mean_torque_norm"],
            "mean_motion_penalty": evaluation["mean_motion_penalty"],
            "mean_smoothness_penalty": evaluation["mean_smoothness_penalty"],
            "episodes": evaluation["episodes"],
            "best_episode_seed": evaluation["best_episode"].seed,
            "best_episode_reward": evaluation["best_episode"].reward,
            "best_episode_success": evaluation["best_episode"].success,
            "artifact_paths": artifact_paths,
        },
    }
    ensure_json(run_dir / "summary.json", summary_payload)

    train_env.close()
    eval_env.close()

    print(f"saved model to {model_path}")
    print(f"saved summary to {run_dir / 'summary.json'}")
    print(f"saved artifacts to {artifacts_dir}")
    print(f"saved rollout video to {artifact_paths['rollout_video']}")


if __name__ == "__main__":
    main()
