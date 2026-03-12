from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence
import json

import numpy as np

try:
    import yaml
except ImportError:  # pragma: no cover - exercised only without PyYAML
    yaml = None

from .dynamics import compute_equivalent_inertia, compute_gravity_torques, step_dynamics
from .kinematics import forward_kinematics, is_pose_above_ground, total_reach
from .reward import compute_reward


@dataclass(frozen=True)
class SimConfig:
    dt: float
    max_steps: int
    gravity: float
    integrator: str


@dataclass(frozen=True)
class RobotConfig:
    link_lengths: list[float]
    link_masses: list[float]
    payload_mass: float
    joint_damping: list[float]
    joint_limits: list[list[float]]
    torque_limits: list[float]


@dataclass(frozen=True)
class TargetSamplingConfig:
    x: list[float]
    y: list[float]
    radius: list[float]


@dataclass(frozen=True)
class TaskConfig:
    home_pose: list[float]
    target_sampling: TargetSamplingConfig
    ground_y: float
    success_tolerance: float
    success_hold_steps: int


@dataclass(frozen=True)
class RewardConfig:
    progress_weight: float
    torque_weight: float
    motion_weight: float
    smoothness_weight: float
    ground_contact_penalty: float
    hold_bonus_weight: float
    success_bonus: float


@dataclass(frozen=True)
class RenderConfig:
    figsize: list[float]
    dpi: int
    history_alpha: float


@dataclass(frozen=True)
class IOConfig:
    output_dir: str


@dataclass(frozen=True)
class EnvConfig:
    sim: SimConfig
    robot: RobotConfig
    task: TaskConfig
    reward: RewardConfig
    render: RenderConfig
    io: IOConfig

    @classmethod
    def load(cls, config_path: str | Path | None = None) -> "EnvConfig":
        path = (
            Path(config_path)
            if config_path
            else Path(__file__).resolve().parents[1] / "configs" / "default.yaml"
        )
        text = path.read_text(encoding="utf-8")
        raw = yaml.safe_load(text) if yaml is not None else json.loads(text)

        return cls(
            sim=SimConfig(**raw["sim"]),
            robot=RobotConfig(**raw["robot"]),
            task=TaskConfig(
                home_pose=raw["task"]["home_pose"],
                target_sampling=TargetSamplingConfig(**raw["task"]["target_sampling"]),
                ground_y=raw["task"]["ground_y"],
                success_tolerance=raw["task"]["success_tolerance"],
                success_hold_steps=raw["task"]["success_hold_steps"],
            ),
            reward=RewardConfig(**raw["reward"]),
            render=RenderConfig(**raw["render"]),
            io=IOConfig(**raw["io"]),
        )


@dataclass
class RobotState:
    q: np.ndarray
    qd: np.ndarray
    qdd: np.ndarray
    joint_positions: np.ndarray
    end_effector_pos: np.ndarray
    end_effector_vel: np.ndarray
    end_effector_acc: np.ndarray
    joint_torques: np.ndarray
    joint_power: np.ndarray
    step_count: int
    applied_action: np.ndarray
    applied_action_norm: np.ndarray
    target_pos: np.ndarray
    distance_to_target: float
    gravity_torques: np.ndarray
    equivalent_inertia: np.ndarray
    consecutive_success_steps: int
    hold_progress: float
    success_ready: bool


@dataclass(frozen=True)
class StepResult:
    observation: dict[str, Any]
    reward: float
    terminated: bool
    truncated: bool
    info: dict[str, Any]


class BridgeRobotEnv:
    def __init__(self, config: EnvConfig | None = None, config_path: str | Path | None = None) -> None:
        self.config = config if config is not None else EnvConfig.load(config_path)
        self.rng = np.random.default_rng()
        self.state: RobotState | None = None
        self.history: list[dict[str, Any]] = []
        self.previous_action_norm = np.zeros(4, dtype=float)
        self._last_figure = None

    def reset(
        self, seed: int | None = None, target: Sequence[float] | None = None
    ) -> dict[str, Any]:
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        q0 = np.asarray(self.config.task.home_pose, dtype=float)
        kin = forward_kinematics(q0, self.config.robot.link_lengths)
        if not is_pose_above_ground(kin.joint_positions, self.config.task.ground_y):
            raise ValueError("home_pose places the robot below the ground boundary.")

        target_pos = (
            self._validate_target(target)
            if target is not None
            else self._sample_target()
        )
        distance = float(np.linalg.norm(kin.end_effector_pos - target_pos))
        gravity_torques = compute_gravity_torques(
            joint_angles=q0,
            link_lengths=self.config.robot.link_lengths,
            link_masses=self.config.robot.link_masses,
            payload_mass=self.config.robot.payload_mass,
            gravity=self.config.sim.gravity,
        )
        equivalent_inertia = compute_equivalent_inertia(
            joint_angles=q0,
            link_lengths=self.config.robot.link_lengths,
            link_masses=self.config.robot.link_masses,
            payload_mass=self.config.robot.payload_mass,
        )

        self.state = RobotState(
            q=q0,
            qd=np.zeros_like(q0),
            qdd=np.zeros_like(q0),
            joint_positions=kin.joint_positions,
            end_effector_pos=kin.end_effector_pos,
            end_effector_vel=np.zeros(2, dtype=float),
            end_effector_acc=np.zeros(2, dtype=float),
            joint_torques=gravity_torques,
            joint_power=np.zeros(4, dtype=float),
            step_count=0,
            applied_action=np.zeros(4, dtype=float),
            applied_action_norm=np.zeros(4, dtype=float),
            target_pos=target_pos,
            distance_to_target=distance,
            gravity_torques=gravity_torques,
            equivalent_inertia=equivalent_inertia,
            consecutive_success_steps=0,
            hold_progress=0.0,
            success_ready=False,
        )
        self.history = []
        self.previous_action_norm = np.zeros(4, dtype=float)
        self._record_history(reward=0.0, terminated=False, truncated=False)
        return self._build_observation()

    def step(self, action: Sequence[float]) -> StepResult:
        if self.state is None:
            raise RuntimeError("Call reset() before step().")

        result = step_dynamics(
            joint_angles=self.state.q,
            joint_velocities=self.state.qd,
            action=action,
            dt=self.config.sim.dt,
            gravity=self.config.sim.gravity,
            link_lengths=self.config.robot.link_lengths,
            link_masses=self.config.robot.link_masses,
            payload_mass=self.config.robot.payload_mass,
            joint_damping=self.config.robot.joint_damping,
            torque_limits=self.config.robot.torque_limits,
            joint_limits=self.config.robot.joint_limits,
            prev_end_effector_vel=self.state.end_effector_vel,
        )
        torque_limits = np.asarray(self.config.robot.torque_limits, dtype=float)
        applied_action_norm = np.divide(
            result.applied_action,
            torque_limits,
            out=np.zeros_like(result.applied_action),
            where=torque_limits > 0.0,
        )
        previous_distance = float(self.state.distance_to_target)
        distance = float(np.linalg.norm(result.end_effector_pos - self.state.target_pos))
        ground_contact = not is_pose_above_ground(
            result.joint_positions,
            self.config.task.ground_y,
        )
        if ground_contact:
            previous_state = self.state
            distance = float(np.linalg.norm(previous_state.end_effector_pos - previous_state.target_pos))
            result_joint_torques = previous_state.gravity_torques.copy()
            result_gravity_torques = previous_state.gravity_torques.copy()
            result_equivalent_inertia = previous_state.equivalent_inertia.copy()
            result_q = previous_state.q.copy()
            result_joint_positions = previous_state.joint_positions.copy()
            result_end_effector_pos = previous_state.end_effector_pos.copy()
            result_qd = np.zeros_like(previous_state.qd)
            result_qdd = np.zeros_like(previous_state.qdd)
            result_end_effector_vel = np.zeros_like(previous_state.end_effector_vel)
            result_end_effector_acc = np.zeros_like(previous_state.end_effector_acc)
            result_joint_power = np.zeros_like(previous_state.joint_power)
        else:
            result_joint_torques = result.joint_torques
            result_gravity_torques = result.gravity_torques
            result_equivalent_inertia = result.equivalent_inertia
            result_q = result.q
            result_joint_positions = result.joint_positions
            result_end_effector_pos = result.end_effector_pos
            result_qd = result.qd
            result_qdd = result.qdd
            result_end_effector_vel = result.end_effector_vel
            result_end_effector_acc = result.end_effector_acc
            result_joint_power = result.joint_power

        success_ready = distance <= self.config.task.success_tolerance
        consecutive_success_steps = self.state.consecutive_success_steps + 1 if success_ready else 0
        hold_progress = min(
            consecutive_success_steps / float(self.config.task.success_hold_steps),
            1.0,
        )
        success = consecutive_success_steps >= self.config.task.success_hold_steps
        reward_breakdown = compute_reward(
            previous_distance=previous_distance,
            current_distance=distance,
            action_normalized=applied_action_norm,
            joint_velocities=result_qd,
            previous_action=self.previous_action_norm,
            ground_contact=ground_contact,
            hold_progress=hold_progress if success_ready else 0.0,
            success=success,
            progress_weight=self.config.reward.progress_weight,
            torque_weight=self.config.reward.torque_weight,
            motion_weight=self.config.reward.motion_weight,
            smoothness_weight=self.config.reward.smoothness_weight,
            ground_contact_penalty=self.config.reward.ground_contact_penalty,
            hold_bonus_weight=self.config.reward.hold_bonus_weight,
            success_bonus=self.config.reward.success_bonus,
        )

        self.state = RobotState(
            q=result_q,
            qd=result_qd,
            qdd=result_qdd,
            joint_positions=result_joint_positions,
            end_effector_pos=result_end_effector_pos,
            end_effector_vel=result_end_effector_vel,
            end_effector_acc=result_end_effector_acc,
            joint_torques=result_joint_torques,
            joint_power=result_joint_power,
            step_count=self.state.step_count + 1,
            applied_action=result.applied_action,
            applied_action_norm=applied_action_norm,
            target_pos=self.state.target_pos,
            distance_to_target=distance,
            gravity_torques=result_gravity_torques,
            equivalent_inertia=result_equivalent_inertia,
            consecutive_success_steps=consecutive_success_steps,
            hold_progress=hold_progress,
            success_ready=success_ready,
        )
        self.previous_action_norm = applied_action_norm.copy()

        terminated = success or ground_contact
        truncated = (not terminated) and self.state.step_count >= self.config.sim.max_steps
        self._record_history(
            reward=reward_breakdown.total,
            terminated=terminated,
            truncated=truncated,
        )

        info = {
            "reward_terms": reward_breakdown.to_dict(),
            "action_clipped": result.action_clipped,
            "success": success,
            "success_ready": success_ready,
            "ground_contact": ground_contact,
            "joint_limit_violation": bool(result.joint_limit_clipped),
            "workspace_violation": bool(result.joint_limit_clipped or ground_contact),
            "applied_action": result.applied_action.copy(),
            "applied_action_norm": applied_action_norm.copy(),
            "gravity_torques": result_gravity_torques.copy(),
            "equivalent_inertia": result_equivalent_inertia.copy(),
            "consecutive_success_steps": consecutive_success_steps,
            "hold_progress": hold_progress,
        }
        return StepResult(
            observation=self._build_observation(),
            reward=reward_breakdown.total,
            terminated=terminated,
            truncated=truncated,
            info=info,
        )

    def render(self, save_path: str | Path | None = None, show: bool = False):
        if self.state is None:
            raise RuntimeError("Call reset() before render().")

        from visualization.render import render_environment_state

        self._last_figure = render_environment_state(
            state=self.state,
            history=self.history,
            config=self.config.render,
            ground_y=self.config.task.ground_y,
            save_path=save_path,
            show=show,
        )
        return self._last_figure

    def close(self) -> None:
        if self._last_figure is None:
            return
        import matplotlib.pyplot as plt

        plt.close(self._last_figure)
        self._last_figure = None

    def _sample_target(self) -> np.ndarray:
        sampling = self.config.task.target_sampling
        radius_min, radius_max = sampling.radius
        for _ in range(1024):
            x = self.rng.uniform(sampling.x[0], sampling.x[1])
            y = self.rng.uniform(sampling.y[0], sampling.y[1])
            if y <= self.config.task.ground_y:
                continue
            radius = float(np.hypot(x, y))
            if radius_min <= radius <= radius_max:
                return np.array([x, y], dtype=float)
        raise RuntimeError("Failed to sample a valid target position.")

    def _validate_target(self, target: Sequence[float]) -> np.ndarray:
        target_pos = np.asarray(target, dtype=float)
        if target_pos.shape != (2,):
            raise ValueError("target must be a 2D position.")
        if target_pos[1] <= self.config.task.ground_y:
            raise ValueError("target must satisfy y > ground_y.")
        return target_pos

    def _build_observation(self) -> dict[str, Any]:
        if self.state is None:
            raise RuntimeError("Environment is not initialized.")
        return {
            "q": self.state.q.copy(),
            "qd": self.state.qd.copy(),
            "qdd": self.state.qdd.copy(),
            "end_effector_pos": self.state.end_effector_pos.copy(),
            "end_effector_vel": self.state.end_effector_vel.copy(),
            "target_pos": self.state.target_pos.copy(),
            "distance_to_target": float(self.state.distance_to_target),
            "joint_torques": self.state.joint_torques.copy(),
            "joint_power": self.state.joint_power.copy(),
            "applied_action_norm": self.state.applied_action_norm.copy(),
            "consecutive_success_steps": int(self.state.consecutive_success_steps),
            "hold_progress": float(self.state.hold_progress),
            "success_ready": bool(self.state.success_ready),
        }

    def _record_history(self, reward: float, terminated: bool, truncated: bool) -> None:
        if self.state is None:
            return
        self.history.append(
            {
                "step": self.state.step_count,
                "q": self.state.q.copy(),
                "qd": self.state.qd.copy(),
                "qdd": self.state.qdd.copy(),
                "end_effector_pos": self.state.end_effector_pos.copy(),
                "end_effector_vel": self.state.end_effector_vel.copy(),
                "target_pos": self.state.target_pos.copy(),
                "distance_to_target": float(self.state.distance_to_target),
                "joint_torques": self.state.joint_torques.copy(),
                "joint_power": self.state.joint_power.copy(),
                "applied_action": self.state.applied_action.copy(),
                "applied_action_norm": self.state.applied_action_norm.copy(),
                "consecutive_success_steps": int(self.state.consecutive_success_steps),
                "hold_progress": float(self.state.hold_progress),
                "success_ready": bool(self.state.success_ready),
                "reward": float(reward),
                "terminated": bool(terminated),
                "truncated": bool(truncated),
            }
        )

    @property
    def workspace_radius(self) -> float:
        return total_reach(self.config.robot.link_lengths)
