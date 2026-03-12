from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
from typing import Any, Sequence

import gymnasium as gym
from gymnasium import spaces
import numpy as np

try:
    import yaml
except ImportError:  # pragma: no cover - exercised only without PyYAML
    yaml = None


@dataclass(frozen=True)
class WorkspaceSamplingConfig:
    num_samples: int
    seed: int
    grid_size: list[int]
    xy_bounds: list[list[float]]


@dataclass(frozen=True)
class WorkspaceVideoConfig:
    fps: int
    frames: int
    point_size: float
    alpha: float


@dataclass(frozen=True)
class LinkAllocationConfig:
    total_length: float
    default_link_lengths: list[float]
    min_link_lengths: list[float]
    max_link_lengths: list[float]
    joint_angle_limits: list[list[float]]
    workspace_sampling: WorkspaceSamplingConfig
    video: WorkspaceVideoConfig

    @classmethod
    def load(cls, config_path: str | Path | None = None) -> "LinkAllocationConfig":
        path = (
            Path(config_path)
            if config_path
            else Path(__file__).resolve().parents[1] / "configs" / "link_allocation_env.yaml"
        )
        text = path.read_text(encoding="utf-8")
        raw = yaml.safe_load(text) if yaml is not None else json.loads(text)
        return cls(
            total_length=raw["total_length"],
            default_link_lengths=raw["default_link_lengths"],
            min_link_lengths=raw["min_link_lengths"],
            max_link_lengths=raw["max_link_lengths"],
            joint_angle_limits=raw["joint_angle_limits"],
            workspace_sampling=WorkspaceSamplingConfig(**raw["workspace_sampling"]),
            video=WorkspaceVideoConfig(**raw["video"]),
        )

    def validate(self) -> None:
        default = np.asarray(self.default_link_lengths, dtype=float)
        lower = np.asarray(self.min_link_lengths, dtype=float)
        upper = np.asarray(self.max_link_lengths, dtype=float)
        joint_limits = np.asarray(self.joint_angle_limits, dtype=float)
        grid_size = np.asarray(self.workspace_sampling.grid_size, dtype=int)
        xy_bounds = np.asarray(self.workspace_sampling.xy_bounds, dtype=float)

        if default.shape != (4,) or lower.shape != (4,) or upper.shape != (4,):
            raise ValueError("Link allocation config expects exactly four link lengths.")
        if joint_limits.shape != (4, 2):
            raise ValueError("joint_angle_limits must have shape (4, 2).")
        if grid_size.shape != (2,) or np.any(grid_size <= 0):
            raise ValueError("workspace_sampling.grid_size must contain two positive integers.")
        if xy_bounds.shape != (2, 2):
            raise ValueError("workspace_sampling.xy_bounds must have shape (2, 2).")
        if np.any(joint_limits[:, 0] >= joint_limits[:, 1]):
            raise ValueError("Each joint angle limit must satisfy lower < upper.")
        if not np.all(lower <= upper):
            raise ValueError("min_link_lengths must be less than or equal to max_link_lengths.")
        if self.total_length <= 0.0:
            raise ValueError("total_length must be positive.")
        if float(np.sum(lower)) > self.total_length or float(np.sum(upper)) < self.total_length:
            raise ValueError("total_length must lie within the feasible bounded-simplex range.")
        if np.any(default < lower) or np.any(default > upper):
            raise ValueError("default_link_lengths must satisfy per-link bounds.")
        if not np.isclose(float(np.sum(default)), self.total_length):
            raise ValueError("default_link_lengths must sum to total_length.")
        if xy_bounds[0, 0] >= xy_bounds[0, 1] or xy_bounds[1, 0] >= xy_bounds[1, 1]:
            raise ValueError("workspace_sampling.xy_bounds must be strictly increasing on both axes.")
        if self.workspace_sampling.num_samples <= 0:
            raise ValueError("workspace_sampling.num_samples must be positive.")
        if self.video.fps <= 0 or self.video.frames <= 0:
            raise ValueError("video.fps and video.frames must be positive.")
        if self.video.point_size <= 0.0 or not 0.0 < self.video.alpha <= 1.0:
            raise ValueError("video.point_size must be positive and video.alpha must be in (0, 1].")


@dataclass(frozen=True)
class WorkspaceMetrics:
    reward: float
    occupied_ratio: float
    workspace_area_estimate: float
    workspace_points: np.ndarray
    joint_angle_samples: np.ndarray
    representative_joint_angles: np.ndarray
    grid_shape: tuple[int, int]
    xy_bounds: np.ndarray


class LinkAllocationEnv(gym.Env[np.ndarray, np.ndarray]):
    metadata = {"render_modes": []}

    def __init__(self, config: LinkAllocationConfig | None = None, config_path: str | Path | None = None) -> None:
        self.config = config if config is not None else LinkAllocationConfig.load(config_path)
        self.config.validate()

        self._default = np.asarray(self.config.default_link_lengths, dtype=np.float32)
        self._lower = np.asarray(self.config.min_link_lengths, dtype=np.float64)
        self._upper = np.asarray(self.config.max_link_lengths, dtype=np.float64)
        self._joint_angle_limits = np.asarray(self.config.joint_angle_limits, dtype=np.float64)
        self._total_length = float(self.config.total_length)
        self._grid_shape = tuple(int(value) for value in self.config.workspace_sampling.grid_size)
        self._xy_bounds = np.asarray(self.config.workspace_sampling.xy_bounds, dtype=np.float64)
        self._observation = self._build_observation()
        self.action_space = spaces.Box(
            low=np.asarray(self.config.min_link_lengths, dtype=np.float32),
            high=np.asarray(self.config.max_link_lengths, dtype=np.float32),
            shape=(4,),
            dtype=np.float32,
        )
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(13,),
            dtype=np.float32,
        )
        self.last_allocated_lengths = self._default.copy()
        self.last_metrics: WorkspaceMetrics | None = None
        self._episode_active = False

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        self.last_allocated_lengths = self._default.copy()
        self.last_metrics = None
        self._episode_active = True
        return self._observation.copy(), {
            "allocated_lengths": self.last_allocated_lengths.copy(),
            "projection_applied": False,
        }

    def step(self, action: Sequence[float]) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        if not self._episode_active:
            raise RuntimeError("Call reset() before step().")

        raw_action = np.asarray(action, dtype=np.float64)
        if raw_action.shape != (4,):
            raise ValueError("LinkAllocationEnv expects a 4D action.")

        allocated = project_bounded_simplex(
            raw_action,
            target_sum=self._total_length,
            lower=self._lower,
            upper=self._upper,
        ).astype(np.float32)
        metrics = evaluate_workspace(
            lengths=allocated,
            joint_angle_limits=self._joint_angle_limits,
            num_samples=self.config.workspace_sampling.num_samples,
            seed=self.config.workspace_sampling.seed,
            grid_shape=self._grid_shape,
            xy_bounds=self._xy_bounds,
        )
        projection_applied = not np.allclose(allocated, raw_action, atol=1e-6, rtol=1e-6)

        self.last_allocated_lengths = allocated
        self.last_metrics = metrics
        self._episode_active = False

        info = {
            "allocated_lengths": allocated.copy(),
            "raw_action": raw_action.astype(np.float32),
            "projection_applied": projection_applied,
            "workspace_points": metrics.workspace_points.copy(),
            "occupied_ratio": float(metrics.occupied_ratio),
            "workspace_area_estimate": float(metrics.workspace_area_estimate),
            "grid_shape": list(metrics.grid_shape),
            "xy_bounds": metrics.xy_bounds.copy(),
            "joint_angle_samples": metrics.joint_angle_samples.copy(),
            "representative_joint_angles": metrics.representative_joint_angles.copy(),
        }
        return self._observation.copy(), float(metrics.reward), True, False, info

    def evaluate_lengths(self, lengths: Sequence[float]) -> WorkspaceMetrics:
        return evaluate_workspace(
            lengths=lengths,
            joint_angle_limits=self._joint_angle_limits,
            num_samples=self.config.workspace_sampling.num_samples,
            seed=self.config.workspace_sampling.seed,
            grid_shape=self._grid_shape,
            xy_bounds=self._xy_bounds,
        )

    def _build_observation(self) -> np.ndarray:
        total = np.float32(self._total_length)
        return np.concatenate(
            [
                self._default / total,
                self._lower.astype(np.float32) / total,
                self._upper.astype(np.float32) / total,
                np.array([total], dtype=np.float32),
            ]
        ).astype(np.float32)


def evaluate_workspace(
    lengths: Sequence[float],
    joint_angle_limits: Sequence[Sequence[float]],
    num_samples: int,
    seed: int,
    grid_shape: Sequence[int],
    xy_bounds: Sequence[Sequence[float]],
) -> WorkspaceMetrics:
    lengths_arr = np.asarray(lengths, dtype=np.float64)
    joint_limits_arr = np.asarray(joint_angle_limits, dtype=np.float64)
    xy_bounds_arr = np.asarray(xy_bounds, dtype=np.float64)
    grid_shape_tuple = (int(grid_shape[0]), int(grid_shape[1]))

    rng = np.random.default_rng(seed)
    sampled_angles = rng.uniform(
        low=joint_limits_arr[:, 0],
        high=joint_limits_arr[:, 1],
        size=(num_samples, lengths_arr.shape[0]),
    )
    points = _sample_end_effector_points(sampled_angles, lengths_arr)
    occupancy = _build_occupancy_mask(points, grid_shape_tuple, xy_bounds_arr)
    occupied_ratio = float(np.mean(occupancy))
    bounds_area = float(
        (xy_bounds_arr[0, 1] - xy_bounds_arr[0, 0]) * (xy_bounds_arr[1, 1] - xy_bounds_arr[1, 0])
    )
    workspace_area_estimate = occupied_ratio * bounds_area
    representative_index = int(np.argmax(np.linalg.norm(points, axis=1)))

    return WorkspaceMetrics(
        reward=occupied_ratio,
        occupied_ratio=occupied_ratio,
        workspace_area_estimate=workspace_area_estimate,
        workspace_points=points.astype(np.float32),
        joint_angle_samples=sampled_angles.astype(np.float32),
        representative_joint_angles=sampled_angles[representative_index].astype(np.float32),
        grid_shape=grid_shape_tuple,
        xy_bounds=xy_bounds_arr.astype(np.float32),
    )


def _sample_end_effector_points(joint_angles: np.ndarray, lengths: np.ndarray) -> np.ndarray:
    x_coords = np.sum(np.cos(joint_angles) * lengths, axis=1)
    y_coords = np.sum(np.sin(joint_angles) * lengths, axis=1)
    return np.stack([x_coords, y_coords], axis=1)


def _build_occupancy_mask(
    points: np.ndarray,
    grid_shape: tuple[int, int],
    xy_bounds: np.ndarray,
) -> np.ndarray:
    width = xy_bounds[0, 1] - xy_bounds[0, 0]
    height = xy_bounds[1, 1] - xy_bounds[1, 0]
    x_min = xy_bounds[0, 0]
    y_min = xy_bounds[1, 0]
    rows, cols = grid_shape

    occupancy = np.zeros((rows, cols), dtype=bool)
    in_bounds = (
        (points[:, 0] >= xy_bounds[0, 0])
        & (points[:, 0] <= xy_bounds[0, 1])
        & (points[:, 1] >= xy_bounds[1, 0])
        & (points[:, 1] <= xy_bounds[1, 1])
    )
    if not np.any(in_bounds):
        return occupancy

    clipped = points[in_bounds]
    x_idx = np.floor((clipped[:, 0] - x_min) / width * cols).astype(int)
    y_idx = np.floor((clipped[:, 1] - y_min) / height * rows).astype(int)
    x_idx = np.clip(x_idx, 0, cols - 1)
    y_idx = np.clip(y_idx, 0, rows - 1)
    occupancy[y_idx, x_idx] = True
    return occupancy


def project_bounded_simplex(
    values: Sequence[float],
    target_sum: float,
    lower: Sequence[float],
    upper: Sequence[float],
    tolerance: float = 1e-9,
    max_iterations: int = 200,
) -> np.ndarray:
    raw = np.asarray(values, dtype=np.float64)
    lower_arr = np.asarray(lower, dtype=np.float64)
    upper_arr = np.asarray(upper, dtype=np.float64)

    if raw.shape != lower_arr.shape or raw.shape != upper_arr.shape:
        raise ValueError("values, lower, and upper must share the same shape.")
    if float(np.sum(lower_arr)) > target_sum or float(np.sum(upper_arr)) < target_sum:
        raise ValueError("target_sum is infeasible for the provided bounds.")

    if np.all(raw >= lower_arr) and np.all(raw <= upper_arr) and np.isclose(np.sum(raw), target_sum):
        return raw.copy()

    left = float(np.min(raw - upper_arr))
    right = float(np.max(raw - lower_arr))
    for _ in range(max_iterations):
        midpoint = 0.5 * (left + right)
        projected = np.clip(raw - midpoint, lower_arr, upper_arr)
        error = float(np.sum(projected) - target_sum)
        if abs(error) <= tolerance:
            return projected
        if error > 0.0:
            left = midpoint
        else:
            right = midpoint

    projected = np.clip(raw - 0.5 * (left + right), lower_arr, upper_arr)
    residual = target_sum - float(np.sum(projected))
    if abs(residual) > 1e-6:
        free_mask = (projected > lower_arr + 1e-8) & (projected < upper_arr - 1e-8)
        if np.any(free_mask):
            projected = projected.copy()
            projected[free_mask] += residual / float(np.sum(free_mask))
            projected = np.clip(projected, lower_arr, upper_arr)
    return projected
