from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class RewardBreakdown:
    progress_reward: float
    torque_penalty: float
    motion_penalty: float
    smoothness_penalty: float
    ground_contact_penalty: float
    hold_bonus: float
    success_bonus: float

    @property
    def total(self) -> float:
        return (
            self.progress_reward
            + self.torque_penalty
            + self.motion_penalty
            + self.smoothness_penalty
            + self.ground_contact_penalty
            + self.hold_bonus
            + self.success_bonus
        )

    def to_dict(self) -> dict[str, float]:
        payload = asdict(self)
        payload["total"] = self.total
        return payload


def compute_reward(
    previous_distance: float,
    current_distance: float,
    action_normalized: Sequence[float],
    joint_velocities: Sequence[float],
    previous_action: Sequence[float],
    ground_contact: bool,
    hold_progress: float,
    success: bool,
    progress_weight: float,
    torque_weight: float,
    motion_weight: float,
    smoothness_weight: float,
    ground_contact_penalty: float,
    hold_bonus_weight: float,
    success_bonus: float,
) -> RewardBreakdown:
    action = np.asarray(action_normalized, dtype=float)
    joint_velocity = np.asarray(joint_velocities, dtype=float)
    prev_action = np.asarray(previous_action, dtype=float)
    progress = float(previous_distance) - float(current_distance)

    return RewardBreakdown(
        progress_reward=progress_weight * progress,
        torque_penalty=-torque_weight * float(np.sum(action**2)),
        motion_penalty=-motion_weight * float(np.sum(joint_velocity**2)),
        smoothness_penalty=-smoothness_weight * float(np.sum((action - prev_action) ** 2)),
        ground_contact_penalty=-ground_contact_penalty if ground_contact else 0.0,
        hold_bonus=hold_bonus_weight * float(hold_progress),
        success_bonus=success_bonus if success else 0.0,
    )
