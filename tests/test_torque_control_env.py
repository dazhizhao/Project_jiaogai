from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("gymnasium")

from env.bridge_robot_env import EnvConfig, RewardConfig, SimConfig, TaskConfig
from env.torque_control_env import TorqueControlEnv


def make_env() -> TorqueControlEnv:
    base_env = TorqueControlEnv()
    config = EnvConfig(
        sim=SimConfig(
            dt=base_env.config.sim.dt,
            max_steps=base_env.config.sim.max_steps,
            gravity=0.0,
            integrator=base_env.config.sim.integrator,
        ),
        robot=base_env.config.robot,
        task=TaskConfig(
            home_pose=base_env.config.task.home_pose,
            target_sampling=base_env.config.task.target_sampling,
            ground_y=base_env.config.task.ground_y,
            success_tolerance=base_env.config.task.success_tolerance,
            success_hold_steps=base_env.config.task.success_hold_steps,
        ),
        reward=RewardConfig(
            distance_weight=base_env.config.reward.distance_weight,
            torque_weight=base_env.config.reward.torque_weight,
            motion_weight=base_env.config.reward.motion_weight,
            smoothness_weight=base_env.config.reward.smoothness_weight,
            ground_contact_penalty=base_env.config.reward.ground_contact_penalty,
            hold_bonus_weight=base_env.config.reward.hold_bonus_weight,
            success_bonus=base_env.config.reward.success_bonus,
        ),
        render=base_env.config.render,
        io=base_env.config.io,
    )
    return TorqueControlEnv(config=config)


def test_reset_returns_flat_observation():
    env = make_env()
    observation, _ = env.reset(seed=7)

    assert observation.shape == (18,)
    assert observation.dtype == np.float32


def test_step_scales_normalized_action_to_torque_limits():
    env = make_env()
    env.reset(seed=7)
    _, _, _, _, info = env.step(np.array([1.0, -1.0, 0.5, 0.0], dtype=np.float32))

    expected = np.array(env.config.robot.torque_limits, dtype=np.float32) * np.array([1.0, -1.0, 0.5, 0.0])
    assert np.allclose(info["applied_action"], expected)
    assert np.allclose(info["action_norm"], np.array([1.0, -1.0, 0.5, 0.0], dtype=np.float32))


def test_observation_tracks_previous_action_and_hold_progress():
    env = make_env()
    observation, _ = env.reset(seed=7)
    assert np.allclose(observation[-5:-1], np.zeros(4, dtype=np.float32))
    assert observation[-1] == 0.0

    next_observation, _, _, _, _ = env.step(np.array([0.25, -0.5, 0.75, 0.0], dtype=np.float32))
    assert np.allclose(next_observation[-5:-1], np.array([0.25, -0.5, 0.75, 0.0], dtype=np.float32))
