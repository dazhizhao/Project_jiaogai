from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("gymnasium")

from env.link_allocation_env import LinkAllocationEnv, project_bounded_simplex


def test_reset_returns_expected_observation_shape_and_dtype():
    env = LinkAllocationEnv()
    observation, info = env.reset(seed=7)

    assert observation.shape == (13,)
    assert observation.dtype == np.float32
    assert np.allclose(info["allocated_lengths"], np.array([1.2, 1.0, 0.8, 0.6], dtype=np.float32))


def test_step_terminates_immediately_without_truncation():
    env = LinkAllocationEnv()
    env.reset(seed=7)
    _, _, terminated, truncated, _ = env.step([0.9, 0.9, 0.9, 0.9])

    assert terminated is True
    assert truncated is False


def test_projected_lengths_respect_sum_and_bounds():
    projected = project_bounded_simplex(
        values=[1.4, 1.4, 1.4, 1.4],
        target_sum=3.6,
        lower=[0.4, 0.4, 0.4, 0.4],
        upper=[1.4, 1.4, 1.4, 1.4],
    )

    assert np.isclose(np.sum(projected), 3.6)
    assert np.all(projected >= 0.4 - 1e-8)
    assert np.all(projected <= 1.4 + 1e-8)


def test_different_allocations_produce_different_occupied_ratios():
    env = LinkAllocationEnv()
    env.reset(seed=7)
    _, balanced_reward, _, _, balanced_info = env.step([0.9, 0.9, 0.9, 0.9])

    env.reset(seed=7)
    _, unbalanced_reward, _, _, unbalanced_info = env.step([1.4, 1.4, 0.4, 0.4])

    assert not np.isclose(balanced_reward, unbalanced_reward)
    assert not np.isclose(balanced_info["occupied_ratio"], unbalanced_info["occupied_ratio"])


def test_same_input_is_deterministic_under_fixed_sampling_seed():
    env = LinkAllocationEnv()
    env.reset(seed=7)
    _, reward_a, _, _, info_a = env.step([1.2, 0.8, 0.8, 0.8])

    env.reset(seed=99)
    _, reward_b, _, _, info_b = env.step([1.2, 0.8, 0.8, 0.8])

    assert np.isclose(reward_a, reward_b)
    assert np.isclose(info_a["occupied_ratio"], info_b["occupied_ratio"])
    assert np.allclose(info_a["workspace_points"], info_b["workspace_points"])


def test_step_exposes_workspace_shapes_and_bounds():
    env = LinkAllocationEnv()
    env.reset(seed=7)
    _, _, _, _, info = env.step([1.0, 1.0, 0.8, 0.8])

    assert info["workspace_points"].shape == (env.config.workspace_sampling.num_samples, 2)
    assert info["joint_angle_samples"].shape == (env.config.workspace_sampling.num_samples, 4)
    assert info["grid_shape"] == env.config.workspace_sampling.grid_size
    assert np.allclose(info["xy_bounds"], np.asarray(env.config.workspace_sampling.xy_bounds, dtype=np.float32))
