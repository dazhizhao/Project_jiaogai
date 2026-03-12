import numpy as np

from env.bridge_robot_env import (
    BridgeRobotEnv,
    EnvConfig,
    RewardConfig,
    RobotConfig,
    RobotState,
    SimConfig,
    TaskConfig,
)
from env.dynamics import compute_equivalent_inertia, compute_gravity_torques
from env.kinematics import forward_kinematics


def make_static_env(*, success_hold_steps: int = 3, success_tolerance: float = 1e-6, max_steps: int = 10):
    env = BridgeRobotEnv()
    config = EnvConfig(
        sim=SimConfig(
            dt=env.config.sim.dt,
            max_steps=max_steps,
            gravity=0.0,
            integrator=env.config.sim.integrator,
        ),
        robot=env.config.robot,
        task=TaskConfig(
            home_pose=env.config.task.home_pose,
            target_sampling=env.config.task.target_sampling,
            ground_y=env.config.task.ground_y,
            success_tolerance=success_tolerance,
            success_hold_steps=success_hold_steps,
        ),
        reward=RewardConfig(
            distance_weight=env.config.reward.distance_weight,
            torque_weight=env.config.reward.torque_weight,
            motion_weight=env.config.reward.motion_weight,
            smoothness_weight=env.config.reward.smoothness_weight,
            ground_contact_penalty=env.config.reward.ground_contact_penalty,
            hold_bonus_weight=env.config.reward.hold_bonus_weight,
            success_bonus=env.config.reward.success_bonus,
        ),
        render=env.config.render,
        io=env.config.io,
    )
    return BridgeRobotEnv(config=config)


def test_reset_seed_reproduces_target():
    env_a = BridgeRobotEnv()
    env_b = BridgeRobotEnv()

    obs_a = env_a.reset(seed=11)
    obs_b = env_b.reset(seed=11)

    assert np.allclose(obs_a["target_pos"], obs_b["target_pos"])


def test_reset_samples_target_above_ground():
    env = BridgeRobotEnv()

    ys = [float(env.reset(seed=seed)["target_pos"][1]) for seed in range(20)]

    assert all(0.8 <= y <= 2.0 for y in ys)


def test_reset_rejects_target_on_or_below_ground():
    env = BridgeRobotEnv()

    with np.testing.assert_raises(ValueError):
        env.reset(target=[1.5, 0.0])

    observation = env.reset(target=[1.5, 0.9])
    assert np.allclose(observation["target_pos"], np.array([1.5, 0.9]))


def test_step_returns_expected_fields():
    env = BridgeRobotEnv()
    observation = env.reset(seed=3)
    assert set(observation.keys()) == {
        "q",
        "qd",
        "qdd",
        "end_effector_pos",
        "end_effector_vel",
        "target_pos",
        "distance_to_target",
        "joint_torques",
        "joint_power",
        "applied_action_norm",
        "consecutive_success_steps",
        "hold_progress",
        "success_ready",
    }

    result = env.step(np.zeros(4, dtype=float))
    assert set(result.info.keys()) >= {
        "reward_terms",
        "action_clipped",
        "success",
        "success_ready",
        "ground_contact",
        "joint_limit_violation",
        "workspace_violation",
        "consecutive_success_steps",
        "hold_progress",
        "applied_action_norm",
    }


def test_terminated_only_after_hold_steps():
    env = make_static_env(success_hold_steps=3, success_tolerance=1e-6)
    obs = env.reset(seed=5)
    env.state.target_pos = obs["end_effector_pos"].copy()

    result_a = env.step(np.zeros(4, dtype=float))
    result_b = env.step(np.zeros(4, dtype=float))
    result_c = env.step(np.zeros(4, dtype=float))

    assert result_a.terminated is False
    assert result_b.terminated is False
    assert result_c.terminated is True
    assert result_c.info["consecutive_success_steps"] == 3
    assert np.isclose(result_c.info["hold_progress"], 1.0)


def test_hold_counter_resets_when_leaving_tolerance_ball():
    env = make_static_env(success_hold_steps=3, success_tolerance=1e-9)
    obs = env.reset(seed=5)
    env.state.target_pos = obs["end_effector_pos"].copy()

    inside_result = env.step(np.zeros(4, dtype=float))
    outside_result = env.step(np.array([env.config.robot.torque_limits[0], 0.0, 0.0, 0.0], dtype=float))

    assert inside_result.info["consecutive_success_steps"] == 1
    assert outside_result.info["consecutive_success_steps"] == 0
    assert outside_result.info["success_ready"] is False


def test_truncated_at_max_steps_without_false_success():
    env = make_static_env(success_hold_steps=5, max_steps=1)
    env.reset(seed=5)
    result = env.step(np.zeros(4, dtype=float))

    assert result.truncated is True
    assert result.terminated is False


def test_reward_breakdown_uses_motion_and_hold_not_power():
    env = make_static_env(success_hold_steps=2, success_tolerance=1e-6)
    obs = env.reset(seed=5)
    env.state.target_pos = obs["end_effector_pos"].copy()
    result = env.step(np.zeros(4, dtype=float))

    reward_terms = result.info["reward_terms"]
    assert "motion_penalty" in reward_terms
    assert "hold_bonus" in reward_terms
    assert "power_penalty" not in reward_terms


def test_ground_contact_rejects_invalid_step_and_zeros_motion():
    env = make_static_env()
    obs = env.reset(seed=5)
    q = np.array([0.05, 0.4, 0.2, 0.0], dtype=float)
    kin = forward_kinematics(q, env.config.robot.link_lengths)
    gravity_torques = compute_gravity_torques(
        joint_angles=q,
        link_lengths=env.config.robot.link_lengths,
        link_masses=env.config.robot.link_masses,
        payload_mass=env.config.robot.payload_mass,
        gravity=env.config.sim.gravity,
    )
    equivalent_inertia = compute_equivalent_inertia(
        joint_angles=q,
        link_lengths=env.config.robot.link_lengths,
        link_masses=env.config.robot.link_masses,
        payload_mass=env.config.robot.payload_mass,
    )
    env.state = RobotState(
        q=q,
        qd=np.array([-10.0, 0.0, 0.0, 0.0], dtype=float),
        qdd=np.zeros(4, dtype=float),
        joint_positions=kin.joint_positions,
        end_effector_pos=kin.end_effector_pos,
        end_effector_vel=np.zeros(2, dtype=float),
        end_effector_acc=np.zeros(2, dtype=float),
        joint_torques=gravity_torques,
        joint_power=np.zeros(4, dtype=float),
        step_count=env.state.step_count,
        applied_action=np.zeros(4, dtype=float),
        applied_action_norm=np.zeros(4, dtype=float),
        target_pos=obs["target_pos"].copy(),
        distance_to_target=float(np.linalg.norm(kin.end_effector_pos - obs["target_pos"])),
        gravity_torques=gravity_torques,
        equivalent_inertia=equivalent_inertia,
        consecutive_success_steps=0,
        hold_progress=0.0,
        success_ready=False,
    )
    previous_q = env.state.q.copy()
    previous_joint_positions = env.state.joint_positions.copy()
    previous_ee = env.state.end_effector_pos.copy()

    result = env.step(np.zeros(4, dtype=float))

    assert result.info["ground_contact"] is True
    assert result.info["workspace_violation"] is True
    assert result.info["joint_limit_violation"] is False
    assert result.terminated is True
    assert result.truncated is False
    assert result.info["reward_terms"]["ground_contact_penalty"] == -env.config.reward.ground_contact_penalty
    assert result.info["reward_terms"]["motion_penalty"] == 0.0
    assert np.allclose(env.state.q, previous_q)
    assert np.allclose(env.state.joint_positions, previous_joint_positions)
    assert np.allclose(env.state.end_effector_pos, previous_ee)
    assert np.allclose(env.state.qd, np.zeros(4, dtype=float))
    assert np.allclose(env.state.qdd, np.zeros(4, dtype=float))
    assert np.allclose(env.state.end_effector_vel, np.zeros(2, dtype=float))
    assert np.allclose(env.state.end_effector_acc, np.zeros(2, dtype=float))


def test_ground_contact_terminates_even_at_max_steps():
    env = make_static_env(max_steps=1)
    obs = env.reset(seed=5)
    q = np.array([0.05, 0.4, 0.2, 0.0], dtype=float)
    kin = forward_kinematics(q, env.config.robot.link_lengths)
    gravity_torques = compute_gravity_torques(
        joint_angles=q,
        link_lengths=env.config.robot.link_lengths,
        link_masses=env.config.robot.link_masses,
        payload_mass=env.config.robot.payload_mass,
        gravity=env.config.sim.gravity,
    )
    equivalent_inertia = compute_equivalent_inertia(
        joint_angles=q,
        link_lengths=env.config.robot.link_lengths,
        link_masses=env.config.robot.link_masses,
        payload_mass=env.config.robot.payload_mass,
    )
    env.state = RobotState(
        q=q,
        qd=np.array([-10.0, 0.0, 0.0, 0.0], dtype=float),
        qdd=np.zeros(4, dtype=float),
        joint_positions=kin.joint_positions,
        end_effector_pos=kin.end_effector_pos,
        end_effector_vel=np.zeros(2, dtype=float),
        end_effector_acc=np.zeros(2, dtype=float),
        joint_torques=gravity_torques,
        joint_power=np.zeros(4, dtype=float),
        step_count=0,
        applied_action=np.zeros(4, dtype=float),
        applied_action_norm=np.zeros(4, dtype=float),
        target_pos=obs["target_pos"].copy(),
        distance_to_target=float(np.linalg.norm(kin.end_effector_pos - obs["target_pos"])),
        gravity_torques=gravity_torques,
        equivalent_inertia=equivalent_inertia,
        consecutive_success_steps=0,
        hold_progress=0.0,
        success_ready=False,
    )

    result = env.step(np.zeros(4, dtype=float))

    assert result.terminated is True
    assert result.truncated is False


def test_joint_limit_violation_sets_workspace_violation():
    env = BridgeRobotEnv()
    tight_limits = [limit[:] for limit in env.config.robot.joint_limits]
    tight_limits[0] = [-0.1, 0.1]
    config = EnvConfig(
        sim=SimConfig(
            dt=env.config.sim.dt,
            max_steps=env.config.sim.max_steps,
            gravity=0.0,
            integrator=env.config.sim.integrator,
        ),
        robot=RobotConfig(
            link_lengths=env.config.robot.link_lengths,
            link_masses=env.config.robot.link_masses,
            payload_mass=env.config.robot.payload_mass,
            joint_damping=env.config.robot.joint_damping,
            joint_limits=tight_limits,
            torque_limits=env.config.robot.torque_limits,
        ),
        task=TaskConfig(
            home_pose=env.config.task.home_pose,
            target_sampling=env.config.task.target_sampling,
            ground_y=env.config.task.ground_y,
            success_tolerance=env.config.task.success_tolerance,
            success_hold_steps=env.config.task.success_hold_steps,
        ),
        reward=env.config.reward,
        render=env.config.render,
        io=env.config.io,
    )
    constrained_env = BridgeRobotEnv(config=config)
    obs = constrained_env.reset(seed=5)
    q = np.array([0.09, 0.4, 0.2, 0.0], dtype=float)
    kin = forward_kinematics(q, constrained_env.config.robot.link_lengths)
    gravity_torques = compute_gravity_torques(
        joint_angles=q,
        link_lengths=constrained_env.config.robot.link_lengths,
        link_masses=constrained_env.config.robot.link_masses,
        payload_mass=constrained_env.config.robot.payload_mass,
        gravity=constrained_env.config.sim.gravity,
    )
    equivalent_inertia = compute_equivalent_inertia(
        joint_angles=q,
        link_lengths=constrained_env.config.robot.link_lengths,
        link_masses=constrained_env.config.robot.link_masses,
        payload_mass=constrained_env.config.robot.payload_mass,
    )
    constrained_env.state = RobotState(
        q=q,
        qd=np.array([10.0, 0.0, 0.0, 0.0], dtype=float),
        qdd=np.zeros(4, dtype=float),
        joint_positions=kin.joint_positions,
        end_effector_pos=kin.end_effector_pos,
        end_effector_vel=np.zeros(2, dtype=float),
        end_effector_acc=np.zeros(2, dtype=float),
        joint_torques=gravity_torques,
        joint_power=np.zeros(4, dtype=float),
        step_count=constrained_env.state.step_count,
        applied_action=np.zeros(4, dtype=float),
        applied_action_norm=np.zeros(4, dtype=float),
        target_pos=obs["target_pos"].copy(),
        distance_to_target=float(np.linalg.norm(kin.end_effector_pos - obs["target_pos"])),
        gravity_torques=gravity_torques,
        equivalent_inertia=equivalent_inertia,
        consecutive_success_steps=0,
        hold_progress=0.0,
        success_ready=False,
    )

    result = constrained_env.step(np.zeros(4, dtype=float))

    assert result.info["joint_limit_violation"] is True
    assert result.info["ground_contact"] is False
    assert result.info["workspace_violation"] is True
