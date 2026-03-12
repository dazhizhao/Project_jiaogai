# Four-Link Reinforcement Learning for Torque Control

## Overview

This repository studies torque control for a four-link manipulator with reinforcement learning. A mechanics-based simulator is used directly as the environment, and SAC is trained to drive the end effector to a target region under dynamic constraints.

The repository also includes a simpler static design direction based on link-length allocation, aimed at enlarging the maximum reachable workspace.

## Project Motivation

This project started from a personal interest in reinforcement learning after watching DeepMind's *The Thinking Game*. I wanted to try the RL workflow myself, especially the idea of training directly through interaction with an environment instead of relying on an offline dataset.

That motivation led to a simple pipeline:

1. Build a physical model from theoretical mechanics.
2. Use that model as the reinforcement learning environment.
3. Train a policy and compare its behavior across optimization stages.

## Method

The simulator models a four-link manipulator with explicit kinematics and dynamics. The RL task uses a 4D normalized torque action, and training is performed with SAC using `MlpPolicy`. A rollout is counted as successful only when the end effector reaches the tolerance region and stays there for several consecutive steps.

## Environment and Learning Setup

The main task uses the following settings.

- 4-DoF continuous torque control
- SAC with `MlpPolicy`
- Simulation time step `dt = 0.02`
- Maximum episode length `250` steps
- Success tolerance `0.08`
- Success hold requirement `5` consecutive steps
- Target region sampled in the upper half-plane
- Python `3.10`, with dependencies defined in [`requirements.yaml`](requirements.yaml)

The training entry point is [`scripts/train_rl.py`](scripts/train_rl.py), and the default configuration is defined in [`configs/train_rl.yaml`](configs/train_rl.yaml) and [`configs/default.yaml`](configs/default.yaml).

## Training Progress and Results

The overall trend is clear: early training mainly improves whether the manipulator can reach the target region, while later training improves how quickly it can do so.

### Stage 50k

![Stage 50k rollout](docs/media/stage-050k.gif)

At 50k steps, the policy is still far from stable target reaching.

### Stage 100k

![Stage 100k rollout](docs/media/stage-100k.gif)

At 100k steps, the controller moves much closer to the target, but it is still not reliably successful.

### Stage 200k

![Stage 200k rollout](docs/media/stage-200k.gif)

At 200k steps, the behavior is close to the success threshold, showing strong improvement in target-reaching quality.

### Best Policy

![Best policy rollout](docs/media/best-policy.gif)

The final best policy is the canonical result of this run. It demonstrates both stable reaching and improved efficiency.

![Best joint torques](docs/media/best_joint_torques.png)

The final torque profile shows how the learned controller distributes effort across the four joints during a successful rollout.

## How to Run

Create the environment and install dependencies:

```bash
conda env create -f requirements.yaml
conda activate bridge-robot-cloud
```

Train the torque-control policy:

```bash
python scripts/train_rl.py
```

Run the test suite:

```bash
python -m pytest tests
```

The original workflow used local development and cloud-based training.

## Conclusion

This project presents a compact pipeline for four-link manipulator control with reinforcement learning. The experiments show a clear two-stage trend: early optimization improves target reaching, and later optimization improves speed. The final policy not only reaches the target region more reliably, but does so in fewer steps.
