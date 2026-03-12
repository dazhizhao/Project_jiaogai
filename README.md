# 桥梁检测机器人控制优化

当前仓库的默认强化学习主线已经切换为四关节连续力矩控制。目标是在机构参数、负载参数和关节力矩上限固定的前提下，学习一个 torque policy，让机械臂末端能够稳定进入目标容差球，并在满足到达要求的同时尽量减少力矩使用、抑制大动作和突变。

旧的杆长分配任务仍然保留，但已经降级为次线能力，不再是默认训练入口。

## 当前主线：Torque Control RL

主训练环境是 `BridgeRobotEnv` 的 RL 包装层 `TorqueControlEnv`：

- 动作：4 维连续归一化力矩 `Box(-1, 1, shape=(4,))`
- 内部执行：按各关节 `torque_limits` 缩放为物理力矩后进入 `BridgeRobotEnv.step(...)`
- 观测：18 维扁平连续向量，包含
  - 归一化关节角 `q`
  - 归一化关节角速度 `qd`
  - 归一化末端位置
  - 归一化目标相对位移
  - 归一化距离
  - 上一步归一化动作
  - `hold_progress`

### 稳定成功判据

默认配置在 [`configs/default.yaml`](/d:/0311_demo/Project_jiaogai/configs/default.yaml)：

- `success_tolerance = 0.08`
- `success_hold_steps = 10`

单步进入容差球不会立即成功。只有末端连续 `10` 个仿真步都满足 `distance_to_target <= success_tolerance`，回合才会以成功终止。中途一旦离开容差球，连续保持计数会清零。

### 奖励设计

当前 reward 由以下部分组成：

- `distance_reward = -distance_weight * distance_to_target`
- `torque_penalty = -torque_weight * sum((applied_torque / torque_limits)^2)`
- `motion_penalty = -motion_weight * sum(qd^2)`
- `smoothness_penalty = -smoothness_weight * sum((action_norm - prev_action_norm)^2)`
- `hold_bonus = hold_bonus_weight * hold_progress`
- `success_bonus`

`joint_power` 仍然会在状态、历史和可视化中保留，但不再作为优化目标进入 reward。

### 训练入口

默认训练脚本是 [`scripts/train_rl.py`](/d:/0311_demo/Project_jiaogai/scripts/train_rl.py)，算法默认仍为 SAC。

默认训练配置位于 [`configs/train_rl.yaml`](/d:/0311_demo/Project_jiaogai/configs/train_rl.yaml)。

训练命令：

```bash
python scripts/train_rl.py
```

覆盖参数示例：

```bash
python scripts/train_rl.py \
  --total-timesteps 5000 \
  --run-name exp001 \
  --output-dir /openbayes/home/results
```

### 训练输出

训练结果写入：

```text
<output_dir>/rl_torque_control/<run_name>_<timestamp>_<suffix>/
```

至少包含：

- `model_final.zip`
- `progress.csv`
- `summary.json`
- `monitor.csv`
- `artifacts/training_curves.png`
- `artifacts/best_rollout.npz`
- `artifacts/best_pose.png`
- `artifacts/best_rollout.mp4`

每次 run 会自动附加时间戳和短随机后缀，避免同名实验互相覆盖。

`summary.json` 会集中保存训练配置、解析后的实际输出目录和评估结果，其中评估部分会输出：

- `success_rate`
- `mean_reward`
- `mean_final_distance`
- `mean_episode_length`
- `mean_torque_norm`
- `mean_motion_penalty`
- `mean_smoothness_penalty`

最佳回合优先从成功回合中选择；若评估时没有成功回合，则退化为总回报最高回合。其余可视化和回放文件统一放在 `artifacts/` 子目录，避免根目录堆太多文件。

训练期间控制台输出已精简为少量开始/结束信息。更完整的训练过程曲线会保存到 `artifacts/training_curves.png`，其中会尽量展示 episode reward、episode length 以及 SB3 记录到 `progress.csv` 中的 actor/critic loss 等常见 RL 指标。
训练结束时也会明确打印 torque rollout 视频路径；同一路径也会写入 `summary.json -> evaluation -> artifact_paths -> rollout_video`。

## 物理环境与可视化

主任务底层物理核心仍是 `BridgeRobotEnv`，相关模块包括：

- [`env/bridge_robot_env.py`](/d:/0311_demo/Project_jiaogai/env/bridge_robot_env.py)
- [`env/dynamics.py`](/d:/0311_demo/Project_jiaogai/env/dynamics.py)
- [`env/kinematics.py`](/d:/0311_demo/Project_jiaogai/env/kinematics.py)
- [`env/reward.py`](/d:/0311_demo/Project_jiaogai/env/reward.py)
- [`visualization/render.py`](/d:/0311_demo/Project_jiaogai/visualization/render.py)
- [`visualization/plots.py`](/d:/0311_demo/Project_jiaogai/visualization/plots.py)
- [`visualization/video.py`](/d:/0311_demo/Project_jiaogai/visualization/video.py)

常用命令：

```bash
python scripts/run_env.py --policy zero --seed 7 --output-dir /openbayes/home/results
python scripts/run_env.py --policy random --seed 7 --output-dir /openbayes/home/results
python scripts/visualize_env.py
```

## 次线任务：Link Allocation RL

旧的杆长分配任务仍然可用，但不再是默认主线。

- 环境：[`env/link_allocation_env.py`](/d:/0311_demo/Project_jiaogai/env/link_allocation_env.py)
- 训练脚本：[`scripts/train_link_allocation.py`](/d:/0311_demo/Project_jiaogai/scripts/train_link_allocation.py)
- 配置：[`configs/train_link_allocation.yaml`](/d:/0311_demo/Project_jiaogai/configs/train_link_allocation.yaml)

训练命令：

```bash
python scripts/train_link_allocation.py
```

输出目录仍为：

```text
<output_dir>/rl_link_alloc/<run_name>_<timestamp>_<suffix>/
```

输出也统一整理为：

- `model_final.zip`
- `progress.csv`
- `summary.json`
- `monitor.csv`
- `artifacts/training_curves.png`
- `artifacts/best_workspace_samples.npz`
- `artifacts/best_workspace.png`
- `artifacts/best_workspace.mp4`

## 仓库结构

```text
project/
├── README.md
├── requirements.yaml
├── configs/
│   ├── default.yaml
│   ├── link_allocation_env.yaml
│   ├── train_link_allocation.yaml
│   └── train_rl.yaml
├── env/
│   ├── __init__.py
│   ├── bridge_robot_env.py
│   ├── dynamics.py
│   ├── kinematics.py
│   ├── link_allocation_env.py
│   ├── reward.py
│   └── torque_control_env.py
├── scripts/
│   ├── run_env.py
│   ├── train_link_allocation.py
│   ├── train_rl.py
│   └── visualize_env.py
├── tests/
└── visualization/
```

## 开发与运行

项目仍采用“本地开发，云端 Linux 执行”的方式：

- 本地负责编辑代码、维护配置和文档
- 云端负责安装依赖、运行脚本、保存结果和训练
- 默认结果目录为 `/openbayes/home/results`

推荐目录：

```text
/openbayes/home/
├── Project_jiaogai/
├── results/
└── envs/
```

创建并激活环境：

```bash
mkdir -p /openbayes/home/Project_jiaogai
mkdir -p /openbayes/home/results
mkdir -p /openbayes/home/envs

conda env create -p /openbayes/home/envs/bridge-robot-cloud -f /openbayes/home/Project_jiaogai/requirements.yaml
conda activate /openbayes/home/envs/bridge-robot-cloud
```

若镜像中 `conda activate` 不可直接使用：

```bash
source /opt/conda/etc/profile.d/conda.sh
conda activate /openbayes/home/envs/bridge-robot-cloud
```

测试命令：

```bash
python -m pytest tests
```
