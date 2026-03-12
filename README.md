# 桥梁检测机器人控制优化

项目当前包含两条能力线：

- Phase 1：已经完成的 `torque-based physics environment`，用于四自由度平面机械臂的最小物理闭环。
- 当前主线：`link allocation RL environment`，在总杆长固定的约束下优化四连杆长度分配，使末端工作空间尽可能大。

## 当前 RL 主线

当前强化学习任务不是直接学习关节力矩，而是优化四根杆的长度分配。

目标表述为：

> 在保持四根杆总长不变的前提下，分配各杆长度，使末端在受限关节角域内形成尽可能大的可达工作空间。

## 为什么旧奖励会退化

旧版本奖励采用解析环域面积：

- `outer_radius = sum(lengths) = total_length`
- `inner_radius = max(0, 2 * max(lengths) - total_length)`

在当前默认约束下：

- `total_length = 3.6`
- `max_link_lengths = [1.4, 1.4, 1.4, 1.4]`

因为任意单杆都小于 `total_length / 2 = 1.8`，所有可行解都会得到 `inner_radius = 0`，于是奖励恒定，RL 无法学到有区分度的策略。

## 新奖励定义

当前奖励已经改为“有限关节角域下的采样工作空间覆盖率”：

1. 在显式关节角限制内采样一批姿态 `q`
2. 用正运动学得到末端点云
3. 将点云映射到 2D occupancy grid
4. 以 `occupied_cells / total_cells` 作为 reward

同时输出：

- `workspace_points`
- `occupied_ratio`
- `workspace_area_estimate`
- `grid_shape`
- `xy_bounds`

这种定义保留了孔洞、非凸边界和角域限制带来的真实差异，不再依赖解析环域公式。

## 仓库结构

```text
project/
├── README.md
├── requirements.yaml
├── configs/
│   ├── default.yaml
│   ├── link_allocation_env.yaml
│   └── train_rl.yaml
├── env/
│   ├── __init__.py
│   ├── bridge_robot_env.py
│   ├── dynamics.py
│   ├── kinematics.py
│   ├── link_allocation_env.py
│   └── reward.py
├── scripts/
│   ├── run_env.py
│   ├── train_rl.py
│   └── visualize_env.py
├── tests/
│   ├── test_dynamics.py
│   ├── test_env.py
│   ├── test_kinematics.py
│   ├── test_link_allocation_env.py
│   ├── test_run_env.py
│   ├── test_train_rl.py
│   └── test_visualization.py
└── visualization/
    ├── __init__.py
    ├── link_allocation.py
    ├── plots.py
    ├── render.py
    └── video.py
```

## Phase 1：已完成的 torque-based 环境

这部分已经完成并稳定，可继续作为物理基线使用：

- `BridgeRobotEnv`
- `env/kinematics.py`
- `env/dynamics.py`
- `env/reward.py`
- `scripts/run_env.py`
- `scripts/visualize_env.py`
- `visualization/`

常用命令：

```bash
python scripts/run_env.py --policy zero --seed 7 --output-dir /openbayes/home/results
python scripts/run_env.py --policy random --seed 7 --output-dir /openbayes/home/results
python scripts/visualize_env.py
python -m pytest tests
```

## Phase 2：杆长分配 RL 环境

当前 RL 环境为 `LinkAllocationEnv`，保持单步、bandit-like 形式：

- 观测：固定 13 维，包含默认杆长、上下界和总长
- 动作：4 维候选杆长
- 约束：动作会被投影到“逐杆上下界 + 总长固定”的 bounded simplex
- 奖励：受限角域下的工作空间 occupancy coverage
- 回合：`step()` 后立即结束

### 默认环境配置

`configs/link_allocation_env.yaml` 当前包含：

- `total_length = 3.6`
- `default_link_lengths = [1.2, 1.0, 0.8, 0.6]`
- `min_link_lengths = [0.4, 0.4, 0.4, 0.4]`
- `max_link_lengths = [1.4, 1.4, 1.4, 1.4]`
- `joint_angle_limits`
- `workspace_sampling`
- `video`

其中：

- `joint_angle_limits` 控制工作空间评估时允许采样的关节角范围
- `workspace_sampling` 控制采样数量、seed、occupancy grid 大小和显示边界
- `video` 控制训练后自动导出的 MP4 参数

### 环境接口

```python
from env.link_allocation_env import LinkAllocationEnv

env = LinkAllocationEnv()
obs, info = env.reset(seed=7)
obs, reward, terminated, truncated, info = env.step([0.9, 0.9, 0.9, 0.9])
```

`step(...).info` 至少包含：

- `allocated_lengths`
- `raw_action`
- `projection_applied`
- `workspace_points`
- `occupied_ratio`
- `workspace_area_estimate`
- `grid_shape`
- `xy_bounds`

## SAC 训练入口

`scripts/train_rl.py` 使用 SAC 训练杆长分配策略。

默认训练配置在 `configs/train_rl.yaml`：

- `algo = sac`
- `policy = MlpPolicy`
- `total_timesteps = 2000`
- `seed = 7`
- `device = auto`
- `learning_starts = 32`
- `buffer_size = 4096`
- `batch_size = 64`
- `train_freq = 1`
- `gradient_steps = 1`
- `learning_rate = 3e-4`
- `gamma = 0.99`
- `tau = 0.005`
- `eval_episodes = 5`
- `run_name = sac_link_alloc`
- `output_dir = /openbayes/home/results`

训练命令：

```bash
python scripts/train_rl.py
```

覆盖参数：

```bash
python scripts/train_rl.py \
  --total-timesteps 5000 \
  --run-name exp001 \
  --output-dir /openbayes/home/results
```

## 训练输出

训练结果写入：

```text
<output_dir>/rl_link_alloc/<run_name>/
```

至少包含：

- `model_final.zip`
- `train_config.json`
- `evaluation.json`
- `best_lengths.json`
- `best_workspace_samples.npz`
- `best_workspace.png`
- `best_workspace.mp4`
- `monitor.csv`

其中：

- `best_workspace_samples.npz` 保存最佳杆长对应的工作空间点云和采样角度
- `best_workspace.png` 是静态预览图
- `best_workspace.mp4` 是点云累积动画，直观展示 RL 训练后最佳机构的工作空间覆盖过程

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

## 当前状态

- 已完成：Phase 1 torque-based 最小物理环境闭环
- 已完成：杆长分配单步 RL 环境
- 已完成：受限角域 occupancy reward
- 已完成：训练后自动导出 `.npz`、`.png`、`.mp4`
- 当前主线：固定总长约束下的四杆长度分配优化
