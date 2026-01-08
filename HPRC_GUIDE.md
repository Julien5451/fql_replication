# HPRC 集群运行指南

本指南帮助你在 HPRC (High Performance Research Computing) 集群上运行 Flow Q-Learning (FQL) 实验。

## 目录

- [快速开始](#快速开始)
- [环境设置](#环境设置)
- [运行实验](#运行实验)
- [监控作业](#监控作业)
- [常见问题](#常见问题)

## 快速开始

```bash
# 1. 登录 HPRC 集群
ssh your_netid@grace.hprc.tamu.edu

# 2. 克隆或复制项目到集群
cd $SCRATCH
git clone <your-repo-url> fql
cd fql

# 3. 修改配置 (重要！)
vim scripts/config.sh  # 修改邮箱和其他设置

# 4. 设置环境 (首次运行)
chmod +x scripts/*.sh
./scripts/setup_env.sh

# 5. 运行单个实验
./scripts/run_single.sh antmaze-large-navigate-singletask-v0 10

# 6. 或批量运行所有实验
./scripts/submit_experiments.sh ogbench_state
```

## 环境设置

### 1. 修改集群配置

首先编辑 `scripts/config.sh`，根据你的集群修改以下设置：

```bash
# 你的邮箱 (用于接收作业通知)
export HPRC_EMAIL="your_netid@tamu.edu"

# GPU 分区 (根据集群调整)
export HPRC_PARTITION="gpu"  # 或 "gpu-a100", "gpu-v100" 等

# 模块版本 (根据集群可用版本调整)
export HPRC_GCC_MODULE="GCC/11.3.0"
export HPRC_CUDA_MODULE="CUDA/12.0.0"
export HPRC_ANACONDA_MODULE="Anaconda3/2023.03"
```

### 2. 查看可用模块

```bash
# 查看可用的模块
module avail GCC
module avail CUDA
module avail Anaconda
```

### 3. 运行环境设置脚本

```bash
./scripts/setup_env.sh
```

这将创建一个名为 `fql` 的 conda 环境，并安装所有依赖。

### 4. 配置 Weights & Biases (可选但推荐)

```bash
conda activate fql
wandb login
# 输入你的 W&B API key
```

如果集群没有网络访问，设置离线模式：
```bash
export WANDB_MODE="offline"
```

## 运行实验

### 方法 1: 运行单个实验

```bash
# 基本用法
./scripts/run_single.sh <env_name> [alpha] [discount] [q_agg] [seed]

# 示例
./scripts/run_single.sh antmaze-large-navigate-singletask-v0 10 0.99 min 0
./scripts/run_single.sh cube-double-play-singletask-v0 300
./scripts/run_single.sh visual-cube-single-play-singletask-task1-v0 300
```

### 方法 2: 直接使用 sbatch

```bash
# 默认参数运行
sbatch scripts/run_fql.slurm

# 自定义参数
sbatch --export=ALL,ENV_NAME="antmaze-large-navigate-singletask-v0",ALPHA=10,SEED=42 scripts/run_fql.slurm
```

### 方法 3: 批量提交实验

```bash
# 提交所有 OGBench 状态空间实验
./scripts/submit_experiments.sh ogbench_state

# 提交所有 OGBench 像素空间实验
./scripts/submit_experiments.sh ogbench_visual

# 提交所有 D4RL 实验
./scripts/submit_experiments.sh d4rl

# 提交 Offline-to-Online 实验
./scripts/submit_experiments.sh offline2online

# 提交所有实验
./scripts/submit_experiments.sh all
```

### 常用实验配置

| 环境 | Alpha | Discount | Q_AGG |
|------|-------|----------|-------|
| antmaze-large-navigate | 10 | 0.99 | min |
| antmaze-giant-navigate | 10 | 0.995 | min |
| humanoidmaze-* | 30 | 0.995 | mean |
| cube-* | 300 | 0.99 | mean |
| puzzle-* | 1000 | 0.99 | mean |
| visual-* | 100-300 | 0.99 | mean |

## 监控作业

### 查看作业状态

```bash
# 查看你的所有作业
squeue -u $USER

# 运行监控脚本
./scripts/check_jobs.sh

# 查看特定作业详情
scontrol show job <job_id>
```

### 查看作业输出

```bash
# 实时查看输出
tail -f logs/fql_<env_name>_<job_id>.out

# 查看错误日志
cat logs/fql_<env_name>_<job_id>.err
```

### 取消作业

```bash
# 取消单个作业
scancel <job_id>

# 取消所有你的作业
scancel -u $USER

# 取消特定名称的作业
scancel --name="fql_antmaze*"
```

## 结果和日志

### 目录结构

```
fql/
├── exp/                    # 实验结果
│   └── fql/               # W&B 项目名
│       └── <run_group>/   # 实验组
│           └── <exp_name>/ # 单次实验
│               ├── flags.json    # 运行参数
│               ├── train.csv     # 训练日志
│               ├── eval.csv      # 评估日志
│               └── *.pkl         # 模型检查点
├── logs/                   # SLURM 日志
│   ├── fql_<job_id>.out   # 标准输出
│   └── fql_<job_id>.err   # 错误输出
└── scripts/               # 运行脚本
```

### 查看 W&B 日志

如果使用在线模式：
1. 访问 https://wandb.ai/
2. 找到你的项目 `fql`
3. 查看训练曲线和评估结果

如果使用离线模式：
```bash
# 同步离线日志到 W&B
wandb sync exp/fql/<run_group>/<exp_name>/wandb/
```

## 常见问题

### Q: 模块加载失败

**A:** 查看可用模块并更新 `config.sh`：
```bash
module avail | grep -i cuda
module avail | grep -i gcc
```

### Q: CUDA 内存不足

**A:** 尝试以下方法：
1. 减少 batch size: `--agent.batch_size=128`
2. 调整 JAX 内存分配:
   ```bash
   export XLA_PYTHON_CLIENT_MEM_FRACTION=0.6
   ```

### Q: D4RL 安装失败

**A:** D4RL 需要 MuJoCo 2.1.0。设置步骤：
```bash
# 下载 MuJoCo
mkdir -p ~/.mujoco
cd ~/.mujoco
wget https://github.com/deepmind/mujoco/releases/download/2.1.0/mujoco210-linux-x86_64.tar.gz
tar -xf mujoco210-linux-x86_64.tar.gz
mv mujoco210 mujoco210

# 添加到环境变量 (添加到 ~/.bashrc)
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/.mujoco/mujoco210/bin
export MUJOCO_PY_MUJOCO_PATH=~/.mujoco/mujoco210
```

### Q: 作业等待时间太长

**A:** 尝试：
1. 使用不同的分区: `--partition=gpu-a100`
2. 减少请求的资源
3. 查看集群负载: `sinfo`

### Q: W&B 连接失败

**A:** 使用离线模式：
```bash
export WANDB_MODE="offline"
```

### Q: 如何恢复训练

**A:** 使用检查点恢复：
```bash
python main.py --env_name=<env> --restore_path=exp/fql/<group>/<exp> --restore_epoch=500000
```

## TAMU HPRC 特定信息

### Grace 集群

```bash
# 推荐分区
--partition=gpu          # 通用 GPU
--partition=gpu-a100     # A100 GPU (更快)

# 模块
module load GCC/11.3.0 CUDA/12.0.0 Anaconda3/2023.03
```

### Terra 集群

```bash
# 推荐分区
--partition=gpu

# 模块 (可能版本不同)
module load GCC/10.2.0 CUDA/11.3.0 Anaconda3/2021.05
```

### 存储建议

- 使用 `$SCRATCH` 运行实验 (大存储空间)
- 重要结果备份到 `$HOME` 或外部存储
- `$SCRATCH` 文件可能会定期清理

## 支持

如有问题，请：
1. 查看 HPRC 文档: https://hprc.tamu.edu/
2. 联系 HPRC 支持: help@hprc.tamu.edu
3. 查看项目 issues
