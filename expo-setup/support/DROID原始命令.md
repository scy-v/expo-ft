# 作者 DROID / Franka 原始命令（历史参考）

这里只保留作者 pick 示例，**不用于当前 UR 插槽任务**。当前操作入口是[UR 在线训练命令](../训练命令.md)。命令在各机器的 `~/expo-ft` 根目录执行，尚未在本项目验证 DROID 真机。

## 服务器（Learner）

### 环境与运行前配置

```bash
cd ~/expo-ft
source expo-setup/activate.sh
```

开始前准备服务器上的原始示教、LeRobot 数据、pi0.5 基座权重与归一化资产；检查 `configs/task/pick.py`、`configs/model/expo_ft_pi_config.py` 和下面的脚本。数据及权重路径均为服务器路径。

示例脚本内部设置 `CUDA_VISIBLE_DEVICES`：同步训练/SFT/评估为 `0,1,2,3`，异步为 `0,1,2`，外部环境变量会被覆盖。按实际可用 GPU 修改脚本；异步至少需要 2 张 GPU。`--resume` 是否恢复已有运行须核对输出目录，首次实验使用独立运行名。

### 数据转换、归一化与 SFT（依次执行）

```bash
bash scripts/pick/convert_data.sh
bash scripts/pick/calculate_norm.sh
bash scripts/pick/finetune_droid.sh
```

默认原始数据为 `data/pick_cube_balance/success`，转换前 10 条，LeRobot ID 为 `expo_ft/droid_pick_cube_10`。SFT 配置名为 `expo_pi05_droid_lora_finetune_sft_cartesian_state`，脚本设置 4001 步、每 2000 步保存；资产 ID、数据 ID 与任务修改时一并更新。固定状态任务的归一化调整遵循上游 README，需结合实际统计确认。

### 在线 RL（二选一）

在本地端与 NUC 按下文启动后运行；learner 等待 actor 连接，因此两端启动先后不强制。

```bash
# 同步
bash scripts/pick/run_server.sh
# 或异步（不要同时启动同端口的两个 learner）
bash scripts/pick/run_server_async.sh
```

示例加载 SFT 的 `2000/params`，不是自动选择最新权重。核对 `pi05_weight_loader_path`、`pi05_assets_dir`、`pi05_asset_id`、`dataset_path`、`run_name` 和 `output_dir`。learner 监听 `0.0.0.0:8102`，actor 主动连接；`client_host` 是 learner 的监听地址。

### 评估

```bash
bash scripts/pick/eval_policy.sh
```

先停止占用相同端口的训练进程，再启动评估。示例 RL checkpoint step 为 4000、评估 35 个 episode；按真实保存的 checkpoint 修改，保持模型、动作参数与归一化一致。未验证的命令不视为训练成功。

## 本地机器人端（Actor）

### 采集示教与同步数据

完成硬件配置、启动 NUC 后：

```bash
cd ~/expo-ft
source client/.venv/bin/activate
bash scripts/pick/collect_data.sh
```

示例采集 15 条到 `data/pick_cube_balance`；上面的服务器转换默认只取 10 条，按实验设计统一数量。将采集目录复制到服务器同一相对位置后再转换。例如在本地端设置服务器地址后：

```bash
export EXPO_SERVER=user@gpu-host
rsync -av data/pick_cube_balance/ "${EXPO_SERVER}:~/expo-ft/data/pick_cube_balance/"
```

### 在线训练或评估的 rollout

```bash
cd ~/expo-ft
SERVER_HOST=服务器可达IP或主机名 bash scripts/pick/run_policy.sh
```

本地需要能访问服务器 TCP 8102；无法直连时可在本地另开终端运行 `ssh -N -L 8102:localhost:8102 user@gpu-host`，并设置 `SERVER_HOST=localhost`。修改任务时同步两端代码及任务配置；机器人端不需要服务器训练环境或训练 checkpoint。

## NUC（机器人控制主机）

从 NUC 的 DROID 安装根目录执行：

```bash
python scripts/server/run_server.py
```

运行前确认 NUC 的机器人参数及本地端 `client/droid/droid/misc/parameters.py` 中的 `nuc_ip`、任务配置相机 ID。机器人型号、IP、相机、数据集和基座权重的实际选择后续补充。
