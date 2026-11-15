# 在线 RL 数据接入与实现核对

2026-11-15；依据当前 EXPO-FT 提交 `21fe3d3`、OpenPI `expo_ft` 提交 `46407a4`，以及本机 `~/vla-precision` 当前源码。以下为源码和接入检查，不是正式训练结果。

## 现有资产与数据转换

用户已完成第一阶段 SFT；使用默认 LeRobot 缓存中的 `scylearning/insert_black_rack_into_blue_slot_20260916_v01`（v3、200 episodes、67,740 frames、15 FPS），以及 `~/vla-precision/checkpoints/stage1/pi05_full_finetune_ur5e/insert_black_rack_into_blue_slot/30000` 作临时检查。后续用户会提供新训练的 5,000 步权重，不需要在此仓库重复 SFT。

[转换脚本](../../scripts/convert_lerobot_to_expo_ft.py)读取本地 LeRobot v2/v3，输出 `数字episode/traj.hdf5`，字段包含 `saved_observation` 与 `action`。它支持内嵌图片、图片路径、视频；v3 按 episode 元数据里的视频文件编号和起始时间定位，跨 Parquet 文件的 episode 按 frame_index 合并。转换保留所有帧，校验时间序列、数值有限性和视频对齐；目标存在时拒绝覆盖，全部成功后才发布输出目录。

[本任务映射](lerobot-ur5e-insert.json)明确保存：

| 字段 | 来源与含义 |
|---|---|
| saved_observation/state | observation.state[0:19]，与当前 SFT 配置一致 |
| saved_observation/source_state | 全部 48 维原始状态 |
| saved_observation/cartesian_position | 原始 TCP pose 前 6 维，不转换旋转表示 |
| saved_observation/gripper_position | 原始状态第 18 维 |
| 三个 image 字段 | RGB 外部/腕部相机；第二外部图像复用第一外部相机，和作者两相机采集方式一致 |
| action/cartesian_delta | 原始 action[0:6]，保留增量动作语义和单位 |
| action/gripper_position | 原始 action[6] |
| saved_observation/prompt | Insert the black rack into the blue slot |

这里的兼容性是 HDF5 读取格式，不是将 UR 变成 DROID。**不能把 UR 增量动作标成 DROID velocity，也不能把 19 维 SFT 状态丢成 7 维**。[UR 输入转换](../../expo_ft/agents/vla/ur5e_data.py)在 OpenPI/replay 链路保留 19 维状态，复用原 checkpoint 的 norm stats，不额外做 delta 转换。完整在线运行还需把该数据工厂接入任务模型配置，并实现/接入 UR 客户端、奖励、重置及在线观测映射；原 `scripts/pick/` 仍是 DROID 示例，不是 UR 启动器。

用户确认 SFT 数据都是成功示教，转换时显式传入 `--assume-success`。上游 loader 将每条 episode 最后一步奖励设为 1、done 为 1，其余奖励为 0、mask 为 1；buffer 将示教标为成功/HIL。若未来加入失败数据，不可沿用这一假设。

## 服务器：转换命令

```bash
cd ~/expo-ft
source expo-setup/activate.sh
python scripts/convert_lerobot_to_expo_ft.py \
  --dataset-root "$HOME/.cache/huggingface/lerobot/scylearning/insert_black_rack_into_blue_slot_20260916_v01" \
  --output-dir data/online_rl/success \
  --mapping expo-setup/support/lerobot-ur5e-insert.json \
  --num-episodes 10 --seed 0 \
  --assume-success
```

默认随机不重复抽取 10 条后再转换，`--num-episodes N` 修改数量，`--seed` 默认 0；不足 N 条时取全部。抽样信息及原始 ID 保存到 `conversion.json`。也可用 `--episodes 0 1` 指定原始 ID，或 `--all-episodes` 显式全量转换，这三个选择参数互斥。2026-11-15 已执行 seed=0 的随机 10 条转换，结果位于 `data/online_rl/success`（实际存储在 `/data/suchenyu/expo-ft/data/online_rl/success`），共 3,622 帧、2,729,126,982 字节（约 2.54 GiB）。抽中原始 ID 为 `[10, 66, 77, 98, 103, 107, 122, 124, 130, 194]`。此前第 0 条、371 帧的单条验证仍保留在 `expo-setup/outputs/ur5e-insert-conversion-check`。保留原始分辨率和原始数据，转换并不恢复视频压缩前的信息。

与上游 `process_droid_dataset` 对接时，任务读取参数必须为 `action_space="cartesian_delta"`、`gripper_action_space="position"`。图片与状态在 `saved_observation` 中；reward/mask/done 由该 loader 生成，不要求写入 HDF5。

## 随机 10 条转换的验证记录

本次使用服务器根 `.venv`（Python 3.11.16），在 CPU 上执行转换及检查；源码基于 `21fe3d3` 加当前未提交改动，转换脚本和字段映射 SHA256 记录在[检查报告](logs/convert-random10-seed0-check-20261115.json)。没有使用模型权重或 GPU，没有启动训练和机器人。

10 条数据逐帧核对 48 维源状态、19 维训练状态、位姿、夹爪及 7 维动作，与原始 LeRobot 数值一致；HDF5 图像字段和提示检查通过。训练使用的读取器以 `num_data=10` 成功加载全部 3,622 条 transition，10 个成功终止标记及 mask 正确。这是数据读取验证，不代表模型训练通过。

[转换日志](logs/convert-random10-seed0-20261115.log)、[检查日志](logs/convert-random10-seed0-check-20261115.log)保存在本机，检查脚本随日志归档。服务器仓库根目录复查命令：

```bash
source expo-setup/activate.sh
PYTHONPATH=. python expo-setup/support/logs/check-converted-random10-20261115.py
```

数据已转换完成，训练时直接使用 `--dataset_path=./data/online_rl/success --num_data=10`，无需再次执行相同输出目录的转换命令。

## 服务器：接入验证命令

```bash
cd ~/expo-ft
source expo-setup/activate.sh
JAX_PLATFORMS=cpu python expo-setup/support/verify_ur5e.py \
  --converted expo-setup/outputs/ur5e-insert-conversion-check \
  --checkpoint "$HOME/vla-precision/checkpoints/stage1/pi05_full_finetune_ur5e/insert_black_rack_into_blue_slot/30000" \
  --asset-id scylearning/insert_black_rack_into_blue_slot_20260916_v01 \
  --report expo-setup/logs/ur5e-audit-20261115.json
```

默认仅校验真实数据、归一化/分词/buffer，以及 checkpoint 树和 LoRA 基础参数形状。要测模型前向，移除 `JAX_PLATFORMS=cpu`，选择空闲 GPU，设置 `XLA_PYTHON_CLIENT_PREALLOCATE=false` 并追加 `--forward`。`--action-horizon` 默认 50，来自当前 Stage-I YAML；权重树本身不能证明历史 action horizon，后续新权重应以实际归档配置为准。

本次 GPU 前向因四卡已各占用约 73.8GB 显存，在放入完整参数时 OOM，未产生动作；见 [失败日志](../logs/ur5e-forward-oom-20261115.log)。不将 checkpoint 参数兼容误写成模型完整前向或训练通过。未启动机器人或修改已有权重。

CPU 审计最终通过：[报告 JSON](../logs/ur5e-audit-20261115.json)、[审计日志](../logs/ur5e-audit-20261115.log)；格式回归 4 项通过：[测试日志](../logs/lerobot-conversion-tests-20261115.log)。权重包含 51 个向量化参数叶子，缺失的 20 个 LoRA 叶子可由加载器初始化；该计数不是按每个 Transformer 层展开的张量数量。

## 作者 SFT、在线 LoRA 与梯度范围

- 作者当前 `scripts/pick/finetune_droid.sh` 用的是 `expo_pi05_droid_lora_finetune_sft_cartesian_state`，第一阶段就是 LoRA。fork 同时提供 full-finetune 配置，但不是该默认脚本所选。
- 用户 30,000 步 checkpoint 不含 LoRA 张量。与作者双 LoRA 模型比对，基础参数名字及形状匹配，缺失的只有 LoRA 参数；加载器允许缺失 LoRA 由初始化补齐。该检查不代表已经实际运行过在线 LoRA 优化器，也不保证新增适配器后的输出与原全参策略逐位相同。
- 默认 LoRA 同时挂在 PaliGemma 的 Gemma 语言骨干（gemma_2b_lora）和 action expert（gemma_300m_lora）。这不等于给 SigLIP 视觉塔挂 LoRA。
- **更正此前“freeze_pi05_encoder=True 会冻结 VLM”的说法**：当前代码此标志控制多候选采样时是否复用一次前缀计算，没有修改 TrainConfig 的梯度过滤器。实际 `get_freeze_filter()` 冻结 llm 的非 LoRA 参数，VLM 和 action expert 的 LoRA 均可训练，视觉塔及部分投影层也未被该过滤器排除。不能声称只训练 action expert 或 encoder 完全冻结。

## 更新顺序与前向

一次 `EXPOLearner.update()` 顺序为：`utd_ratio` 次 critic 更新 → 1 次 pi0.5 成功轨迹监督更新 → 1 次 edit actor 更新 → 1 次温度更新（编辑开启时）。默认 utd_ratio=20；同步示例每次 episode 触发 3 次 update，因此是 60 次 critic minibatch 更新、3 次 pi0.5 更新、3 次 edit actor 更新及温度更新。不是先完成一个独立模仿训练阶段再开始 RL。

异步入口仅将采集与 learner 更新放到不同 GPU/线程；learner 内仍调用同一 update 顺序，更新后发布推理参数。同步入口的十 episode 门槛和异步入口的 buffer batch-size 门槛不同，详见代码；本次未修改这些调度。

critic 回归当前 Q 时使用 replay 中的动作；计算下一状态 target 时调用 pi0.5 产生候选动作。edit actor 更新使用最后一个 critic minibatch 的 replay 动作作为编辑基底，并不在该函数里重新调用 pi0.5。pi0.5 模仿更新走完整 flow-matching loss 前向。

EXPO-FT 的 pi0.5 采样调用先执行图像编码和 VLM prefix 前向得到 KV，然后供同次调用中的多个候选和多步去噪复用；不是每个去噪步都重新跑 VLM。但 replay 只存图像、状态、token 等，没有保存供跨 update 复用的 prefix KV，因此每次新的训练采样调用仍计算 prefix。pi0.5 监督更新也重新计算 prefix+suffix。

`~/vla-precision` 当前 ACoB 路径显式保存 ContextBuffer 的 prefix KV，learner 通过 `sample_actions_from_context` / `velocity_from_context` 只执行 suffix；其训练过滤器只允许 action expert 的 LoRA 更新。新观测在采集/预处理侧仍需要 VLM 编码，不能理解为整套系统完全不跑 VLM。这与 EXPO-FT 的“单次采样内部复用 KV”不同。

## UR 迁移复核（RLinf-rltoken 更新后）

2026-11-15 静态核对来源：`~/RLinf-rltoken`，提交 `f54ac81`。本节替代此前基于旧 checkout 的“没有键盘动作遥操作”的判断；尚未实施迁移或真机验证。

- 新版 `rlinf/robotics/parts/teleop/keyboard.py` 已实现键盘动作接管；UR 在线配置 `ur_rlt_stage2_ac_mlp.yaml` 显式启用。按住运动/夹爪/保持键覆盖当前动作，松开后返回策略，HOLD_WINDOW=0。平移 s/w、d/a、q/e，旋转 r/t、g/f、b/v，l 关闭、o 打开、[ 零运动接管，/ 切换步长；z 当前不激活接管。在线主/精细步长分别 0.03/0.006 m 和 0.05/0.03 rad。
- 必须同时迁移 `UR5eEnv.keyboard_action_to_env`：先应用 force-mode 轴选择及控制坐标旋转，再转换到 TCP/reference-frame 的策略增量与 action_scale；不能直接把按键向量作为策略动作。夹爪动作 0 关、1 开，观测保留测量开度。
- UR 使用本地 RTDE，无需独立 NUC 服务。在线 force-mode 后台跟踪为 100 Hz；当前继承的环境动作频率为 25 Hz，独立 VLA-Precision 评估配置为 20 Hz。文档的在线 20 Hz 与当前 YAML 不一致，以配置为准。continuous_tracking=true 时持续跟踪最新目标，不重复累加动作。新版旋转增益为 20/4，计算不再额外除频率，不能混用旧公式与新参数。
- 评估保留 VLA-Precision 的硬件、复位、录像、结果统计，模型边界替换为当前 EXPO-FT 输入、归一化、JAX SFT 或完整在线 checkpoint 推理。新版评估自带 RLT critical-phase/MLP 切换，该部分属于待替换模型逻辑，不能迁入 EXPO-FT。
- 在线按 EXPO-FT 的 real_action/action_type/is_hil 和原有 chunk 清空、重新规划逻辑接入键盘，不照搬 RLT 的策略切换、关键阶段筛选、预热、更新比例或 terminal_hold 特殊记录规则。复位确认只控制环境可用时机；成功/失败通过 EXPO-FT 的 reward/done/mask 表达。


## UR 适配实现进展

上述新版核对之后已完成迁移代码、启动入口和 CPU/假硬件检查。最新实现边界、评估配置与按键说明见[UR 迁移与运行](UR迁移与运行.md)，实际验证见[测试进展](../测试进展.md)。上层频率按用户最终决定使用 EXPO 的 10 Hz，底层保持 100 Hz；不会采用上文参考仓库的 20/25 Hz 上层配置。
