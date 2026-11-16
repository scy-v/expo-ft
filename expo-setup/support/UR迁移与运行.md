# UR 迁移与运行

实现来源：RLinf-rltoken `f54ac81` 的 UR/VLA-Precision 评估、键盘和阻抗控制；完整来源及改动见 `client/ur/provenance.json`。迁移代码在本仓库内独立使用，不在运行时导入 RLinf，不修改其他仓库或环境。

## 部署与调度

- 服务器（learner）：根目录 `.venv`，标准 `EXPOLearner`。10 Hz 逐动作下发，默认执行 8 步后同步重规划；模型预测长度单独设为 50。保留 EXPO 的候选、编辑、Q 选择、成功数据更新和 buffer 逻辑。没有启用 Real-Time 或移植 RLT 的关键阶段/MLP 路由。
- 本地机器人端（actor）：`client/.venv`，直接连接 UR RTDE、夹爪和 RealSense。无需 NUC。后台 forceMode 100 Hz 跟踪最新目标，在线环境不额外 sleep 限速；网络/推理超时仍会降低上层实际频率。
- 评估默认远程模型：GPU 上启动 `serve_eval.sh`，本地启动 `evaluate.sh`。远程评估端口 8000，与在线环境端口 8102 是两个独立协议。`policy.location: local` 时同一进程加载模型，本地 client 环境需要额外具备与服务器锁文件一致的模型依赖。

## 数据与模型契约

- 状态：复位参考下的 EE 位姿 6、TCP 速度 6、力/力矩 6、实测夹爪开度 1；两路 RGB 图像，右腕为模型填充视角。
- 动作：7 维，TCP 参考系的平移增量、Euler XYZ 旋转增量和夹爪开度；不能把 DROID 速度动作用于 UR。底层处理工具偏移、目标保持和阻抗；在线目标边界沿用 RLinf 的 UR workspace。
- 全参 SFT 评估配置：`configs/ur/insert_sft.yaml`。当前使用导入的 2500 步 JAX 权重（`checkpoints/sft/ur_insert/2500`）及其配套 norm stats，严格按 full 模型树加载。50 步预测长度来自已有任务配置，不由权重形状推断。
- 在线训练默认：`configs/model/expo_ft_ur_config.py`，全参 SFT 初始化双 LoRA Pi05，算法配置沿用 EXPO。全参在线可显式设 `--config.pi05_config_name=expo_pi05_ur_full`，尚未验证其显存与梯度运行。
- 完整在线模型评估配置：`configs/ur/insert_expo.yaml`。恢复 EXPO 的 `agent` 和 `params`，包含 Pi05、edit actor、critic、目标网络等；不是只抽取 Pi05。需要对应运行自动生成的 `ur_run.json`，检查架构、预测/执行长度和 norm stats SHA256。该文件记录实际模型配置与源码指纹；新权重不能混用旧归一化。
- 在线默认保持上游 10 条示教、offline_ratio=0、episode 更新 3 次及原训练门槛，不复制 RLT warmup、更新比例或关键阶段筛选。LeRobot 转换后数据路径显式传入；现已按 seed=0 随机转换 10 条到 `data/online_rl/success`，可由 `num_data=10` 全部读取；此前单条验证目录仍只含第 0 条，不用于替代这 10 条数据。

## 在线预热与固定 SFT 对照

服务器 `configs/ur/rollout.yaml` 默认 `warmup=true`、`sft_only=true`。预热在连接机器人之前完成，输出动作丢弃；不会把首次梯度编译也提前执行。`sft_only=true` 额外加载固定全参 SFT Pi05，使用在线相同的观测/动作/键盘/节拍链路，跳过 EXPO edit actor 和 critic 的动作选择，learner 仍照常更新；不等价于正式 EXPO rollout 或完整 EXPO 推理性能。命令与输出说明见[训练命令](../训练命令.md)。

默认同步入口的更新与推理共用服务器 agent，episode 更新后下一次采样直接使用新参数，不向本地 UR 客户端发送权重。异步入口每次更新发布推理缓存，rollout 循环拾取缓存；固定 SFT 诊断不使用这些更新。控制频率仍为动作10Hz、forceMode100Hz，同步 chunk 间推理等待仍存在。

本地交互终端显示已执行动作折算的 chunk 额度及当前余量；人工动作占额度，不将推理调用次数当作预算。非交互日志不输出回车覆盖行。

## 键盘与复位

在线使用机器人电脑的实体键盘（evdev）；多个键盘时用 `EXPO_KEYBOARD_DEVICE=/dev/input/eventX` 指定。按住动作键覆盖当前策略动作，松开后归还策略；回传实际采用的增量动作及 human/policy，EXPO 按原路径清空 chunk 并记录 is_hil。

| 键 | 操作 |
|---|---|
| s/w、d/a、q/e | 控制坐标系 x/y/z 正负移动 |
| r/t、g/f、b/v | 对应旋转正负；受 forceMode 轴选择限制 |
| l/o | 夹爪关/开（0/1） |
| [ | 零运动接管，保持夹爪目标 |
| / | 切换平移 0.03/0.006 m、旋转 0.05/0.03 rad |
| Enter | 首次允许复位；复位后确认开始 |
| 双击 Space（0.5 秒） | 成功并结束当前 episode |
| 运行中按 Enter / 双击 Enter | 不结束 episode；Enter 仅用于等待阶段确认 |

终止时停止 forceMode；等待确认不写 transition。终止若发生在观测和 step 之间，协议返回 `executed=false`，保留上一条真实动作并在下一次观测上补终止标签，避免伪造保持动作。成功/失败使用 EXPO 的 reward/done/mask，不采用 RLT terminal_hold 数据规则。上层在接管交接时通过 action_available 标记队列是否有策略动作；松键后的无动作轮只归还控制，保持当前夹爪目标，避免把原 DROID 的零速度占位解释为 UR 关夹爪。

独立评估保持原 VLA-Precision 的终端按键与录像/结果统计，不使用在线遥操作键。移除了 RLT 专属双击点号切换 MLP。评估夹爪阈值沿用 0.7，在线沿用 UR RLT 的 0.5；均显式记录在配置中。在线显示与日志保留，但不在内存无限积累评估录像帧；需要录像时使用独立评估入口。

## 验证边界

2026-11-16 本地补充：独立 actor 环境、22 项本地回归及两台相机逐台取帧通过；UR/夹爪实际控制和跨机器训练未验证，详见[本地安装检查](本地安装检查.md)。下列为迁移当时的服务器验证记录。

CPU/假硬件测试覆盖按键/坐标转换/轴选择/松键、成功锁存、动作额度耗尽终止、数据字段、EXPO 节拍边界、100 Hz 后台控制结构、MessagePack 通信、终止未执行动作的标记和 native Orbax 小参数 checkpoint 完整恢复。实际 30000 步 checkpoint 已在主机 CPU 上通过新的评估加载器，未执行前向。沙箱内 Orbax 元数据读取会等待，主机权限下同一加载约 11 秒通过。

四张 A800 本次检查各已占用约 73.8 GB，未干扰这些任务。GPU 前向/梯度、真实训练后的完整 EXPO checkpoint 推理、机器人端依赖安装、跨机器网络、真实夹爪/相机/UR 尚未验证。没有连接机器人，没有运行正式实验。小参数恢复测试不代表真实在线模型前向已通过。
