# 在线更新 OOM 核查（2026-11-16）

用户提供的 run_001 终端记录在21:40:36明确报告 GPU0 RESOURCE_EXHAUSTED，申请770703360字节失败。该运行保存的 rollout_config.json 为 sft_only=false，不能归因于额外固定SFT模型。检查时进程265023已不存在，四卡显存已释放；无法追溯进程环境覆盖或获得完整异常栈。未修改训练参数、未重启训练或连接机器人。

## 已验证的显存因素

- sample_jax 接收 data_sharding 但转换数组时使用 jnp.asarray，默认设备上先准备完整批次；get_iterator 默认预取3批，转换float32后才由BatchProcessor.next_batch调用apply_data_sharding。
- 使用真实采样／转换／分片函数、合成32帧存储及batch16，在四张A800上验证：转换后的图像为SingleDeviceSharding GPU0；最后分片才变成每卡4张。不加载模型。第一次诊断导入遇到既有循环导入，调整为训练入口采用的agents优先导入顺序后通过。
- 正式批次64×UTD20=1280；单个float32 RGB数组1280×224×224×3×4=770703360字节（735MiB），与错误请求大小完全相同。每批含当前／下一／延迟下一观测的9个图像字段。大小对应强烈指向数据准备压力，但缺少完整异常栈，不能仅凭大小锁定唯一失败算子。
- 推理缓存cache_infer_params还把完整actor_train_state放到GPU0，不仅是推理权重；梯度、优化器、新旧参数和编译临时数组进一步叠加。
- 独立标量GPU测试：XLA_PYTHON_CLIENT_PREALLOCATE=false，两个MEM_FRACTION变量均未设置，jax.devices()[0].memory_stats()返回bytes_limit=63820333056（约59.44GiB）。启动脚本默认只关闭预分配，未增加额度；原进程环境已不可读，因此该值是默认环境复验，不是原进程实测额度。此前原进程四卡约60GiB占用与此相符。

## 更新计数与后续修复方向

1/3来自num_updates=3，每个episode触发3次完整update调用，每次UTD20次critic更新，并包含Pi05 actor、残差actor和温度更新；不是3张GPU，也不是3个episode。JAX执行异步，进度条前进不等于所有设备计算已成功同步。

优先消除GPU0未分片预取（CPU准备后分片上传或提前分片）与不必要推理状态复制，保留batch64、UTD20和更新次数。独占GPU条件下可评估显式提高分配器额度，但不能只靠提高额度宣称修复。需再完成连续3次更新并同步检查结果；本轮仅诊断，未完成修复／完整模型连续更新验证。

证据：[小规模GPU分片日志](logs/replay-placement-20261116.log)、[诊断脚本](logs/replay-placement-20261116.py)。


## CPU预取与训练编译优化

基于 EXPO 工作区提交 f2fb2dc7ab173c54c8643c71c3c9dde97fa98421 修改：

- get_iterator预取以及成功样本actor batch均使用自有NumPy数据，避免复用gather staging覆盖已缓存样本。延迟为0的共享图像复用归一化结果；混合buffer也在CPU拼接／洗牌，最后使用既有分片上传函数。
- 推理cache中的actor TrainState保留采样所需params/model_def，opt_state改为空元组、ema_params清空；learner本体与checkpoint保存状态不变。释放入口的初始化状态局部引用，避免跨更新驻留旧权重。
- warmup_update默认true，独立临时采样器用于准备正式shape／sharding并执行lower().compile()，不执行梯度。恢复buffer RNG，不消费正式iterator；不修改episode、优化器或模型计数。编译时临时关闭OpenPI对JAX ArgInfo元数据不适用的运行时类型检查，成功或失败都恢复。
- 更新循环显式等待设备完成后才记录进度／耗时或发布权重，避免把异步提交当作更新完成。
- batch64、UTD20、num_updates3、LoRA学习率及显存额度保持不变。CPU内存会承担预取数据；训练和实际推理仍在GPU执行。

验证：87项CPU／假硬件／localhost测试通过；新增覆盖采样数值、数组所有权、混合洗牌、编译不更新、随机数恢复及失败清理。真实模型首轮测试因opt_state=None不满足TrainState类型契约失败，改为空元组；第二轮AOT lowering遇到ArgInfo类型检查，已增加限定范围的处理。原始失败日志保留。四卡真实模型连续更新结果见下文。


### 连续更新与二次编译排查

真实2500步全参初始化权重、10条成功示教、4×A800 80GB，batch64／UTD20；诊断buffer容量4000（正式命令仍100000），不连接机器人、不创建正式运行或保存模型。进程使用XLA_PYTHON_CLIENT_PREALLOCATE=false，未提高显存额度。

第一版仅编译初始化签名时，首轮更新39.85秒；第二轮166.75秒，随后仍触发编译。检查真实更新前后pytree发现：首轮target critic从单卡变为四卡复制，温度参数及Adam nu从weak float32变为普通float32；第二轮Adam mu再变为普通float32，之后签名稳定。预热现通过抽象输出推演并编译至稳定签名（最多4种，未稳定则启动失败），不执行梯度或改写参数，避免把这些编译留给在线episode。

进一步将主循环one-ahead预取也留在CPU，消费时再上传；中间版六次完整更新成功，时间为38.04／36.45／51.77／29.88／30.04／30.19秒，所有返回指标有限。该轮仍仅提前编译前两种签名，第三次的额外耗时推动了完整预热修正；第四次起JIT缓存数量保持3。每卡峰值约57.4GiB，分配器上限59.44GiB；没有OOM、没有调低batch或UTD。这是短跑验证，不是100000步正式实验。


### 最终版验证

完整预热实测编译3种签名，模型step和采样RNG保持不变。随后6次同步完成的更新耗时40.77／39.14／39.45／32.22／32.49／32.48秒，后3次平均32.40秒，未再次出现分钟级编译等待。所有返回指标有限；更新后完整EXPO动作推理通过（约0.168秒，非真机端到端延迟）。

GPU分配器统计最高峰61600945152字节，约57.37GiB，低于本机默认上限63820333056字节（59.44GiB）。未设置额外MEM_FRACTION、未降低batch／UTD。两轮各6次更新通过不能替代100000步长期运行或真机验证；固定SFT额外模型模式本轮未测试。

最终日志：[CPU 87项](logs/update-prefetch-cpu-20261116.log)、[GPU最终6次更新及后续推理](logs/update-prefetch-final-20261116.log)、[JSON](logs/update-prefetch-final-20261116.json)、[诊断脚本](logs/update-prefetch-final-20261116.py)。中间证据：[首次类型错误](logs/update-prefetch-initial-failure-20261116.log)、[AOT类型错误](logs/update-prefetch-aot-failure-20261116.log)、[重复编译诊断，随后主动停止](logs/update-prefetch-recompile-20261116.log)、[中间版6次更新](logs/update-prefetch-six-updates-20261116.log)。
