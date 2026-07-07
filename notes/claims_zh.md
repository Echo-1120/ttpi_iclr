# 论文结论台账

英文同步版本：`notes/claims.md`。

## 论文定位

工作标题：
张量链策略迭代中的模态排序：分块保留的耦合感知重排序。

核心定位：
模态排序是 TTPI 中的一等算法变量，因为 TT 秩由排序诱导的序列展开决定。当前证据不再支持把静态 rank-aware proxy 作为主方法。更稳妥的主线是
block-preserving coupling-aware ordering：确定性的结构 coupling 先验比朴素 rank-aware cut proxy 或简单 physics--sensitivity 混合更稳定地降低 TTPI 资源负担。`sensitivity_pam` 应被视为代码名；当前分数不是经验轨迹敏感度估计器。

安全主张：
在固定 TTPI 近似预算下，动作模态排序会改变中间矩阵展开秩、TT-Cross 请求负担、峰值 GPU 显存、OOM 率和运行时间。在当前 5080 预实验中，确定性 CouplingAwarePAM（代码中为 `sensitivity_pam`）是最稳定的已实现排序候选。

不要作为标题级主张：
- 物理感知排序总是更好。
- 更低的物理线性排列目标值会直接带来更高回报。
- 当前静态 RankAwareProxy 已经是解决好的 rank-aware 方法。
- 在重新设计混合规则前，HybridPAM 是主方法。
- 当 OOM 率不同时，只报告完成运行成功率已经足够。

## Claim C1

在固定近似预算下，模态排序会影响 TTPI 的资源消耗。

状态：当前 HM8/HM12 RTX 5080 预实验支持。

证据：
- HM8 formal 运行中所有主排序均可完成，但峰值显存差异明显：BlockPAM
  2539.8 MB，CouplingAwarePAM 2597.1 MB，Local 2806.2 MB，RankAwareProxy
  5012.2 MB。
- HM12 暴露了 OOM 差异：Local 为 3/10 OOM，BlockPAM 为 4/10，CouplingAwarePAM
  为 1/10，HybridPAM 为 7/10，RankAwareProxy 为 10/10。
- TT-Cross 请求负担也会变化：HM12 CouplingAwarePAM 平均 148.80M 次请求，而
  Local 为 190.56M，BlockPAM 为 185.53M，HybridPAM 为 190.57M。
- 当前 formal 配置下 HM16 在 RTX 5080 上 150/150 OOM，应留给更高显存服务器。

需要的实验：
- 整理 actuator-relabelled 与 cross-coupled formal 运行后，单独汇总压力测试环境。
- 最终投稿时对主要资源指标报告置信区间或 paired seed 检验。

不要声称：
- 排序总能提升控制性能。
- HM8 不 OOM 意味着单张 RTX 5080 可以扩展到 HM16。

## Claim C2

Coupling-aware block-preserving ordering 是当前最强主方法候选。

状态：支持作为当前实现主线；仍需要更广泛压力测试确认。

证据：
- 在 HM8 上，CouplingAwarePAM（代码中为 `sensitivity_pam`）在主方法完成运行中平均成功率最高（0.717），同时显存接近 BlockPAM。
- 在 HM12 上，CouplingAwarePAM 是最稳定的资源选择：1/10 OOM，平均峰值显存
  9476.6 MB，平均 TT-Cross 请求 148.80M。
- HM12 的 completion-aware performance 也支持 CouplingAwarePAM：完成概率
  0.90，完成运行条件成功率 0.481，将 OOM 视为失败的 operational success
  为 0.433。Local 与 BlockPAM 的 operational success 更低（0.330 和
  0.341），HybridPAM 因为只完成 3/10 个 seed，降为 0.167。
- HM8 seeds 0--2 的 hybrid objective-toggle 预实验显示，sensitivity-only
  目标在 block-only、physics-only、sensitivity-only、current-hybrid 四组中同时取得最低显存（2505.7 MB）、最短时间（156.2 s）、最低查询数（25.88M）和最高成功率（0.747）。

需要的实验：
- 在 actuator-relabelled 和 cross-coupled HardMove 压力测试中确认。
- 使用相同 seed 和评估设置对比 Local、BlockPAM、BadSplit、Random。

不要声称：
- 当前确定性 sensitivity proxy 是经验轨迹敏感度估计器。它目前是确定性的角距离 coupling proxy。
- 方法优势是脱离资源可行性的纯控制性能优势。

## Claim C3

当前静态 RankAwareProxy 目标不能作为主方法。

状态：支持作为失败消融。

证据：
- RankAwareProxy 使用静态分段 cut-load proxy，而不是奇异谱或动态 TT-Cross 秩估计。
- 在 HM8 上它将显存提高到 5012.2 MB，请求数提高到 45.16M，差于 Local、BlockPAM 和 CouplingAwarePAM。
- 在 HM12 上它 10/10 OOM。
- 当前报告中的 proxy--actual 相关性弱且不一致：与峰值显存的相关在 HM8 为
  -0.15，在 HM12 为 -0.01。

需要的实验：
- 如果论文中继续保留 rank-aware 方法，应围绕真实 cut-rank 或 spectral cache 重新设计，而不是使用当前静态 proxy。
- 使用逐 cut 秩热力图和动态 TT-Cross process 日志识别瓶颈 cut。

不要声称：
- 当前 rank-aware proxy 能预测显存。
- 静态 cut load 等价于动态 TT-Cross 秩负担。

## Claim C4

加权线性排列是 cut-profile 面积目标，而不是直接的 peak-rank 目标。

状态：理论上支持；经验上与静态 proxy 失败一致。

证据：
- 对加权 cut cost \(C_k^W(\pi)\)，恒等式
  \(J_{\mathrm{LA}}(\pi)=\sum_k C_k^W(\pi)\) 成立。
- TT 存储、TT-Round 负担和 GPU 显存相比总 cut 面积，更容易受到峰值秩或瓶颈 cut 影响。
- RankAwareProxy 可以有较好的静态目标值，但在动态 TTPI 运行中失败，尤其是 HM12。

需要的实验：
- 同时报告 LA、peak-cut、proxy score、actual rank-AUC 和 peak memory。
- 只有在奇异谱估计足够代表性时，才加入奇异谱机制主张。

不要声称：
- LA 是错误或无用的。它仍是合理的累计负担替代指标。

## Claim C5

经验/轻量 sensitivity 变体目前不能替代当前确定性 CouplingAwarePAM。

状态：HM8 seed 0--2 预实验支持。

证据：
- SensitivityLiteFD5 可以完成，但代价高：峰值显存 6452.1 MB，运行时间
  211.6 s，请求数 61.55M，mean max advantage rank 为 11.67。
- SensitivityLiteFirstOrder 同样代价高且性能更不稳定：7565.9 MB，211.8 s，
  53.75M 请求，成功率 0.497，mean max advantage rank 为 15.67。
- 二者都明显比当前确定性 CouplingAwarePAM 更昂贵。

需要的实验：
- 除非重新设计构造目标，否则不要把这些 lite 变体扩展到 HM12。

不要声称：
- 更多经验 sensitivity 信息会自动改善 TTPI 资源使用。

## Claim C6

排序只能通过有限预算近似误差或动作搜索误差影响控制性能。

状态：理论桥梁已确定；当前性能证据应从属于资源证据。

证据：
- 如果某个排序对应的 TT 近似具有一致 advantage 误差 \(\varepsilon_\pi\)，则它的贪心动作距离真实一步贪心 advantage 至多 \(2\varepsilon_\pi\)。
- HM8 上 CouplingAwarePAM 的平均成功率高于 Local 和 BlockPAM，但 HM12 完成运行成功率会受到 OOM 过滤影响。有效结论必须同时报告 OOM 率和完成运行性能。
- 在 HM12 上，CouplingAwarePAM 的完成运行条件成功率低于 BlockPAM（0.481
  对 0.568），但将 OOM 视为失败后的 operational success 更高（0.433 对
  0.341）。这一差异是本文“资源可行性”叙事的核心。

需要的实验：
- 最终性能主张应同时报告 OOM-as-failure 的 effective success 和 completed-run success。
- 除非二者在共享 seed 上一致，否则资源结论与性能结论要分开写。

不要声称：
- 更好的资源指标会自动推出更好的学习控制效果。
- 在 OOM 率不同的方法之间，raw rank-AUC/query/runtime 平均值可以不加截尾说明地直接比较。
