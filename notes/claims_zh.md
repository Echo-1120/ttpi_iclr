# 论文结论台账

英文同步版本：`notes/claims.md`。

## 论文定位

工作标题：
面向张量链策略迭代的秩感知、分块保留模态重排序。

核心定位：
模态排序是 TTPI 中的一等算法变量，因为 TT 秩由排序诱导的序列展开决定。物理感知耦合是有用先验，但论文应把它作为秩感知排序框架中的一个信息源，而不是把它作为完整贡献本身。

安全主张：
在固定 TTPI 近似预算下，动作模态排序可以改变中间矩阵展开秩、TT-Cross 请求负担、峰值 GPU 显存和运行时间。

不要作为标题级主张：
- 物理感知排序总是更好。
- 更低的物理线性排列目标值会直接带来更高回报。
- 当前 pilot 结果已经证明策略质量提升。

## Claim C1

在固定近似预算下，模态排序会影响 TTPI 的资源消耗。

状态：部分支持。

证据：
- HM8 冒烟诊断显示，相比 Local，Random 和 BadSplit 会提高峰值显存。已上传 pilot 数字显示 Local 约为 648.7 MB，Random 约为 1485.1 MB，BadSplit 约为 1902.0 MB。
- 仅统计 TT-Cross 调用次数不足以解释现象，因为这些 smoke 变体的调用次数几乎相同。
- reward-side TT-Cross 的秩剖面会随排序变化：在已上传 pilot 中，Local 和 OppositePair 的峰值约为秩 3，而 Random 和 BadSplit 的峰值更高。

需要的实验：
- HM8 seeds 0-9，配置 state40/action50/iter30。
- HM12 和 HM16 核心基线。
- rank-memory 和 query-memory 相关性分析。
- TT-Round 前后 cross-process 瞬时秩日志。

不要声称：
- 排序总能提升控制性能。
- 在获得多种子证据之前，PAM 能提升最终策略质量。

## Claim C2

分块保留 PAM 比自由置换或错误物理拆分更鲁棒。

状态：尚未支持。

证据：
- 分块保留 PAM 的实现已经存在。

需要的实验：
- 在 HM8/HM12 上比较 Local、BadSplit、Free PAM、Block PAM、Rank-aware PAM。
- Index-permuted HardMove 压力测试。

不要声称：
- 分块约束在所有情况下都是最优的。

## Claim C3

相比单纯 LA，秩感知目标能更好预测 TTPI 显存。

状态：开放。

证据：
- smoke 数据规模太小，目前只包含四种排序。

需要的实验：
- 比较 LA、peak-cut、rank-aware proxy 和 spectral-cache 目标。
- 加入奇异谱诊断。

不要声称：
- 静态奇异谱直接等于动态 TT-Cross 秩。

## Claim C4

加权线性排列是 cut-profile 面积目标，而不是直接的 peak-rank 目标。

状态：理论上支持；经验支持待验证。

证据：
- 对加权 cut cost \(C_k^W(\pi)\)，恒等式 \(J_{\mathrm{LA}}(\pi)=\sum_k C_k^W(\pi)\) 成立。
- TT 存储、TT-Round 负担和 GPU 显存相比总 cut 面积，更容易受到峰值秩或峰值 cut 负担影响。

需要的实验：
- 同时绘制 LA、peak-cut、rank-aware score 和 peak memory。
- 验证在 HM8/HM12/HM16 上，peak-cut 和 rank-aware 目标是否比 LA 更好地相关于显存。

不要声称：
- LA 是错误或无用的。它是合理的累计负担替代指标。

## Claim C5

排序只能通过有限预算近似误差或动作搜索误差影响控制性能。

状态：理论桥梁已确定；经验支持仍开放。

证据：
- 如果某个排序对应的 TT 近似具有一致 advantage 误差 \(\varepsilon_\pi\)，则它的贪心动作距离真实一步贪心 advantage 至多 \(2\varepsilon_\pi\)。
- 当前 pilot 运行的 success 全为 0，且出现相同的病态 return，因此不能支持性能提升主张。

需要的实验：
- 修复评估异常，或明确把短运行标记为诊断运行。
- 在确认诊断指标行为符合预期后，运行多种子、更长 horizon 的 HM8/HM12/HM16。

不要声称：
- 更好的资源指标会自动推出更好的学习控制效果。
