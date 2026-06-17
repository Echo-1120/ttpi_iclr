# PAM Ordering 工作量、代码审查与理论验证

## 1. 当前产物范围

本文件对应当前 PAM / Physics-aware mode ordering 阶段，不覆盖后续完整 Parameter-Augmented Robust TTPI。

已归档产物：

| 类别 | 路径 |
|---|---|
| 全量汇总 CSV | `repro/results/pam_full_results.csv` |
| HM8 多 seed 原始结果 | `repro/results/pam_v4_full.json` |
| HM8/HM12/HM16 scaling 原始结果 | `repro/results/pam_scaling.json` |
| HM12/HM16 Local extra seeds | `repro/results/local_extra_seeds.json` |
| LaTeX 主表 | `repro/results/main_table.tex` |
| 实验小节草稿 | `repro/EXPERIMENT_SECTION.md` |
| HM8 图 | `repro/figures/pam/fig1_rank_bars.png` 等 |
| Scaling 图 | `repro/figures/pam/scaling_rank.png`, `scaling_peak_memory.png`, `scaling_performance.png` |
| Local 3-seed scaling 图 | `repro/figures/pam/local_3seed_peak_memory.png`, `local_3seed_max_rank.png`, `local_3seed_success_tradeoff.png` |
| Local 3-seed summary | `repro/results/local_3seed_scaling_summary.csv` |
| DPRP/LaX 脚本 | `repro/scripts/dprp_lax_ablation.py` |
| DPRP/LaX 初步结果 | `repro/results/dprp_lax_smoke_seed0.json`, `repro/results/dprp_lax_paper_local_lax_v2_seed0.json` |
| PointMassVelocity 对比脚本 | `repro/scripts/run_pointmass_velocity_comparison.py` |
| PointMassVelocity 对比结果 | `repro/results/pointmass_velocity_notebook_seed0.json`, `repro/results/pointmass_velocity_notebook_seed0.summary.csv` |
| PointMassVelocity 轨迹图 | `repro/figures/pointmass_velocity/pointmass_velocity_notebook_seed0_trajectories.png` |
| 归档脚本 | `repro/scripts/pam_v4_experiment.py`, `pam_scaling.py`, `run_hm_lowbatch.py`, `run_local_seeds.py`, `plot_pam_scaling.py` |
| 新增探索脚本 | `repro/scripts/auto_ordering.py`, `repro/scripts/mla_block_experiment.py`, `repro/scripts/pmi_ordering.py`, `repro/scripts/rank_eps_sensitivity.py` |

## 2. 实际完成的工作量

### 2.1 复现与显存控制

完成内容：

- 在 RTX 5080 16GB 环境下摸清 TTPI 的主要显存压力。
- 将训练稳定到可重复运行的配置：
  - `n_iter_v=1`
  - `rmax_v/rmax_a=60`
  - `max_batch_v=5000`
  - `max_batch_a=20000`
  - policy iteration 后清理 CUDA cache 和 Python GC
- 形成 hardware-constrained scaling 设置：
  - HM8: `NS=40, NA=50`
  - HM12: `NS=30, NA=40`
  - HM16: `NS=25, NA=30`

工作量性质：

这是复现实验工程量，不是单纯跑脚本。它解决的是“在 16GB 显存下如何得到可比较结果”的问题。

### 2.2 Mode ordering 实验设计

完成内容：

- 设计并实现四类 ordering：
  - `Local`
  - `BadSplit`
  - `Random`
  - `OppositePair`
- 明确了各自角色：
  - `Local` 是原始/自然物理局部 baseline。
  - `BadSplit` 是结构破坏 negative control。
  - `Random` 是无结构对照。
  - `OppositePair` 是保留局部 pair 的 ablation。

工作量性质：

这是当前工作的核心实验设计部分。它让实验具备 baseline、negative control 和 ablation。

### 2.3 数据规模

当前 CSV 中共有 25 条记录：

- HM8:
  - Local 5 seeds
  - BadSplit 5 seeds
  - Random 5 seeds
- HM12:
  - Local 3 seeds
  - BadSplit 1 seed
  - Random 1 seed
- HM16:
  - Local 3 seeds
  - BadSplit 1 seed
  - Random 1 seed

其中 BadSplit/Random 在 HM12/HM16 作为 stress/negative control，没有补齐多 seed。

这个选择可以接受，但论文中必须写清楚：

> BadSplit and Random at larger scales are used as stress cases because they already reach rank saturation or OOM.

不能写成完整统计显著性对照。

### 2.4 图表与写作素材

完成内容：

- HM8 rank/memory/time/pareto 四张图。
- scaling rank/memory/performance 三张图。
- Local-only 3-seed scaling 三张图：
  - `local_3seed_peak_memory.png`
  - `local_3seed_max_rank.png`
  - `local_3seed_success_tradeoff.png`
- LaTeX 主表。
- 实验小节草稿。

工作量性质：

已经从“结果文件”推进到“可写论文实验小节”的阶段。

### 2.5 新增 Step 1：Local 3-seed scaling curve

用户要求：

```text
n_act = 8, 12, 16 下 Local 的 seed=0,1,2，导出 Peak Memory、Max Rank、S×mu 三张带 std 阴影的折线图。
```

完成状态：已完成。

数据来源：

- `repro/results/pam_full_results.csv`

新增脚本：

- `repro/scripts/plot_local_3seed_scaling.py`

新增结果：

- `repro/results/local_3seed_scaling_summary.csv`
- `repro/figures/pam/local_3seed_peak_memory.png`
- `repro/figures/pam/local_3seed_max_rank.png`
- `repro/figures/pam/local_3seed_success_tradeoff.png`

当前 summary：

```text
HM8  Local: S×mu=0.620±0.066, MaxRank=9.0±0.0,  Peak=2.86±1.33 GB
HM12 Local: S×mu=0.517±0.036, MaxRank=14.0±1.0, Peak=5.86±0.00 GB
HM16 Local: S×mu=0.567±0.041, MaxRank=17.0±3.6, Peak=5.87±0.00 GB
```

### 2.6 新增 Step 2：DPRP / LaX 实验入口

用户要求：

```text
动态秩 DPRP 与 LaX 消融：
1. 对比带 LaX vs 不带 LaX 的控制精度，证明 LaX 能以极低显存开销挽回降低网格带来的精度损失。
2. 对比 Local 下 DPRP 将平均秩压低至 5~8，而 Random 下由于纠缠不衰减导致 DPRP 失效。
```

完成状态：实验入口已完成，正式论文证据尚未完成。

新增脚本：

- `repro/scripts/dprp_lax_ablation.py`

脚本能力：

- `--preset smoke`：快速验证 DPRP/LaX hook、评估和导出。
- `--preset paper`：使用较接近论文预算的 HM8 low-grid 设置。
- `local_lax`：同一训练过程中同时评估 No-LaX 与 LaX。
- `local_dprp` / `random_dprp`：训练时做 validation-aware dynamic TT rounding，并记录平均/最大 rank。

已运行结果：

- `repro/results/dprp_lax_smoke_seed0.json`
- `repro/results/dprp_lax_paper_local_lax_v2_seed0.json`

当前 LaX 单 seed 结果：

```text
HM8 Local low-grid, seed=0, NS=30, NA=25, n_iter=30
Best No-LaX S×mu: 0.6297
Best LaX    S×mu: 0.6155
Best paired same-callback LaX delta: +0.1355 at cb19
LaX extra memory: ~0.000007 GB
```

解释：

- 当前实现能证明 LaX 的额外显存开销极低。
- 当前单 seed 结果显示 LaX 在同一 callback/model 上能改善控制指标，但全程 best-over-training 仍略低于 No-LaX。
- 因此还不能写成“已证明 LaX 挽回降低网格带来的精度损失”。需要至少补 seed=1,2，并固定比较口径（same-callback vs best-over-training）。

当前 DPRP smoke 结果：

```text
smoke setting: NS=18, NA=16, n_iter=2
Local DPRP 和 Random DPRP 都能跑通并输出 rank/pruning events。
```

解释：

- smoke 只验证代码路径，不验证论文 claim。
- 要证明 “Local 平均秩 5~8，Random DPRP 失效”，需要跑 `--preset paper --cases local_dprp,random_dprp`，最好至少 seed=0,1,2。

### 2.7 新增 PointMassVelocity 对比实验

用户要求：

```text
参考原文论文和 PointMassVelocity.ipynb，先仿照这个完成对比试验。
```

完成状态：已完成单 seed notebook-like 对比。

新增脚本：

- `repro/scripts/run_pointmass_velocity_comparison.py`

脚本内容：

- 将 `PointMassVelocity.ipynb` 脚本化：
  - 2D velocity-control point mass
  - obstacle center `(0, -0.4)`
  - obstacle radius `0.2`
  - `n_state=50`
  - `n_action=50`
  - `n_iter=200`
  - `callback_freq=20`
- 添加轻量 baseline：
  - `straight`
  - `potential`
  - `random`
  - `zero`

注意：

原文 ICLR 论文没有给 PointMassVelocity 的正式 baseline 表；它在附录/视频中作为 continuous-control demo。因此这里的 baseline 是 reproduction sanity check，不应写成原文 HyAR baseline 对照。

结果：

```text
TTPI:       S=0.900, mu=0.960, S×mu=0.864, collision=0.000
Straight:   S=1.000, mu=0.967, S×mu=0.967, collision=0.133
Potential:  S=0.033, mu=0.000, S×mu=0.000, collision=0.000
Random:     S=0.033, mu=0.000, S×mu=0.000, collision=0.033
Zero:       S=0.033, mu=0.000, S×mu=0.000, collision=0.000
```

解释：

- `straight` 成功率最高，但 13.3% 初始状态穿过障碍。
- TTPI 的 `S×mu` 略低于 straight，但 collision 为 0，更符合 notebook 的 obstacle-aware objective。
- `potential` 在当前手工参数下避障但几乎不达标，说明简单势场对该设置较敏感。

产物：

- `repro/results/pointmass_velocity_notebook_seed0.json`
- `repro/results/pointmass_velocity_notebook_seed0.summary.csv`
- `repro/figures/pointmass_velocity/pointmass_velocity_notebook_seed0_trajectories.png`

## 3. 代码审查结论

### 3.1 语法检查

以下脚本通过 `python3 -m py_compile`：

- `repro/scripts/pam_v4_experiment.py`
- `repro/scripts/pam_scaling.py`
- `repro/scripts/run_hm_lowbatch.py`
- `repro/scripts/run_local_seeds.py`
- `repro/scripts/plot_pam_scaling.py`
- `repro/scripts/auto_ordering.py`
- `repro/scripts/mla_block_experiment.py`
- `repro/scripts/pmi_ordering.py`
- `repro/scripts/rank_eps_sensitivity.py`

说明它们没有 Python 语法错误。

### 3.2 已处理问题 1：`auto_ordering.json` 曾不是合法 JSON

文件：

- `repro/archive/results/exploratory/auto_ordering.json`

问题：

`auto_ordering.py` 曾在保存结果时失败：

```text
TypeError: Object of type int64 is not JSON serializable
```

原因：

`generate_ordering` 的 `spectral` / `rcm` 可能返回 `numpy.int64`，`json.dump` 不能直接序列化。

影响：

- `auto_ordering.py` 已加入 JSON 类型转换。
- `repro/archive/results/exploratory/auto_ordering.json` 已替换为从日志恢复的有效 run summary。
- 该结果仍只能作为探索记录，不能作为最终论文主证据，因为 coupling estimator 主要依赖 random-action correlation 和 same-actuator manual boost。

修复建议：

在保存前递归转换 numpy 类型：

```python
def to_jsonable(x):
    if isinstance(x, dict):
        return {k: to_jsonable(v) for k, v in x.items()}
    if isinstance(x, list):
        return [to_jsonable(v) for v in x]
    if isinstance(x, tuple):
        return [to_jsonable(v) for v in x]
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, np.floating):
        return float(x)
    return x
```

并将 ordering 强制转为 Python int：

```python
order = [int(i) for i in order]
```

### 3.3 已处理问题 2：rank/eps sensitivity 不能称为 Robust TTPI

文件：

- `repro/scripts/rank_eps_sensitivity.py`
- `repro/archive/results/exploratory/rank_eps_sensitivity.json`
- legacy: `repro/archive/results/legacy/robust_ttpi_phase1.json`

问题：

原先脚本名和注释写的是 robust / parameter sensitivity，但代码实际没有引入物理参数 `alpha`，也没有做 parameter-augmented state。

它实际做的是：

```text
Local vs BadSplit under different rmax and eps settings
```

即 rank/epsilon sensitivity，不是 Parameter-Augmented Robust TTPI。

影响：

- 这组结果已经改按 rank/eps sensitivity 归类。
- 仍不能把它称为 robust TTPI 或 parameter-augmented robust TTPI。
- 不能把 legacy `robust_ttpi_phase1.json` 作为方案一 Phase 1 的完成证据。

真正实现参数增强后，才可以新建 `robust_ttpi.py` 或对应实验名。

### 3.4 已处理问题 3：`EXPERIMENT_SECTION.md` 中有一处结论与表格冲突

文件：

- `repro/EXPERIMENT_SECTION.md`

原问题行：

```text
Random ordering triggers OOM at n_act=8 — the smallest scale.
```

但表格中 HM8 Random 的 OOM 是 `N`。真正 OOM 的是：

```text
HM12 BadSplit
HM12 Random
```

已修正为：

改为：

```text
Random and BadSplit both complete at HM8 but already show severe rank and memory inflation. At HM12, BadSplit and Random trigger OOM, while Local remains executable.
```

### 3.5 中等问题 1：异常处理过宽

多个脚本有如下模式：

```python
except Exception as e:
    print(f'Error:{e}', flush=True)
```

或在早期 sensitivity 脚本中：

```python
except Exception as e:
    pass
```

问题：

- 会把真实 bug 当作训练失败继续记录。
- `pass` 会吞掉错误，导致 status 可能变成 `cb-1` 或错误结果。

建议：

至少记录 exception type 和 message 到结果 JSON：

```python
error_msg = None
except Exception as e:
    error_msg = repr(e)
    print(f"Error: {error_msg}", flush=True)
```

并在 result 中加入：

```python
"error": error_msg
```

### 3.6 中等问题 2：`auto_ordering.py` 的耦合估计仍然偏弱

当前 auto ordering 使用：

```text
random actions -> absolute correlation -> manually boost same-actuator pairs
```

问题：

- 随机独立动作之间的 correlation 本身不反映动力学耦合。
- 主要有效信息来自手工 boost `(acc_i, sw_i)`。
- 这还不能证明自动发现了物理结构，只能说“带物理先验的 ordering generator 可以生成结构排序”。

更严谨的下一版：

1. 用解析 coupling graph：
   - `(acc_i, sw_i)` 权重高。
   - 相邻 actuator 或对称 actuator 权重可选。
2. 或用 finite-difference sensitivity：
   - 改变某个 action variable，测量 `next_state` / reward 的变化。
3. 记录 ordering objective：
   - `sum_ij w_ij |pi(i)-pi(j)|`
   - 和 Local/BadSplit/Random 比较 objective 值。

### 3.7 已处理问题 3：LaTeX 表中 `±` 字符

文件：

- `repro/results/main_table.tex`

问题：

表格此前混用了 Unicode `±` 和 LaTeX `\pm`：

```tex
$0.62 ± 0.05$
```

已统一改为：

```tex
$0.62 \pm 0.05$
```

此外，`\ding{55}` 需要 `pifont` 包。若不想增加依赖，建议用 `Yes/No` 或 `OOM`。

## 4. 理论验证

### 4.1 TT rank 与 unfolding cut

对于一个 D 阶张量：

```text
T(i_1, ..., i_D)
```

第 k 个 TT rank 满足：

```text
r_k = rank(T_[k])
```

其中 `T_[k]` 是 canonical unfolding：

```text
T_[k]((i_1, ..., i_k), (i_{k+1}, ..., i_D))
```

物理意义：

```text
r_k 表示 TT chain 的第 k 个切分处，需要跨 cut 传递多少依赖信息。
```

如果强耦合变量被放在 cut 两侧，则 `T_[k]` 必须编码跨 cut 依赖，rank 会升高。

### 4.2 Local 为什么低 rank

Local ordering：

```text
[acc_0, sw_0, acc_1, sw_1, ..., acc_N, sw_N]
```

每个 actuator 的连续控制和离散开关相邻。TT cut 通常只会切断局部少量耦合。

实验对应：

```text
HM8  Local: Ar=7,  Pr=9
HM12 Local: Ar≈12, Pr≈14
HM16 Local: Ar≈15, Pr≈17
```

这与理论预期一致：如果物理依赖局部化，unfolding rank 增长慢。

### 4.3 BadSplit 为什么 rank 膨胀

BadSplit ordering：

```text
[acc_0, acc_1, ..., acc_N, sw_0, sw_1, ..., sw_N]
```

中间 cut 将所有 acceleration 放在左侧，将所有 switch 放在右侧。每个 `(acc_i, sw_i)` 耦合都被跨 cut 分离。

理论预期：

```text
中间 unfolding 需要同时编码所有 acc-switch 对应关系，rank 会快速增长。
```

实验对应：

```text
HM8  BadSplit: Ar≈49, Pr≈51
HM12 BadSplit: Ar=54, Pr=56, OOM
HM16 BadSplit: Ar=60, Pr=62
```

这与理论预期一致。

### 4.4 Random 为什么不稳定

Random ordering 会随机切断不同 actuator 的局部 coupling。它不一定总是最坏，但期望上会把强耦合变量分散到 TT chain 中。

实验对应：

```text
HM8  Random: Ar≈50, Pr≈52
HM12 Random: Ar=60, Pr=62, OOM
HM16 Random: Ar=60, Pr=62
```

这说明随机排列容易接近 rank budget。

### 4.5 OppositePair 的解释

OppositePair 保留 `(acc_i, sw_i)` pair，因此 rank 与 Local 接近。

这说明：

```text
低 rank 的关键不是唯一固定排列，而是保留强耦合局部变量的邻接关系。
```

不能解释为 OppositePair 优于 Local。

### 4.6 MLA 目标函数

可以将 physics-aware ordering 表述为 weighted minimum linear arrangement：

```text
min_pi sum_{i,j} w_ij |pi(i) - pi(j)|
```

其中：

- `w_ij` 是变量 i 和 j 的物理耦合强度。
- `pi(i)` 是变量 i 在 TT chain 中的位置。

理论含义：

```text
强耦合变量距离越远，跨 cut 传播的信息越多，TT rank 越容易升高。
```

当前实验已经验证该理论的负面对照：

- Local: 高权重 pair 距离为 1。
- BadSplit: 高权重 pair 距离约为 `n_act`。
- Random: 高权重 pair 距离随机，平均较大。

## 5. 当前能成立的结论

可以成立：

1. TTPI 的 action mode ordering 会显著影响 TT rank、显存和训练时间。
2. 保留 `(acc_i, sw_i)` 物理局部耦合的 ordering 可以压制 rank。
3. BadSplit/Random 会导致 rank 膨胀、rank saturation，部分设置触发 OOM。
4. Local 是 HardMove 上的原始/自然 baseline，不是新方法本身。
5. 当前实验支撑“physics-aware ordering principle”，而不是完整自动排序算法。

不能成立：

1. 不能说 OppositePair 明显优于 Local。
2. 不能说 BadSplit/Random 是原论文 baseline。
3. 不能说已经完成 Parameter-Augmented Robust TTPI。
4. 不能说 auto ordering 已经稳定成为最终方法；`auto_ordering.json` 现在只是有效探索摘要，且 coupling 主要依赖手工 boost。
5. 不能说 Local 永远不会 OOM，只能说在当前 hardware-constrained setting 下 Local 更可执行。
6. 不能说 DPRP/LaX 已经完成论文级证明；目前只有脚本、smoke 和 LaX 单 seed pilot。

## 6. 需要优先改进的地方

### 必须改

1. 保持 `auto_ordering.py` 为探索脚本，若要作为论文证据需要重跑并补更可靠的 coupling estimator。
2. 保持 `rank_eps_sensitivity.py` 与 Robust TTPI 叙事分离。
3. 修正或重跑 `mla_block_experiment.py`，确保 BadSplit/Random 的 block MLA 不被误写成 Local 的 block cost。
4. PMI 结果只能作为 exploratory metric，不写成“PMI 越高越好”。

### 应该改

1. 给每个脚本增加命令行参数，而不是硬编码 `N_ACT`、`N_ITER`、`NS/NA`。
2. 在所有结果 JSON 中保存完整配置：
   - `NS`
   - `NA`
   - `rmax`
   - `max_batch`
   - `n_iter`
   - `n_test`
   - `order`
   - `a_order`
3. 避免吞掉异常，记录 `error` 字段。
4. 给 `plot_pam_scaling.py` 加图注说明 OOM X 标记含义。
5. 将 `auto_ordering.py` 的 coupling objective 值作为正式字段输出到 JSON。
6. 对 `dprp_lax_ablation.py` 补完整 paper-budget 多 seed：
   - `local_lax` seeds 1,2
   - `local_dprp` seeds 0,1,2
   - `random_dprp` seeds 0,1,2

### 后续研究

1. 真正实现 Parameter-Augmented Robust TTPI：
   - `state_aug = [s, alpha]`
   - `forward_aug(s, alpha, a) = [f(s,a;alpha), alpha]`
2. 实现解析/敏感度 coupling graph。
3. 比较自动排序和手工 Local/BadSplit/Random。
4. 做最小 domain contraction 验证。

## 7. 下一步执行顺序

推荐顺序：

1. 先完成 DPRP/LaX 正式证据：
   - 跑 `dprp_lax_ablation.py --preset paper --cases local_lax --seed 1/2`
   - 跑 `dprp_lax_ablation.py --preset paper --cases local_dprp,random_dprp --seed 0/1/2`
   - 汇总 same-callback LaX delta、best-over-training delta、rank avg/max、peak memory。
2. 再修结果与文档口径：
   - `archive/results/exploratory/auto_ordering.json`
   - `EXPERIMENT_SECTION.md`
   - `main_table.tex`
   - `pmi_experiment.json`
   - `mla_block_experiment.json`
3. 再清理代码命名：
   - 保持 `rank_eps_sensitivity.py` 与真正 Robust TTPI 分离。
4. 再做理论验证补充：
   - 输出每个 ordering 的 MLA objective。
   - 证明 objective 与 rank 膨胀方向一致。
5. 最后进入方案一 Phase 1：
   - 参数增强 state。
   - Nominal vs robust。
   - LocalAction vs BadSplitAction。
