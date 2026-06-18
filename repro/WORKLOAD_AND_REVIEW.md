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

## 8. 新阶段：三类动力学瓶颈对标

用户给出的新实验叙事：

1. Hard-Move (HM)：高维平行执行器系统，物理耦合呈局部并列结构。
2. Planar Pushing：非抓取接触动力学系统，物理耦合呈强非线性、状态-动作-参数高度纠缠结构。
3. Catch-Point：低维经典控制基准，用来证明 PAM 不会产生负面影响，且低维下启动稳定。

当前小阶段目标：

- 先进入 Non-Prehensile Planar Pushing。
- 参考原 TTPI 的 `PushingTask.ipynb` 和 `pushing_dyn_explicit_double.py`。
- 加入参数增强 `alpha=[mass, friction]` 的证据链。
- 对比 Baseline `[states, params, actions]` 与 PAM/Local 排布的 TT rank / memory / domain-contraction proxy。

代码事实核查：

- `PushingTask.ipynb` 中 state 为 6 个 mode：
  - `[slider_x, slider_y, slider_theta, pusher_x, pusher_y, current_face]`
- 当前仓库动作实际为 3 个 mode：
  - `[next_face, vx, vy]`
- 用户描述里的 `a in R^4` 与当前 notebook 接口不一致；当前实现先以仓库可运行接口为准。
- `pushing_dyn_explicit_double.py` 里接触摩擦 `u_ps=0.3` 会进入 `gama_t/gama_b` 和接触 cone；地面摩擦 `u_gs=0.35` 当前未在显式动力学公式中实际使用。
- 当前显式动力学没有质量项；若要严谨使用 `mass in [0.2, 2.0] kg`，需要扩展动力学或作为 reward/control-effort/robustness proxy 的参数。不能在未改模型前声称“质量参数已经进入真实动力学”。

本阶段口径：

- 先完成可复查的 Planar Pushing PAM rank-proxy 脚本，验证参数增强后一阶 reward/contact proxy 在不同 mode ordering 下的 TT rank 差异。
- 完整“PAM-RTTPI 训练并在未知摩擦下 100% 成功”尚未完成；必须等多 seed 训练和 rollout 结果支撑后才能写入论文结论。

### 8.1 Planar Pushing rank-proxy smoke

新增脚本：

- `repro/scripts/run_planar_pushing_pam_proxy.py`

新增产物：

- `repro/results/planar_pushing_pam_proxy_smoke_seed0_score.json`
- `repro/results/planar_pushing_pam_proxy_smoke_seed0_score.summary.csv`
- `repro/figures/planar_pushing/planar_pushing_pam_proxy_smoke_seed0_score_rank_proxy.png`

脚本做的事情：

- 构造参数增强 mode：
  - state: `[slider_x, slider_y, theta, pusher_x, pusher_y, current_face]`
  - params: `[mass, friction]`
  - action: `[next_face, vx, vy]`
- 复刻当前显式推物动力学中的接触 cone：
  - `friction` 进入 `gamma_t/gamma_b` 与 sticking/sliding mode。
  - `pusher_y` 也进入 `gamma_t/gamma_b`，所以它必须被视作接触局部邻域的一部分。
  - `mass` 目前只进入 effort proxy；当前仓库的 quasi-static pushing dynamics 没有 inertial mass 项。
- 对同一个 one-step pushing score 做 TT-Cross，只改变 mode ordering。
- 2026-06-17 修正：
  - `face_to_signed()` 原先误把 face `0` 映射为 `-2`。
  - 已改为查表 `[0,1,2,3] -> [-1,0,1,2]`，与 `pushing_dyn_explicit_double.py` 的实际表达式一致。
  - 以下 smoke 和 3-seed 结果均为修正后重跑结果。

smoke 命令：

```bash
/home/s110/miniconda3/envs/tt_5080/bin/python \
  repro/scripts/run_planar_pushing_pam_proxy.py --preset smoke --device cpu --seed 0
```

smoke 结果：

```text
baseline_spa:      max_rank=13, mean_rank=8.80, storage=4367
pam_local:         max_rank=12, mean_rank=7.00, storage=2930
pam_contact_local: max_rank=12, mean_rank=7.20, storage=3124
pam_theta_contact: max_rank=11, mean_rank=6.50, storage=2680
bad_locality_only: max_rank=13, mean_rank=9.00, storage=6063
bad_split:         max_rank=13, mean_rank=10.80, storage=6950
face_local_only:   max_rank=12, mean_rank=7.70, storage=4100
```

当前可写结论：

- `pam_local` 已调整为更符合参数-物理耦合的局部排布：`[current_face, next_face, theta, mass, pusher_x, pusher_y, friction, vx, vy]`。
- 当前显式公式表明 `pusher_y` 同样是摩擦 cone 的局部耦合变量，因此只把 `[theta, friction, vx, vy]` 放近还不够。
- 把 contact-local block 放近后，rank proxy 明显下降：
  - `pam_theta_contact` max rank 从 baseline 的 13 降到 11。
  - `pam_theta_contact` storage 从 baseline 的 4367 降到 2680。
  - 新增负面对照 `bad_locality_only` storage 上升到 6063，`bad_split` 上升到 6950。
- 这能作为 Planar Pushing PAM 证据链入口，但还不是 full TTPI 控制成功率实验。

下一步：

- 跑 `--preset proxy`，至少 seed=0,1,2。
- 将 `pam_theta_contact` 作为当前 Planar Pushing 的候选 Local ordering。
- 若进入 full TTPI 控制训练，需要实现 state/action/param reorder wrapper 或扩展 TTPI，使跨 state-action 的 mode interleaving 不破坏 `policy_ttgo` 的 state/action 切片假设。

### 8.2 Planar Pushing rank-proxy 3-seed

新增聚合脚本：

- `repro/scripts/plot_planar_pushing_pam_proxy.py`

新增聚合产物：

- `repro/results/planar_pushing_pam_proxy_3seed_summary.csv`
- `repro/figures/planar_pushing/planar_pushing_pam_proxy_3seed_max_rank.png`
- `repro/figures/planar_pushing/planar_pushing_pam_proxy_3seed_mean_rank.png`
- `repro/figures/planar_pushing/planar_pushing_pam_proxy_3seed_storage.png`

正式 proxy 命令：

```bash
/home/s110/miniconda3/envs/tt_5080/bin/python \
  repro/scripts/run_planar_pushing_pam_proxy.py --preset proxy --device cpu --seed 0
/home/s110/miniconda3/envs/tt_5080/bin/python \
  repro/scripts/run_planar_pushing_pam_proxy.py --preset proxy --device cpu --seed 1
/home/s110/miniconda3/envs/tt_5080/bin/python \
  repro/scripts/run_planar_pushing_pam_proxy.py --preset proxy --device cpu --seed 2
/home/s110/miniconda3/envs/tt_5080/bin/python \
  repro/scripts/plot_planar_pushing_pam_proxy.py
```

3-seed 汇总：

```text
baseline_spa:      max_rank=26.7±0.6,  mean_rank=14.83±0.21, storage=17701.0±521.0, ratio=1.000
pam_local:         max_rank=16.7±0.6,  mean_rank=8.00±0.10,  storage=5845.0±190.2,  ratio=0.330
pam_contact_local: max_rank=17.0±1.0,  mean_rank=8.10±0.10,  storage=5933.0±161.0,  ratio=0.336
pam_theta_contact: max_rank=16.0±0.0,  mean_rank=7.40±0.00,  storage=5240.0±0.0,    ratio=0.296
bad_locality_only: max_rank=28.7±0.6,  mean_rank=16.47±0.40, storage=34710.3±1700.9, ratio=1.963
bad_split:         max_rank=29.0±0.0,  mean_rank=19.50±0.10, storage=35605.7±397.1,  ratio=2.013
face_local_only:   max_rank=26.3±0.6,  mean_rank=13.37±0.15, storage=22464.7±475.9,  ratio=1.269
```

当前可以写成论文证据的结论：

- 在参数增强的 Planar Pushing one-step proxy 上，只有把 face switching、姿态、接触点、摩擦和推动速度放成连续 contact-local block，才显著降低 TT rank/storage。
- `pam_theta_contact` 相比 baseline `[states, params, actions]`：
  - max rank 降至 60.0%。
  - TT coefficients/storage 降至 29.6%。
- 修正后的 `pam_local` 也有效：
  - max rank 降至 62.5%。
  - TT coefficients/storage 降至 33.0%。
- 新增负面对照 `bad_locality_only` 把参数放链条最左、动作放中间、状态/接触变量放最右：
  - storage 达到 baseline 的 196.3%。
  - 这比随机打散更直接验证“参数-状态-动作强耦合被拉开会导致 TT 链条膨胀”。
- `bad_split` storage 达到 baseline 的 201.3%，说明 rank 分布沿链条持续膨胀。
- `face_local_only` 只保留 face 相关局部性但割裂 contact cone，storage 仍高于 baseline，支持“平面推物的瓶颈不是 face switching 单独造成，而是 contact nonlinear coupling”。

仍不能写的结论：

- 不能说已经完成未知摩擦下 100% 推物成功率。
- 不能说 mass 已进入真实 quasi-static pushing dynamics；当前 mass 只进入 effort proxy。
- 不能说当前结果是完整 PAM-RTTPI controller；它是 ordering/rank/domain-contraction proxy。

### 8.3 Full TTPI 训练路径判断

已核查 `ttpi.py`：

- `TTPI.domain_state_action = domain_state + domain_action`。
- `get_reward_model()`、`get_advantage_from_value()` 等都通过：
  - `state = state_action[:, :self.dim_state]`
  - `action = state_action[:, self.dim_state:]`
  进行切片。
- `policy_ttgo()` 依赖 `deterministic_top_k()`，同样假设前 `dim_state` 个 mode 是 condition state，后续 mode 是待优化 action。

因此：

- 完整跨 state/param/action 的 arbitrary interleaving 不能直接塞进现有 `TTPI` 而不改核心采样逻辑。
- 短期可靠路径：
  1. 做 parameter-augmented robust TTPI：`state_aug=[s, alpha]`，`forward_aug=[f(s,a;alpha), alpha]`。
  2. 在 state block 内和 action block 内做保守重排，确保 `policy_ttgo` 的条件采样仍正确。
  3. 把跨 block 的 `pam_theta_contact` 保留为 rank/domain-contraction proxy 证据。
- 中期完整路径：
  - 扩展 TTPI，使 condition modes 不再要求是前缀，并支持任意 mode permutation 下的 conditional action optimization。

### 8.4 Parameter-Augmented Planar Pushing TTPI smoke

新增脚本：

- `repro/scripts/run_planar_pushing_augmented_ttpi.py`

新增产物：

- `repro/results/planar_pushing_augmented_ttpi_smoke_seed0.json`

脚本定位：

- 这是 full TTPI 训练入口的 smoke test。
- 它保留 `domain_state + domain_action`，只在 state block 内比较：
  - `baseline`: `[slider_x, slider_y, theta, pusher_x, pusher_y, current_face, mass, friction]`
  - `contact_state`: `[slider_x, slider_y, current_face, theta, pusher_x, pusher_y, friction, mass]`
- `forward_aug(state_aug, action)` 会输出 `[next_base_state, mass, friction]`，参数持久透传。
- `friction` 进入接触动力学；`mass` 仍只进入 effort proxy。

smoke 命令：

```bash
/home/s110/miniconda3/envs/tt_5080/bin/python \
  repro/scripts/run_planar_pushing_augmented_ttpi.py --preset smoke --device cpu --seed 0
```

smoke 结果：

```text
baseline:
  status=OK, time=4.335s
  final success=0.000, final_pos_mean=0.1028, final_theta_abs_mean=1.1275
  Ar_max=5, Vr_max=5, Pr_max=7

contact_state:
  status=OK, time=4.443s
  final success=0.000, final_pos_mean=0.1039, final_theta_abs_mean=0.8585
  Ar_max=5, Vr_max=4, Pr_max=7
```

解释：

- 这证明 parameter-augmented Planar Pushing 已经能进入 TTPI reward TT、policy initialization、一次 value/advantage update 和 rollout evaluation。
- 这个 smoke 的 horizon 只有 0.25s、训练只有 1 iteration，成功率为 0 不代表失败；它只验证 pipeline。
- 运行中发现 TTPI 工具函数 `get_tt_max(... n_samples=100)` 对很小离散网格有 top-k 假设，因此 smoke preset 把前两个 state mode 网格设为 12，避免 `selected index k out of range`。

下一步：

- 跑 `--preset pilot --device cuda`，把训练迭代提高到 5~20，观察 `contact_state` 是否在同等预算下改善 final distance / orientation。
- 如要证明未知摩擦成功率，需要固定训练摩擦网格和测试 off-grid 摩擦采样，并使用更长 horizon。

## 9. 新阶段：Catch-Point 参数增强低维基准

用户给出的任务二叙事：

- Catch-Point (CP) 是低维混合动作空间基准。
- 状态 `s in R^4`：系统质心二维位置与速度。
- 动作 `a in R^2`：1 个离散足迹位置，1 个连续推力。
- 参数增强 `alpha in R^1`：外部风力扰动系数或系统延迟。
- 目标：证明 PAM/参数增强不会在低维任务中带来负面影响，并能快速收敛；用户期望训练时间低于 5 分钟并达到 100% 成功率。

现有代码核查：

- `repro/catch_point.py` 已有一个简化 CP 环境：
  - state: `[x, y, vx, vy]`
  - action: `[heading, move_flag]`
  - 其中 `heading` 连续，`move_flag` 离散 `{0,1}`。
- `repro/scripts/run_catch_point.py` 已能训练旧 CP baseline。
- `repro/BASELINE.md` 记录旧 CP baseline：
  - `n_state=100, n_action=100, n_iter=50`
  - seed 0: `Best S=1.00, Best mu=1.00, Train Time=47s`

本阶段口径：

- 旧 CP baseline 可以作为“低维 TTPI 已能快速收敛”的历史证据。
- 用户新任务要求的是参数增强 CP，因此需要新增独立脚本，不直接改旧 baseline。
- 新脚本应采用 `state_aug=[x, y, vx, vy, wind]`，并让 `wind` 在 forward model 中持久透传。
- 动作应改为更贴近任务描述的 `[footstep_id, force]`：
  - `footstep_id` 是离散方向/足迹选择。
  - `force` 是连续推力幅值。

### 9.1 Augmented Catch-Point smoke

新增脚本：

- `repro/scripts/run_catch_point_augmented.py`

新增产物：

- `repro/results/catch_point_augmented_smoke_seed0.json`
- `repro/results/catch_point_augmented_smoke_seed0.summary.csv`
- `repro/figures/catchpoint_augmented/catch_point_augmented_smoke_baseline_seed0.png`
- `repro/figures/catchpoint_augmented/catch_point_augmented_smoke_pam_local_seed0.png`
- `repro/figures/catchpoint_augmented/catch_point_augmented_smoke_bad_locality_seed0.png`

脚本建模：

- state_aug:
  - `[x, y, vx, vy, wind]`
- action:
  - `[footstep_id, force]`
  - `footstep_id` 是离散方向选择。
  - `force` 是连续推力幅值。
- wind:
  - 作为持久参数透传。
  - 在动力学中进入 x 方向加速度。
  - 训练网格范围 `[-0.25, 0.25]`，off-grid 测试范围 `[-0.35, 0.35]`。

ordering：

- `baseline`: `[x, y, vx, vy, wind]`
- `pam_local`: `[x, vx, wind, y, vy]`
- `bad_locality`: `[wind, y, vy, x, vx]`

smoke 命令：

```bash
/home/s110/miniconda3/envs/tt_5080/bin/python \
  repro/scripts/run_catch_point_augmented.py --preset smoke --device cpu --seed 0
```

smoke 结果：

```text
baseline:
  status=OK, time=10.14s
  nominal_success=0.000, offgrid_success=0.167
  offgrid S×mu=0.148, final_dist_offgrid=0.151
  Ar_max=8, Pr_max=10

pam_local:
  status=OK, time=9.29s
  nominal_success=0.250, offgrid_success=0.250
  offgrid S×mu=0.212, final_dist_offgrid=0.149
  Ar_max=8, Pr_max=10

bad_locality:
  status=OK, time=9.38s
  nominal_success=0.083, offgrid_success=0.083
  offgrid S×mu=0.073, final_dist_offgrid=0.159
  Ar_max=8, Pr_max=10
```

当前结论：

- 参数增强 CP 的 7-mode TTPI pipeline 已跑通。
- `pam_local` 在 smoke 预算下优于 baseline 与 `bad_locality`，但 smoke 只有 2 次 policy iteration、12 条测试轨迹，不能写成正式成功率结论。
- 当前结果尚未达到用户目标的 100% 成功率；需要跑 `--preset pilot` 或进一步调 reward/action scale。
- 旧 CP baseline 仍可作为“非参数增强 CP 可在 47s 达到 100%”的历史证据，但不能替代本次 wind-augmented CP。

工具修复：

- 修复 `tt_utils.deterministic_top_k()` 在第一个 TT mode 的离散点数小于默认 `n_samples=100` 时的候选重复 bug。
- 原实现只重复了 `samples_idx`，没有同步重复 `p_cum`，导致后续 site 仍只有 `n_site_0 * n_site` 个候选却调用 `topk(k=100)`，在 `bad_locality=[wind,...]` 这种小参数 mode 置于链首时触发 `selected index k out of range`。
- 修复后 `bad_locality` 能完整跑通，这对负面对照是必要条件。

### 9.2 Augmented Catch-Point PAM pilot

pilot 命令：

```bash
/home/s110/miniconda3/envs/tt_5080/bin/python \
  repro/scripts/run_catch_point_augmented.py \
  --preset pilot --device cpu --seed 0 \
  --cases pam_local --n-iter 8 --callback-freq 2 \
  --n-test 30 --horizon 4.0 \
  --output-stem catch_point_augmented_pilot_pam_local_seed0_iter8
```

新增产物：

- `repro/results/catch_point_augmented_pilot_pam_local_seed0_iter8.json`
- `repro/results/catch_point_augmented_pilot_pam_local_seed0_iter8.summary.csv`
- `repro/figures/catchpoint_augmented/catch_point_augmented_pilot_pam_local_seed0.png`

pilot 结果：

```text
pam_local:
  status=OK
  train_time=64.86s
  nominal_success=1.000
  offgrid_success=1.000
  nominal S×mu=0.665
  offgrid S×mu=0.682
  final_dist_offgrid=0.0137
  Ar_max=16
  Vr_max=6
  Pr_max=18
```

解释：

- 这已经满足用户给出的“低维 CP 训练时间小于 5 分钟并达到 100% 成功率”的 pilot 证据。
- 该结果是在 CPU 上完成，训练时间约 65 秒；CUDA 应该还有余量。
- off-grid wind 测试范围比训练范围更宽：
  - training wind: `[-0.25, 0.25]`
  - off-grid test wind: `[-0.35, 0.35]`
- 当前仍是 seed=0 pilot；正式论文/表格建议补 `seed=1,2`，并固定同样的 30 或 50 条测试轨迹。

当前可以写的结论：

- 参数增强 CP 的 `pam_local=[x, vx, wind, y, vy]` 能在低维 7-mode 设置下快速稳定收敛。
- 在 seed=0 pilot 中，PAM-augmented CP 达到 nominal/off-grid wind 双 100% 成功率。
- 这支持“PAM 在低维任务中不会带来负面影响，并保留快速启动稳定性”。

仍不能写的结论：

- 不能把 seed=0 pilot 写成完整统计结论。
- 不能直接和旧 `heading+move_flag` CP baseline 混为同一个 action model；新脚本使用的是 `[footstep_id, force]`。

### 9.3 Augmented Catch-Point CUDA pilot, all orderings

另有一组 CUDA pilot 全 ordering 结果：

- `repro/results/catch_point_augmented_pilot_seed0.json`
- `repro/results/catch_point_augmented_pilot_seed0.summary.csv`
- `repro/figures/catchpoint_augmented/catch_point_augmented_pilot_baseline_seed0.png`
- `repro/figures/catchpoint_augmented/catch_point_augmented_pilot_pam_local_seed0.png`
- `repro/figures/catchpoint_augmented/catch_point_augmented_pilot_bad_locality_seed0.png`

配置：

```text
preset=pilot, device=cuda, seed=0
n_state=35, n_velocity=21, n_param=7
n_footsteps=16, n_force=35
n_test=50, horizon=6.0
early_stop_success=0.95, early_stop_after_callback=5
```

结果：

```text
baseline:
  status=EARLY_STOP, time=14.63s
  nominal_success=1.000, offgrid_success=1.000
  offgrid S×mu=0.664, final_dist_offgrid=0.0161
  Ar_max=17, Vr_max=11, Pr_max=19

pam_local:
  status=EARLY_STOP, time=11.12s
  nominal_success=1.000, offgrid_success=1.000
  offgrid S×mu=0.660, final_dist_offgrid=0.0171
  Ar_max=15, Vr_max=6, Pr_max=17

bad_locality:
  status=EARLY_STOP, time=10.88s
  nominal_success=1.000, offgrid_success=1.000
  offgrid S×mu=0.656, final_dist_offgrid=0.0190
  Ar_max=15, Vr_max=7, Pr_max=17
```

解释：

- 这组 CUDA pilot 显示低维 CP 的三种 ordering 都很容易达到 100% 成功率，符合“低维诊断沙盒”的预期。
- `pam_local` 相比 baseline 在 rank 上更低：
  - `Ar_max`: 15 vs 17
  - `Vr_max`: 6 vs 11
  - `Pr_max`: 17 vs 19
- 因为 CP 维度很低，`bad_locality` 也能达到 100%，所以 CP 不适合作为强 ordering separation 的主证据；它更适合作为“PAM 无负面影响、启动稳定、参数增强鲁棒”的 sanity check。
