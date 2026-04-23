import numpy as np
import torch

# 简化示例：2维张量(2,2)
X_simple = np.array([[1.0, 2.0], [3.0, 4.0]])
# TT分解（秩=2，理论上可无损重构）
tt_decomp_simple = tl.tt_decompose(X_simple, rank=2)
X_tt_simple = tl.tt_to_tensor(tt_decomp_simple)

# 手动计算误差
diff = X_simple - X_tt_simple
# 手动算平方和：(1-X_tt[0,0])² + (2-X_tt[0,1])² + (3-X_tt[1,0])² + (4-X_tt[1,1])²
square_sum = np.sum(diff **2)
err_abs_manual = np.sqrt(square_sum)

print("\n===== 手动验证误差 =====")
print(f"手动计算绝对误差：{err_abs_manual:.6f}")
print(f"库函数计算绝对误差：{np.linalg.norm(diff, ord='fro'):.6f}")








