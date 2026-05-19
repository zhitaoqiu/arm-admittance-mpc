# OSQP MPC Upgrade — From SciPy L-BFGS-B to OSQP QP

## 1. Why OSQP?

| 对比维度 | SciPy L-BFGS-B (旧) | OSQP QP (新) |
|----------|---------------------|--------------|
| 求解方法 | 通用非线性优化 | 结构化二次规划 |
| 约束处理 | 仅 box bounds | 等式 + 不等式（动力学/力矩/关节限位） |
| 求解速度 | ~50-200 ms | 预期 <5 ms |
| 实时性 | 不适合 >50 Hz | 适合实时 MPC |
| 数值稳定性 | 依赖初始猜测 | 凸 QP 全局最优 |
| 工程落地 | 原型验证 | 可直接对接 C++ OSQP |

SciPy 版本用 `minimize(method="L-BFGS-B")` 把力矩序列送入 `rollout_cost` 做无梯度优化，每步需要数十次 `rollout_cost` 调用（每次 N=12 步 Euler 积分 + FK），计算量不可控。

OSQP 将问题结构化：一次构建稀疏 QP → 求解器直接利用问题结构 → 计算量可预测。

---

## 2. QP Formulation

### 2.1 状态与控制

| 符号 | 含义 | 维度 |
|------|------|------|
| $s_k$ | 关节状态 $[q_1, q_2, \dot{q}_1, \dot{q}_2]$ | $\mathbb{R}^4$ |
| $u_k$ | 关节力矩 $[\tau_1, \tau_2]$ | $\mathbb{R}^2$ |
| $N$ | 预测步数 (horizon) | 12 |
| $dt$ | 预测步长 | 0.03 s |

### 2.2 优化变量

变量排列（分组方式）：

$$z = [s_0, s_1, \dots, s_N, u_0, u_1, \dots, u_{N-1}]$$

总维度：$(N+1) \times 4 + N \times 2 = 76$（N=12 时）

### 2.3 线性动力学

连续动力学（冻结 M, bias）：

$$M \ddot{q} + b = \tau \quad\Rightarrow\quad \ddot{q} = M^{-1}(\tau - b)$$

欧拉离散化：

$$\begin{aligned}
q_{k+1} &= q_k + dt \cdot \dot{q}_k \\
\dot{q}_{k+1} &= \dot{q}_k + dt \cdot M^{-1}(u_k - b)
\end{aligned}$$

状态空间形式 $s_{k+1} = A_d s_k + B_d u_k + c_d$：

$$A_d =
\begin{bmatrix}
I_2 & dt \cdot I_2 \\
0   & I_2
\end{bmatrix}
\quad
B_d =
\begin{bmatrix}
0 \\
dt \cdot M^{-1}
\end{bmatrix}
\quad
c_d =
\begin{bmatrix}
0 \\
-dt \cdot M^{-1} b
\end{bmatrix}$$

### 2.4 代价函数

末端位置通过雅可比在当前状态 $q_0$ 处线性化：

$$FK(q_k) \approx FK(q_0) + J(q_0) \cdot (q_k - q_0)$$

$$\begin{aligned}
\min_{z} \quad &\sum_{k=0}^{N-1} \Big(
  \|J_0 q_k + c_{\text{pose}}\|^2_{Q_x}
+ \|J_0 \dot{q}_k - \dot{x}_{\text{ref}}\|^2_{Q_{\dot{x}}}
+ \|\dot{q}_k\|^2_{Q_{\dot{q}}}
+ \|u_k\|^2_R
+ \|u_k - u_{k-1}\|^2_S
\Big) \\
&+ \|J_0 q_N + c_{\text{pose}}\|^2_{Q_{x,\text{term}}}
\end{aligned}$$

其中 $c_{\text{pose}} = FK(q_0) - J_0 q_0 - x_{\text{ref}}$ 在每次求解时固定。

### 2.5 权重配置

| 参数 | 值 | 含义 |
|------|-----|------|
| Qx | [600, 600] | 末端位置跟踪 |
| Qdx | [20, 20] | 末端速度跟踪 |
| Qdq | [1, 1] | 关节速度惩罚 |
| R | [0.01, 0.01] | 控制力矩惩罚 |
| S | [0.005, 0.005] | 力矩变化率惩罚 (Δu) |
| Qx_terminal | [1500, 1500] | 终端位置误差 |

### 2.6 约束

| 约束 | 范围 | 类型 |
|------|------|------|
| 初始状态 $s_0$ | $= [q_{\text{meas}}, \dot{q}_{\text{meas}}]$ | 等式 |
| 动力学 $s_{k+1} = A_d s_k + B_d u_k + c_d$ | $k=0..N-1$ | 等式 |
| 力矩 $u_k$ | $\in [-20, 20]$ N·m | 不等式 |
| 关节角 $q_k$ | $\in [-3.14, 3.14]$ rad | 不等式 |
| 关节速度 $\dot{q}_k$ | $\in [-10, 10]$ rad/s | 不等式 |

### 2.7 OSQP 标准形式

$$\begin{aligned}
\min_z \quad &\frac{1}{2} z^T P z + q^T z \\
\text{s.t.} \quad &l \leq A_{\text{cons}} z \leq u
\end{aligned}$$

- $P$：由权重矩阵构建的稀疏正半定矩阵
- $q$：线性项（来自参考轨迹和雅可比线性化）
- $A_{\text{cons}}$：稀疏约束矩阵，编码动力学等式和变量边界

---

## 3. Current Simplifications

| 简化项 | 说明 | 后续升级方向 |
|--------|------|-------------|
| 冻结动力学 | 每个求解周期固定 M(q) 和 bias | 连续线性化（每一步预测重新算 M） |
| 冻结雅可比 | 末端位置通过 $J(q_0)$ 线性化 | 沿预测轨迹逐步更新 J |
| 欧拉积分 | 一阶显式欧拉离散 | RK4 或零阶保持精确离散化 |
| 仿真真值状态 | 直接从 MuJoCo 读 q, dq | 加入状态估计/滤波 |
| 无传感器噪声 | 仿真环境完美信息 | 添加观测噪声模型 |

---

## 4. How to Run

### 4.1 安装依赖

```bash
pip install osqp
```

或：

```bash
pip install -r requirements.txt
```

### 4.2 运行 OSQP MPC（带求解统计 + CSV）

```bash
python main_admittance_mpc_osqp_control.py
```

仿真结束后：
- 终端打印求解时间统计
- 弹出 matplotlib 图表
- 生成 `results/osqp_mpc_solve_stats.csv`

### 4.3 运行 SciPy vs OSQP 对比

```bash
python main_compare_mpc.py
```

对比实验会依次运行两个控制器，输出对比表格和并排图表，保存 `results/compare_scipy_vs_osqp.csv`。

### 4.4 运行原版 SciPy MPC（未改动）

```bash
python main_admittance_mpc_control.py
```

---

## 5. Code Structure

```
├── kinematics.py                              # FK, Jacobian (unchanged)
├── main_admittance_mpc_control.py              # 原版 SciPy MPC (unchanged)
├── main_admittance_mpc_osqp_control.py         # NEW: OSQP MPC
│   ├── AdmittanceController                    #   (same as original)
│   ├── TaskSpaceOSQPMPC                        #   OSQP-based MPC solver
│   │   ├── get_frozen_dynamics()               #   M, bias from MuJoCo
│   │   ├── _linear_dynamics()                  #   A_d, B_d, c_d
│   │   ├── _build_qp()                         #   P, q, A_cons, l, u
│   │   └── solve()                             #   OSQP solve + timing
│   ├── plot_results()                          #   8 张图含 solve time
│   ├── print_solve_stats()                     #   终端统计输出
│   └── save_solve_stats_csv()                  #   CSV 导出
├── main_compare_mpc.py                         # NEW: SciPy vs OSQP comparison
├── results/
│   ├── osqp_mpc_solve_stats.csv                #   (generated)
│   ├── mpc_comparison_summary.csv              #   (generated — comparison table)
│   ├── solve_time_comparison.png               #   (generated)
│   ├── tracking_error_comparison.png           #   (generated)
│   └── torque_comparison.png                   #   (generated)
└── docs/
    └── OSQP_MPC_UPDATE.md                      #   This document
```

---

## 6. Experimental Results

### 6.1 Critical Bug: Factor-2 Error in P Matrix

**Problem**: OSQP uses the standard form:

$$\min_z \frac{1}{2} z^T P z + q^T z$$

For a quadratic tracking cost `||Cz + d||²_Q`, expanding:

$$(Cz + d)^T Q (Cz + d) = z^T C^T Q C z + 2 d^T Q C z + d^T Q d$$

To match OSQP's `1/2 z^T P z + q^T z` form:

$$\boxed{P = 2 C^T Q C} \qquad \boxed{q = 2 C^T Q d}$$

**The bug**: All P blocks were constructed as `Cᵀ Q C` (missing factor 2), while the linear term q was correctly `2 Cᵀ Q d`. This caused the QP to underweight the quadratic cost by 2× relative to the linear term, producing incorrect optimal controls.

**Impact**: Before fix — OSQP RMSE = ~713 mm. After fix — OSQP RMSE = 59.1 mm (matching SciPy's 58.2 mm).

**Fix location**: `_build_qp()` in `main_admittance_mpc_osqp_control.py`, all P entries multiplied by 2:
```python
# Before (BUGGY):
Qx_ee = J0.T @ np.diag(self.Qx) @ J0          # missing 2×

# After (FIXED):
Qx_ee = 2.0 * J0.T @ np.diag(self.Qx) @ J0    # correct for OSQP
```

Same fix applied to Qdx, Qdq, R, S (Δu), and Qx_terminal blocks.

### 6.2 Final Comparison

10 s simulation, external force 5 N +x on t ∈ [2, 6] s. Reproduce with `python main_compare_mpc.py`.

| Metric | SciPy L-BFGS-B | OSQP-QP |
|--------|:---:|:---:|
| End-effector RMSE X [mm] | 40.7 | 40.7 |
| End-effector RMSE Y [mm] | 41.7 | 42.9 |
| End-effector RMSE overall [mm] | 58.2 | 59.1 |
| Max tracking error [mm] | 417.2 | 417.2 |
| Max torque [N·m] | 20.00 | 20.00 |
| Mean torque norm [N·m] | 0.267 | 0.234 |
| Avg solve time [ms] | 104.8 | **0.20** |
| Max solve time [ms] | 573 | **0.54** |
| Min solve time [ms] | 37.5 | 0.18 |
| P95 solve time [ms] | 293 | **0.25** |
| Success rate [%] | 100 | 100 |
| Fallback count | 0 | 0 |
| Total solves | 500 | 500 |
| 10s simulation wall time [s] | 52.6 | 9.7 |
| Constraint violations | 0 | 0 |

### 6.3 Conclusions

1. **Tracking**: OSQP-QP achieves nearly identical tracking accuracy as the SciPy baseline (RMSE 59.1 mm vs 58.2 mm, <2% difference). The residual gap is attributable to the frozen-Jacobian linearization in the QP cost.

2. **Solve speed**: OSQP reduces average solve time by **~520×** (104.8 ms → 0.20 ms) and max solve time by **~1000×** (573 ms → 0.54 ms). At 0.2 ms per solve, the MPC can run at >1 kHz, far exceeding real-time requirements.

3. **Reliability**: 100% solve success rate, zero fallbacks, zero constraint violations. The structured QP formulation with explicit inequality constraints provides robustness guarantees that the unconstrained L-BFGS-B approach cannot.

4. **Wall time**: A 10 s simulation completes in 9.7 s with OSQP vs 52.6 s with SciPy — OSQP runs faster than real-time on this setup.

5. **Limitations**: The current implementation uses frozen dynamics (M, bias) and frozen Jacobian within each MPC solve. These approximations are acceptable for the 2-DOF case but may need re-linearization for more dynamic 6/7-DOF systems.

### 6.4 Figures

Generated automatically by `main_compare_mpc.py`:

- `results/solve_time_comparison.png` — bar chart with avg/max error bars
- `results/tracking_error_comparison.png` — X-direction tracking overlay (two subplots)
- `results/torque_comparison.png` — torque profiles with ±20 N·m limit lines
