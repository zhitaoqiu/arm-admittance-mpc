# Real-time Admittance-MPC Control for a 2-DOF Manipulator in MuJoCo

A progressive implementation of compliant motion controllers and Model Predictive Control (MPC) for a 2-DOF planar robotic arm in MuJoCo. Starting from basic PD control, the project builds through impedance/admittance control and culminates in a **real-time-capable OSQP-QP MPC** achieving ~520× solve-time speedup over the SciPy L-BFGS-B baseline while maintaining identical tracking accuracy.

---

## Highlights

- **9 controllers** from joint-space PD to admittance + OSQP-MPC
- **~520× average solve speedup** (104.8 ms → 0.20 ms) via structured QP formulation
- **Identical tracking accuracy** (RMSE: 58.2 mm vs 59.1 mm)
- **Hard constraint satisfaction**: torque ±20 N·m, joint limits ±3.14 rad, velocity ±10 rad/s
- **100% solve success rate** with zero fallbacks
- Ready to extend to 6/7-DOF manipulators with Pinocchio

---

## Control Architecture

```
 External Force F_ext
        │
        ▼
 ┌──────────────────┐
 │ Admittance Model  │   M·ẍ + D·ẋ + K·(x - x₀) = F_ext
 │  (outer loop)     │
 └────────┬─────────┘
          │  x_ref(t), ẋ_ref(t)
          ▼
 ┌──────────────────┐
 │  OSQP-QP MPC      │   min  Σ ||xₖ - x_ref||²_Q + ||uₖ||²_R + ||Δuₖ||²_S
 │  (inner loop)     │   s.t.  sₖ₊₁ = A_d sₖ + B_d uₖ + c_d
 │                   │         τ_min ≤ uₖ ≤ τ_max
 │  Horizon N=12     │         q_min ≤ qₖ ≤ q_max
 │  dt = 0.03 s      │         dq_min ≤ dqₖ ≤ dq_max
 └────────┬─────────┘
          │  τ_cmd
          ▼
 ┌──────────────────┐
 │  MuJoCo           │
 │  2-DOF Arm        │   L₁=0.5m, L₂=0.4m
 │  Simulation       │   dt = 0.002 s
 └──────────────────┘
```

---

## Project Structure

```
.
├── kinematics.py                          # Forward kinematics & Jacobian
├── models/
│   └── two_link_arm.xml                   # MuJoCo 2-DOF arm model
│
├── main.py                                #  1. Joint-space PD
├── main_task_space_pd.py                  #  2. Task-space PD
├── main_impedance_control.py              #  3. Impedance control
├── main_impedance_plot.py                 #  4. Impedance + data logging
├── main_admittance_control.py             #  5. Admittance + PD tracking
├── main_admittance_visual.py              #  6. Admittance + visualization
├── main_mpc_joint_tracking.py             #  7. Joint-space MPC
├── main_mpc_task_space_tracking.py        #  8. Task-space MPC
├── main_admittance_mpc_control.py         #  9. Admittance + MPC (SciPy L-BFGS-B)
├── main_admittance_mpc_osqp_control.py    # 10. Admittance + MPC (OSQP-QP)
├── main_compare_mpc.py                    # 11. SciPy vs OSQP benchmark
│
├── docs/
│   └── OSQP_MPC_UPDATE.md                 # OSQP MPC documentation
├── results/                               # Generated CSV & PNG outputs
│   ├── mpc_comparison_summary.csv
│   ├── osqp_mpc_solve_stats.csv
│   ├── solve_time_comparison.png
│   ├── tracking_error_comparison.png
│   └── torque_comparison.png
├── requirements.txt
└── README.md
```

---

## OSQP-QP MPC

### State & Control

| Variable | Meaning | Dimension |
|----------|---------|-----------|
| $s_k = [q_1, q_2, \dot{q}_1, \dot{q}_2]_k$ | Joint state | $\mathbb{R}^4$ |
| $u_k = [\tau_1, \tau_2]_k$ | Torque command | $\mathbb{R}^2$ |
| $N = 12$ | Prediction horizon | — |
| $dt = 0.03$ s | MPC time step | — |

### Linear Dynamics (frozen M, bias from MuJoCo)

$$s_{k+1} = A_d s_k + B_d u_k + c_d$$

$$A_d = \begin{bmatrix} I_2 & dt \cdot I_2 \\ 0 & I_2 \end{bmatrix}, \quad
B_d = \begin{bmatrix} 0 \\ dt \cdot M^{-1} \end{bmatrix}, \quad
c_d = \begin{bmatrix} 0 \\ -dt \cdot M^{-1} b \end{bmatrix}$$

### Task-Space Cost (linearized FK via Jacobian)

$$x_{ee}(q_k) \approx FK(q_0) + J(q_0)(q_k - q_0) = J_0 q_k + b$$

$$\min_z \sum_{k=0}^{N-1} \Big( \|J_0 q_k + b - x_{ref}\|^2_{Q_x} + \|J_0 \dot{q}_k - \dot{x}_{ref}\|^2_{Q_{\dot{x}}} + \|\dot{q}_k\|^2_{Q_{\dot{q}}} + \|u_k\|^2_R + \|u_k - u_{k-1}\|^2_S \Big) + \|J_0 q_N + b - x_{ref}\|^2_{Q_{x,term}}$$

### QP Form & OSQP

The problem is converted to OSQP standard form:

$$\min_z \frac{1}{2} z^T P z + q^T z \quad \text{s.t.} \quad l \leq A_{cons} z \leq u$$

- **Decision variables** $z = [s_0, ..., s_N, u_0, ..., u_{N-1}]$, 76 vars for N=12
- **Constraints**: $s_0$ equality, dynamics equality (N×4), torque/joint/velocity inequality
- **Sparse matrices**: P and $A_{cons}$ constructed via `scipy.sparse.block_diag`
- **Important**: For cost ||Cz + d||²_Q, OSQP requires **P = 2 Cᵀ Q C** (the 1/2 factor in `1/2 zᵀPz`)

---

## Final Experimental Results

10-second simulation, external force 5N in +x from t=2s to t=6s.

| Metric | SciPy L-BFGS-B | OSQP-QP |
|--------|:---:|:---:|
| End-effector RMSE overall [mm] | 58.2 | 59.1 |
| Average solve time [ms] | 104.8 | **0.20** |
| Max solve time [ms] | 573 | **0.54** |
| P95 solve time [ms] | 293 | **0.25** |
| Max torque [N·m] | 20.00 | 20.00 |
| Mean torque norm [N·m] | 0.267 | 0.234 |
| Success rate [%] | 100 | 100 |
| Constraint violations | 0 | 0 |
| 10s simulation wall time [s] | 52.6 | 9.7 |

**Conclusion**: The OSQP-QP controller achieves nearly identical tracking accuracy compared with the SciPy baseline, while reducing the average MPC solve time by approximately **520×** (104.8 ms → 0.20 ms) and the max solve time by approximately **1000×** (573 ms → 0.54 ms).

---

## Installation & Running

### Dependencies

```bash
pip install -r requirements.txt
```

### Run Individual Controllers

```bash
# Basic
python main.py                          # Joint-space PD
python main_task_space_pd.py            # Task-space PD

# Compliance control
python main_impedance_control.py        # Impedance control
python main_impedance_plot.py           # Impedance + plotting
python main_admittance_control.py       # Admittance + PD
python main_admittance_visual.py        # Admittance + visualization

# MPC
python main_mpc_joint_tracking.py       # Joint-space MPC
python main_mpc_task_space_tracking.py  # Task-space MPC
python main_admittance_mpc_control.py   # Admittance + SciPy MPC
python main_admittance_mpc_osqp_control.py  # Admittance + OSQP MPC
```

### Reproduce Results

```bash
# 1. Install
pip install -r requirements.txt

# 2. Run benchmark (SciPy then OSQP, ~60s total)
python main_compare_mpc.py

# 3. Check outputs
ls results/
#   mpc_comparison_summary.csv
#   osqp_mpc_solve_stats.csv
#   solve_time_comparison.png
#   tracking_error_comparison.png
#   torque_comparison.png
```

**Expected results** (on a typical laptop):
- OSQP RMSE should be within ±5 mm of SciPy RMSE
- OSQP average solve time < 1 ms
- Success rate = 100%, constraint violations = 0

---

## Current Simplifications

- 2-DOF planar manipulator (L₁=0.5m, L₂=0.4m)
- MuJoCo simulation only (no physical hardware)
- Local linearized / frozen dynamics inside MPC prediction
- Frozen Jacobian for task-space cost linearization
- Ideal state feedback from simulation (no sensor noise)
- Simplified external force schedule (step function)

---

## Future Work

- **Real contact force**: replace the force schedule with `data.contact` from MuJoCo for wall/environment interaction
- **Contact tasks**: constant-force tracking, peg-in-hole, surface following
- **6/7-DOF extension**: replace kinematics/dynamics with Pinocchio, model Franka Panda or KUKA iiwa
- **Continuous re-linearization**: update Jacobian along the prediction horizon for better accuracy on large motions
- **Hardware deployment**: interface with libfranka / RTDE for real-time torque control
- **Obstacle avoidance**: add collision-avoidance constraints to the QP

---

## Key References

- MuJoCo: https://mujoco.readthedocs.io/
- OSQP: https://osqp.org/
- Siciliano, B. et al. "Robotics: Modelling, Planning and Control"
- Rawlings, J.B. et al. "Model Predictive Control: Theory, Computation, and Design"
