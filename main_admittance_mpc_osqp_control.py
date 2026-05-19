import csv
import time
from pathlib import Path

import numpy as np
import mujoco
import mujoco.viewer

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

import osqp
from scipy import sparse

from kinematics import forward_kinematics, jacobian

# ============================================================
# 1. MuJoCo 模型路径
# ============================================================
MODEL_PATH = Path(__file__).parent / "models" / "two_link_arm.xml"

# ============================================================
# 2. 导纳控制器（与原版相同）
# ============================================================
class AdmittanceController:
    """二维平面导纳控制器。"""

    def __init__(self, x0):
        self.x0 = x0.copy()
        self.x_ref = x0.copy()
        self.dx_ref = np.array([0.0, 0.0])
        self.M = np.array([1.0, 1.0])
        self.D = np.array([15.0, 15.0])
        self.K = np.array([60.0, 60.0])

    def update(self, F_ext, dt):
        ddx_ref = (
            F_ext - self.D * self.dx_ref - self.K * (self.x_ref - self.x0)
        ) / self.M
        self.dx_ref = self.dx_ref + ddx_ref * dt
        self.x_ref = self.x_ref + self.dx_ref * dt
        return self.x_ref.copy(), self.dx_ref.copy(), ddx_ref.copy()


# ============================================================
# 3. 末端空间力矩 OSQP MPC 控制器
# ============================================================
class TaskSpaceOSQPMPC:
    """
    末端空间力矩 MPC，使用 OSQP 求解 QP。

    状态:   s_k = [q1, q2, dq1, dq2]_k       (4 维)
    控制:   u_k = [tau1, tau2]_k              (2 维)

    QP 变量顺序（分组排列）:
        z = [s_0, s_1, ..., s_N, u_0, u_1, ..., u_{N-1}]

    总变量数 = (N+1)*4 + N*2
    """

    def __init__(self, model, horizon=12, dt_mpc=0.03):
        self.model = model
        self.N = horizon
        self.dt = dt_mpc
        self.nq = 2
        self.nv = 2
        self.nu = 2
        self.ns = 4                          # 状态维度 [q, dq]

        # 总变量数
        self.n_states = (self.N + 1) * self.ns
        self.n_controls = self.N * self.nu
        self.n_vars = self.n_states + self.n_controls

        # --------------------------------------------------------
        # 力矩约束（与 MuJoCo motor ctrlrange 对齐）
        # --------------------------------------------------------
        self.tau_min = -20.0
        self.tau_max = 20.0

        # --------------------------------------------------------
        # 关节限位（保守默认值；XML range="-3.14 3.14"）
        # --------------------------------------------------------
        self.q_min = np.array([-3.14, -3.14])
        self.q_max = np.array([3.14, 3.14])
        self.dq_min = np.array([-10.0, -10.0])
        self.dq_max = np.array([10.0, 10.0])

        # --------------------------------------------------------
        # 代价函数权重
        # --------------------------------------------------------
        self.Qx = np.array([600.0, 600.0])            # 末端位置误差
        self.Qdx = np.array([20.0, 20.0])             # 末端速度误差
        self.Qdq = np.array([1.0, 1.0])               # 关节速度惩罚
        self.R = np.array([0.01, 0.01])               # 力矩惩罚
        self.S = np.array([0.005, 0.005])             # 力矩变化率惩罚 Δu
        self.Qx_terminal = np.array([1500.0, 1500.0]) # 终端末端位置误差

        # 上一时刻的控制量（用于 Δu 项）
        self.previous_tau = np.zeros(self.nu)

        # OSQP solver 实例（每次 solve 重新 setup）
        self._solver = None
        self._first_solve = True

    # -----------------------------------------------------------
    # 动力学
    # -----------------------------------------------------------
    def get_frozen_dynamics(self, model, data):
        """从 MuJoCo 提取 M 和 bias。"""
        mujoco.mj_forward(model, data)
        M = np.zeros((model.nv, model.nv))
        mujoco.mj_fullM(model, M, data.qM)
        bias = data.qfrc_bias.copy() + data.qfrc_passive.copy()
        return M, bias

    def _linear_dynamics(self, M, bias):
        """
        构造线性离散动力学矩阵:
            s_{k+1} = A_d * s_k + B_d * u_k + c_d

        其中 s = [q; dq], u = tau

        连续动力学:
            qdd = M^{-1} * (u - bias)

        欧拉离散化:
            q_{k+1}  = q_k + dt * dq_k
            dq_{k+1} = dq_k + dt * M^{-1} * (u_k - bias)

        矩阵形式:
            A_d = [[I, dt*I],
                   [0, I  ]]
            B_d = [[0           ],
                   [dt * M^{-1} ]]
            c_d = [[0                      ],
                   [-dt * M^{-1} @ bias    ]]
        """
        M_inv = np.linalg.inv(M)
        dt = self.dt

        A_d = np.zeros((self.ns, self.ns))
        A_d[0:2, 0:2] = np.eye(2)
        A_d[0:2, 2:4] = dt * np.eye(2)
        A_d[2:4, 2:4] = np.eye(2)

        B_d = np.zeros((self.ns, self.nu))
        B_d[2:4, :] = dt * M_inv

        c_d = np.zeros(self.ns)
        c_d[2:4] = -dt * M_inv @ bias

        return A_d, B_d, c_d, M_inv

    # -----------------------------------------------------------
    # QP 矩阵构造
    # -----------------------------------------------------------
    def _build_qp(self, q0, dq0, x_ref, dx_ref, M, bias):
        """
        构造 OSQP 标准形式的 QP:
            min  0.5 * z^T P z + q^T z
            s.t. l <= A_cons z <= u

        末端位置通过雅可比线性化近似:
            FK(q_k) ≈ FK(q0) + J0 * (q_k - q0)

        所以:
            x_k - x_ref ≈ J0 * q_k + (FK0 - J0 * q0 - x_ref)
            dx_k - dx_ref ≈ J0 * dq_k - dx_ref
        """
        dt = self.dt
        N = self.N

        J0 = jacobian(q0)
        FK0 = forward_kinematics(q0)
        pose_bias = FK0 - J0 @ q0 - x_ref           # 常数偏移
        A_d, B_d, c_d, M_inv = self._linear_dynamics(M, bias)

        # ---- P (cost quadratic) ----
        # OSQP 使用 1/2 z^T P z + q^T z 形式，因此 P = 2 * C^T Q C
        # 块结构: 状态块 P_s (4x4) 和控制块 P_u (2x2)
        P_s = np.zeros((self.ns, self.ns))
        Qx_ee = 2.0 * J0.T @ np.diag(self.Qx) @ J0         # Qx 映射到关节空间
        Qdx_ee = 2.0 * J0.T @ np.diag(self.Qdx) @ J0       # Qdx 映射到关节空间
        P_s[0:2, 0:2] = Qx_ee
        P_s[2:4, 2:4] = Qdx_ee + 2.0 * np.diag(self.Qdq)

        # 控制块基础: 2*R。Δu 的 diagonal 后面单独加
        P_u = 2.0 * np.diag(self.R)

        # 终端状态块（仅位置）
        P_s_terminal = np.zeros((self.ns, self.ns))
        Qx_term_ee = 2.0 * J0.T @ np.diag(self.Qx_terminal) @ J0
        P_s_terminal[0:2, 0:2] = Qx_term_ee

        # 拼成大 P（先用 LIL 方便修改）
        P_diag_blocks = [P_s] * N + [P_s_terminal] + [P_u] * N
        P_dense = sparse.block_diag(P_diag_blocks).tolil()

        # Δu 的 diagonal 和 cross 项: ||u_k - u_{k-1}||²_S
        # ||Δu||²_S: P contrib = 2*S diagonal, -2*S off-diagonal
        # u_0: +2S (k=0) + 2S (k=1) = 4S total (2S base, 2S extra)
        # u_k (1≤k≤N-2): +2S (k) + 2S (k+1) = 4S total
        # u_{N-1}: +2S (k=N-1) = 2S total
        u_start_col = (N + 1) * self.ns
        Su = 2.0 * np.diag(self.S)                         # 2*S for OSQP form
        for j in range(self.nu):
            P_dense[u_start_col + j, u_start_col + j] += Su[j, j]
        for k in range(1, N):
            idx_k = u_start_col + k * self.nu
            idx_prev = idx_k - self.nu
            for j in range(self.nu):
                P_dense[idx_prev + j, idx_prev + j] += Su[j, j]
                P_dense[idx_k + j, idx_prev + j] = -Su[j, j]
                P_dense[idx_prev + j, idx_k + j] = -Su[j, j]

        P_dense = P_dense.tocsc()

        # ---- q (cost linear) ----
        q_vec = np.zeros(self.n_vars)

        # 状态步 k=0..N-1 线性项: 2 * pose_bias^T * Qx * J0 for q part
        #                       -2 * dx_ref^T * Qdx * J0 for dq part
        q_pose_coeff = 2.0 * J0.T @ np.diag(self.Qx) @ pose_bias
        q_vel_coeff = -2.0 * J0.T @ np.diag(self.Qdx) @ dx_ref

        for k in range(N):
            idx = k * self.ns
            q_vec[idx:idx+2] += q_pose_coeff
            q_vec[idx+2:idx+4] += q_vel_coeff

        # 终端步 N 线性项
        idx_N = N * self.ns
        q_pose_term = 2.0 * J0.T @ np.diag(self.Qx_terminal) @ pose_bias
        q_vec[idx_N:idx_N+2] += q_pose_term

        # Δu 线性项: for k=0, -2 * S * u_{-1}
        q_u_start = (N + 1) * self.ns
        q_vec[q_u_start:q_u_start+self.nu] += -2.0 * np.diag(self.S) @ self.previous_tau

        # ---- 约束 A_cons ----
        # 约束结构:
        #   (1) s_0 = s_measured            (4 eq)
        #   (2) dynamics k=0..N-1           (4N eq)
        #   (3) tau_min <= u_k <= tau_max   (2N ineq)
        #   (4) q_min <= q_k <= q_max       (2(N+1) ineq)
        #   (5) dq_min <= dq_k <= dq_max    (2(N+1) ineq)
        n_eq_init = self.ns
        n_eq_dyn = N * self.ns
        n_ineq_tau = N * self.nu
        n_ineq_q = (N + 1) * self.nq
        n_ineq_dq = (N + 1) * self.nv
        n_eq_total = n_eq_init + n_eq_dyn
        n_ineq_total = n_ineq_tau + n_ineq_q + n_ineq_dq
        n_rows = n_eq_total + n_ineq_total

        A_rows = []
        l_vec = np.zeros(n_rows)
        u_vec = np.zeros(n_rows)

        row = 0

        # (1) 初始状态等式约束: s_0 = [q0; dq0]
        for i in range(self.ns):
            cols = sparse.csc_matrix(([1.0], ([0], [i])), shape=(1, self.n_vars))
            A_rows.append(cols)
            s0_val = np.concatenate([q0, dq0])
            l_vec[row] = s0_val[i]
            u_vec[row] = s0_val[i]
            row += 1

        # (2) 动力学等式约束: s_{k+1} - A_d*s_k - B_d*u_k = c_d
        for k in range(N):
            idx_s_k = k * self.ns
            idx_s_next = (k + 1) * self.ns
            idx_u_k = (N + 1) * self.ns + k * self.nu

            for i in range(self.ns):
                cols = sparse.lil_matrix((1, self.n_vars))
                # -A_d term
                for j in range(self.ns):
                    if abs(A_d[i, j]) > 1e-15:
                        cols[0, idx_s_k + j] = -A_d[i, j]
                # +I term on s_{k+1}
                cols[0, idx_s_next + i] = 1.0
                # -B_d term
                for j in range(self.nu):
                    if abs(B_d[i, j]) > 1e-15:
                        cols[0, idx_u_k + j] = -B_d[i, j]

                A_rows.append(cols.tocsc())
                l_vec[row] = c_d[i]
                u_vec[row] = c_d[i]
                row += 1

        # (3) 力矩不等式约束: tau_min <= u_k <= tau_max
        for k in range(N):
            idx_u_k = (N + 1) * self.ns + k * self.nu
            for j in range(self.nu):
                cols = sparse.csc_matrix(([1.0], ([0], [idx_u_k + j])), shape=(1, self.n_vars))
                A_rows.append(cols)
                l_vec[row] = self.tau_min
                u_vec[row] = self.tau_max
                row += 1

        # (4) 关节位置不等式约束: q_min <= q_k <= q_max
        for k in range(N + 1):
            idx_s_k = k * self.ns
            for j in range(self.nq):
                cols = sparse.csc_matrix(([1.0], ([0], [idx_s_k + j])), shape=(1, self.n_vars))
                A_rows.append(cols)
                l_vec[row] = self.q_min[j]
                u_vec[row] = self.q_max[j]
                row += 1

        # (5) 关节速度不等式约束: dq_min <= dq_k <= dq_max
        for k in range(N + 1):
            idx_s_k = k * self.ns
            for j in range(self.nv):
                cols = sparse.csc_matrix(([1.0], ([0], [idx_s_k + self.nq + j])), shape=(1, self.n_vars))
                A_rows.append(cols)
                l_vec[row] = self.dq_min[j]
                u_vec[row] = self.dq_max[j]
                row += 1

        A_cons = sparse.vstack(A_rows).tocsc()

        return P_dense, q_vec, A_cons, l_vec, u_vec

    # -----------------------------------------------------------
    # 求解 MPC
    # -----------------------------------------------------------
    def solve(self, model, data, x_ref, dx_ref):
        q0 = data.qpos.copy()
        dq0 = data.qvel.copy()

        M, bias = self.get_frozen_dynamics(model, data)

        P, q_vec, A_cons, l_vec, u_vec = self._build_qp(
            q0, dq0, x_ref, dx_ref, M, bias
        )

        # ---- 创建 OSQP solver（每次 setup，矩阵小，开销可忽略） ----
        # 注：Qx_ee 的稀疏结构随 J0 变化，用 update 会导致 nnz 不匹配。
        # 对于 N=12（76 变量），每次 setup 的 overhead < 0.5ms，完全可接受。
        self._solver = osqp.OSQP()
        self._solver.setup(
            P=P, q=q_vec, A=A_cons, l=l_vec, u=u_vec,
            verbose=False,
            warm_starting=True,
            polishing=True,
            eps_abs=1e-3,
            eps_rel=1e-3,
            max_iter=500,
        )
        self._first_solve = False

        # ---- 计时求解 ----
        t_start = time.perf_counter()
        result = self._solver.solve()
        solve_time_ms = (time.perf_counter() - t_start) * 1000.0

        status = result.info.status_val
        iters = result.info.iter

        # ---- 提取结果 ----
        status_ok = status in (1, 2)  # solved or solved inaccurate
        fallback = False

        if status_ok and result.x is not None:
            u_start = (self.N + 1) * self.ns
            tau_cmd = result.x[u_start:u_start + self.nu]
        else:
            # Fallback: 使用上一时刻力矩或零力矩
            tau_cmd = self.previous_tau.copy()
            fallback = True

        tau_cmd = np.clip(tau_cmd, self.tau_min, self.tau_max)

        # 更新 previous_tau（用于下一轮的 Δu）
        self.previous_tau = tau_cmd.copy()

        info = {
            "status": status,
            "status_ok": status_ok,
            "iters": iters,
            "cost": result.info.obj_val if status_ok else None,
            "solve_time_ms": solve_time_ms,
            "fallback": fallback,
        }

        return tau_cmd, info


# ============================================================
# 4. 外力输入
# ============================================================
def external_force_schedule(sim_time):
    """t=2s..6s: +x 方向 5N"""
    if 2.0 <= sim_time <= 6.0:
        return np.array([5.0, 0.0])
    return np.array([0.0, 0.0])


# ============================================================
# 5. 更新黄色参考点
# ============================================================
def update_admittance_ref_site(model, ref_site_id, x_ref):
    model.site_pos[ref_site_id] = np.array([x_ref[0], x_ref[1], 0.09])


# ============================================================
# 6. 画图函数
# ============================================================
def plot_results(log):
    t_arr = np.array(log["time"])
    x_arr = np.array(log["x"])
    x_ref_arr = np.array(log["x_ref"])
    x0_arr = np.array(log["x0"])
    F_ext_arr = np.array(log["F_ext"])
    tau_arr = np.array(log["tau"])
    cost_arr = np.array(log["cost"])
    solvetime_arr = np.array(log["solve_time_ms"])
    fallback_arr = np.array(log["fallback"], dtype=bool)

    tracking_err = x_ref_arr - x_arr
    tracking_err_norm = np.linalg.norm(tracking_err, axis=1)

    # 图 1：x 方向位置
    plt.figure()
    plt.plot(t_arr, x_arr[:, 0], label="actual x")
    plt.plot(t_arr, x_ref_arr[:, 0], "--", label="admittance ref x_ref")
    plt.plot(t_arr, x0_arr[:, 0], ":", label="original target x0")
    plt.xlabel("Time [s]"); plt.ylabel("X [m]")
    plt.title("Admittance + OSQP-MPC: X Position")
    plt.grid(True); plt.legend()

    # 图 2：y 方向位置
    plt.figure()
    plt.plot(t_arr, x_arr[:, 1], label="actual y")
    plt.plot(t_arr, x_ref_arr[:, 1], "--", label="admittance ref y_ref")
    plt.plot(t_arr, x0_arr[:, 1], ":", label="original target y0")
    plt.xlabel("Time [s]"); plt.ylabel("Y [m]")
    plt.title("Admittance + OSQP-MPC: Y Position")
    plt.grid(True); plt.legend()

    # 图 3：外力
    plt.figure()
    plt.plot(t_arr, F_ext_arr[:, 0], label="Fx")
    plt.plot(t_arr, F_ext_arr[:, 1], label="Fy")
    plt.xlabel("Time [s]"); plt.ylabel("External force [N]")
    plt.title("External Force Input")
    plt.grid(True); plt.legend()

    # 图 4：跟踪误差
    plt.figure()
    plt.plot(t_arr, tracking_err[:, 0], label="x_ref - x")
    plt.plot(t_arr, tracking_err[:, 1], label="y_ref - y")
    plt.plot(t_arr, tracking_err_norm, "--", label="error norm")
    plt.xlabel("Time [s]"); plt.ylabel("Tracking error [m]")
    plt.title("OSQP-MPC Tracking Error")
    plt.grid(True); plt.legend()

    # 图 5：关节力矩
    plt.figure()
    plt.plot(t_arr, tau_arr[:, 0], label="tau1")
    plt.plot(t_arr, tau_arr[:, 1], label="tau2")
    plt.axhline(20.0, linestyle="--", color="gray", label="tau limit ±20")
    plt.axhline(-20.0, linestyle="--", color="gray")
    plt.xlabel("Time [s]"); plt.ylabel("Joint torque [N.m]")
    plt.title("Admittance + OSQP-MPC: Joint Torque")
    plt.grid(True); plt.legend()

    # 图 6：求解时间
    plt.figure()
    plt.plot(t_arr, solvetime_arr)
    plt.xlabel("Time [s]"); plt.ylabel("Solve time [ms]")
    plt.title("OSQP Solve Time")
    plt.grid(True)

    # 图 7：MPC cost
    plt.figure()
    plt.plot(t_arr, cost_arr, label="QP objective")
    plt.xlabel("Time [s]"); plt.ylabel("Cost")
    plt.title("Admittance + OSQP-MPC: Optimization Cost")
    plt.grid(True); plt.legend()

    # 标记 fallback 时刻
    fallback_times = t_arr[fallback_arr]
    if len(fallback_times) > 0:
        for ax in plt.gcf().get_axes():
            for ft in fallback_times:
                ax.axvline(ft, color="red", alpha=0.3, linestyle="--")

    plt.show()


def print_solve_stats(stats):
    """打印求解时间统计。"""
    times = stats["solve_time_ms"]
    if len(times) == 0:
        return
    times_arr = np.array(times)
    fallback_count = stats["fallback_count"]
    total_solves = len(times)
    success_count = total_solves - fallback_count

    print("\n" + "=" * 60)
    print("  OSQP MPC Solve Statistics")
    print("=" * 60)
    print(f"  Total solves:       {total_solves}")
    print(f"  Successful:         {success_count}")
    print(f"  Fallbacks:          {fallback_count}")
    print(f"  Success rate:       {100.0 * success_count / total_solves:.1f} %")
    print(f"  Avg solve time:     {np.mean(times_arr):.3f} ms")
    print(f"  Min solve time:     {np.min(times_arr):.3f} ms")
    print(f"  Max solve time:     {np.max(times_arr):.3f} ms")
    print(f"  Median solve time:  {np.median(times_arr):.3f} ms")
    print(f"  95th percentile:    {np.percentile(times_arr, 95):.3f} ms")
    print("=" * 60)


def save_solve_stats_csv(stats, filepath):
    """保存求解统计为 CSV。"""
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "solve_index", "sim_time", "solve_time_ms",
            "osqp_status", "osqp_iter", "objective_value",
            "fallback_used", "tau1", "tau2",
        ])
        for i in range(len(stats["solve_time_ms"])):
            writer.writerow([
                i,
                stats["sim_time"][i],
                stats["solve_time_ms"][i],
                stats["status"][i],
                stats["iters"][i],
                stats["cost"][i],
                1 if stats["fallback"][i] else 0,
                stats["tau"][i][0],
                stats["tau"][i][1],
            ])
    print(f"\nSolve stats saved to: {filepath}")


# ============================================================
# 7. 主程序
# ============================================================
def main():
    model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    data = mujoco.MjData(model)

    data.qpos[:] = np.array([0.3, -0.9])
    data.qvel[:] = np.array([0.0, 0.0])
    mujoco.mj_forward(model, data)

    ee_site_id = model.site("ee_site").id
    target_site_id = model.site("target_site").id
    ref_site_id = model.site("admittance_ref_site").id

    target_pos_3d = data.site_xpos[target_site_id].copy()
    x0 = target_pos_3d[:2]

    admittance_controller = AdmittanceController(x0=x0)

    mpc = TaskSpaceOSQPMPC(
        model=model,
        horizon=12,
        dt_mpc=0.03,
    )

    update_admittance_ref_site(model=model, ref_site_id=ref_site_id, x_ref=x0)
    mujoco.mj_forward(model, data)

    print("Model loaded successfully.")
    print("Model path:", MODEL_PATH)
    print("Green fixed target x0:", x0)
    print("Control: F_ext -> Admittance -> x_ref -> OSQP-MPC -> tau")
    print("External force: 5 N +x from t=2s to t=6s")
    print("MPC horizon:", mpc.N, "  dt:", mpc.dt)
    print("Torque limit:", mpc.tau_min, mpc.tau_max)
    print("\nViewer: red=ee, green=target, yellow=admittance ref")

    log = {
        "time": [], "x": [], "x_ref": [], "x0": [],
        "F_ext": [], "tau": [], "cost": [],
    }

    stats = {
        "sim_time": [], "solve_time_ms": [],
        "status": [], "iters": [], "cost": [],
        "fallback": [], "tau": [],
        "fallback_count": 0,
    }

    sim_duration = 10.0
    sim_dt = model.opt.timestep
    control_interval_steps = 10

    step_count = 0
    tau_cmd = np.array([0.0, 0.0])
    mpc_info = {
        "cost": None, "solve_time_ms": 0.0,
        "status": 0, "iters": 0, "status_ok": True, "fallback": False,
    }

    with mujoco.viewer.launch_passive(model, data) as viewer:

        while viewer.is_running() and data.time < sim_duration:
            step_start = time.time()

            sim_time = data.time
            dt = model.opt.timestep

            q = data.qpos.copy()
            dq = data.qvel.copy()

            # 外力
            F_ext = external_force_schedule(sim_time)

            # 导纳生成参考
            x_ref, dx_ref, ddx_ref = admittance_controller.update(
                F_ext=F_ext, dt=dt
            )

            update_admittance_ref_site(
                model=model, ref_site_id=ref_site_id, x_ref=x_ref
            )

            # OSQP MPC
            if step_count % control_interval_steps == 0:
                tau_cmd, mpc_info = mpc.solve(
                    model=model, data=data, x_ref=x_ref, dx_ref=dx_ref
                )

                if mpc_info["fallback"]:
                    print(f"[WARN] OSQP fallback at t={sim_time:.3f}s, "
                          f"status={mpc_info['status']}")

                # 记录本次求解统计
                stats["sim_time"].append(sim_time)
                stats["solve_time_ms"].append(mpc_info["solve_time_ms"])
                stats["status"].append(mpc_info["status"])
                stats["iters"].append(mpc_info["iters"])
                stats["cost"].append(
                    mpc_info["cost"] if mpc_info["cost"] is not None else 0.0
                )
                stats["fallback"].append(mpc_info["fallback"])
                stats["tau"].append(tau_cmd.copy())
                if mpc_info["fallback"]:
                    stats["fallback_count"] += 1

            data.ctrl[:] = tau_cmd
            mujoco.mj_step(model, data)
            viewer.sync()

            # 真实末端状态
            q_after = data.qpos.copy()
            dq_after = data.qvel.copy()
            x = forward_kinematics(q_after)
            J = jacobian(q_after)
            dx = J @ dq_after

            log["time"].append(data.time)
            log["x"].append(x.copy())
            log["x_ref"].append(x_ref.copy())
            log["x0"].append(x0.copy())
            log["F_ext"].append(F_ext.copy())
            log["tau"].append(tau_cmd.copy())
            log["cost"].append(
                mpc_info["cost"] if mpc_info["cost"] is not None else 0.0
            )

            if step_count % 500 == 0:
                tracking_error = x_ref - x
                print(
                    f"t = {data.time:.2f}s, "
                    f"F_ext=[{F_ext[0]:.1f},{F_ext[1]:.1f}], "
                    f"x_ref=[{x_ref[0]:.3f},{x_ref[1]:.3f}], "
                    f"x=[{x[0]:.3f},{x[1]:.3f}], "
                    f"err=[{tracking_error[0]:.3f},{tracking_error[1]:.3f}], "
                    f"tau=[{tau_cmd[0]:.3f},{tau_cmd[1]:.3f}], "
                    f"solve={mpc_info['solve_time_ms']:.2f}ms, "
                    f"status={mpc_info['status']}"
                )

            step_count += 1

            # 控制速率填充日志（非求解步的 solve_time 填 0）
            if step_count % control_interval_steps != 0:
                pass

            elapsed = time.time() - step_start
            if elapsed < sim_dt:
                time.sleep(sim_dt - elapsed)

    # 扩展 solve_time 到 log 长度（非求解步填 NaN，方便画图）
    solve_times_full = np.full(len(log["time"]), np.nan)
    for i, si in enumerate(range(0, len(log["time"]), control_interval_steps)):
        if i < len(stats["solve_time_ms"]):
            idx = min(si, len(solve_times_full) - 1)
            solve_times_full[idx] = stats["solve_time_ms"][i]
    log["solve_time_ms"] = solve_times_full
    log["fallback"] = np.zeros(len(log["time"]), dtype=bool)

    # 打印统计
    print_solve_stats(stats)

    # 保存 CSV
    results_dir = Path(__file__).parent / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    save_solve_stats_csv(stats, str(results_dir / "osqp_mpc_solve_stats.csv"))

    # 画图
    print("Simulation finished. Plotting results...")
    plot_results(log)


if __name__ == "__main__":
    main()
