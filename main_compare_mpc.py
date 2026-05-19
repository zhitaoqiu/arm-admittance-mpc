"""
SciPy L-BFGS-B MPC vs OSQP MPC 对比实验。

运行方式:
    python main_compare_mpc.py

输出:
    - 终端对比表
    - results/mpc_comparison_summary.csv
    - results/solve_time_comparison.png
    - results/tracking_error_comparison.png
    - results/torque_comparison.png
"""
import csv
import time
from pathlib import Path

import numpy as np
import mujoco

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---- 导入两个 MPC 控制器 ----
from main_admittance_mpc_control import (
    AdmittanceController as AdmittanceController1,
    TaskSpaceTorqueMPC as SciPyMPC,
    external_force_schedule as force_schedule1,
)
from main_admittance_mpc_osqp_control import (
    AdmittanceController as AdmittanceController2,
    TaskSpaceOSQPMPC as OSQPMPC,
    external_force_schedule as force_schedule2,
)

from kinematics import forward_kinematics, jacobian

MODEL_PATH = Path(__file__).parent / "models" / "two_link_arm.xml"
RESULTS_DIR = Path(__file__).parent / "results"


def run_simulation(mpc_type, sim_duration=10.0):
    """
    运行一次仿真，返回 log 和统计。

    mpc_type: "scipy" 或 "osqp"
    """
    model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    data = mujoco.MjData(model)

    data.qpos[:] = np.array([0.3, -0.9])
    data.qvel[:] = np.array([0.0, 0.0])
    mujoco.mj_forward(model, data)

    target_site_id = model.site("target_site").id
    target_pos_3d = data.site_xpos[target_site_id].copy()
    x0 = target_pos_3d[:2]

    if mpc_type == "scipy":
        admittance = AdmittanceController1(x0=x0)
        mpc = SciPyMPC(model=model, horizon=12, dt_mpc=0.03)
        force_fn = force_schedule1
        tau_min, tau_max = -20.0, 20.0
    else:
        admittance = AdmittanceController2(x0=x0)
        mpc = OSQPMPC(model=model, horizon=12, dt_mpc=0.03)
        force_fn = force_schedule2
        tau_min, tau_max = mpc.tau_min, mpc.tau_max

    # 约束限位
    q_min = np.array([-3.14, -3.14])
    q_max = np.array([3.14, 3.14])
    dq_min = np.array([-10.0, -10.0])
    dq_max = np.array([10.0, 10.0])

    log = {
        "time": [], "x": [], "x_ref": [], "F_ext": [],
        "tau": [], "q": [], "dq": [], "cost": [],
    }
    solve_stats = {
        "solve_time_ms": [], "status_ok": [],
        "fallback_count": 0,
        # per-solve details (for osqp_mpc_solve_stats.csv)
        "sim_time": [], "status": [], "iters": [],
        "tau_at_solve": [], "cost_at_solve": [],
    }

    sim_dt = model.opt.timestep
    control_interval_steps = 10
    step_count = 0
    tau_cmd = np.array([0.0, 0.0])
    mpc_info = {
        "cost": None, "solve_time_ms": 0.0,
        "status": 0, "iters": 0, "status_ok": True, "fallback": False,
    }

    t_start = time.perf_counter()

    while data.time < sim_duration:
        sim_time = data.time
        dt = model.opt.timestep

        q = data.qpos.copy()
        dq = data.qvel.copy()

        F_ext = force_fn(sim_time)
        x_ref, dx_ref, _ddx_ref = admittance.update(F_ext=F_ext, dt=dt)

        if step_count % control_interval_steps == 0:
            if mpc_type == "scipy":
                t_s = time.perf_counter()
                tau_cmd, info = mpc.solve(
                    model=model, data=data, x_ref=x_ref, dx_ref=dx_ref
                )
                t_ms = (time.perf_counter() - t_s) * 1000.0
                tau_cmd = np.clip(tau_cmd, tau_min, tau_max)
                status_ok = info["success"]
                fallback = False
                cost_val = info["cost"]
            else:
                t_s = time.perf_counter()
                tau_cmd, info = mpc.solve(
                    model=model, data=data, x_ref=x_ref, dx_ref=dx_ref
                )
                t_ms = info["solve_time_ms"]
                status_ok = info["status_ok"]
                fallback = info["fallback"]
                cost_val = info["cost"] if info["cost"] is not None else 0.0
                if fallback:
                    solve_stats["fallback_count"] += 1
                # per-solve details for CSV
                solve_stats["sim_time"].append(sim_time)
                solve_stats["status"].append(info["status"])
                solve_stats["iters"].append(info["iters"])
                solve_stats["tau_at_solve"].append(tau_cmd.copy())
                solve_stats["cost_at_solve"].append(cost_val)

            solve_stats["solve_time_ms"].append(t_ms)
            solve_stats["status_ok"].append(status_ok)

            mpc_info = {
                "cost": cost_val,
                "solve_time_ms": t_ms,
                "status_ok": status_ok,
                "fallback": fallback,
            }

        data.ctrl[:] = tau_cmd
        mujoco.mj_step(model, data)

        q_after = data.qpos.copy()
        dq_after = data.qvel.copy()
        x = forward_kinematics(q_after)

        log["time"].append(data.time)
        log["x"].append(x.copy())
        log["x_ref"].append(x_ref.copy())
        log["F_ext"].append(F_ext.copy())
        log["tau"].append(tau_cmd.copy())
        log["q"].append(q_after.copy())
        log["dq"].append(dq_after.copy())
        log["cost"].append(
            mpc_info["cost"] if mpc_info["cost"] is not None else 0.0
        )

        step_count += 1

    wall_time = time.perf_counter() - t_start

    # ---- 约束违反检查 ----
    tau_arr = np.array(log["tau"])
    q_arr = np.array(log["q"])
    dq_arr = np.array(log["dq"])

    n_tau_violations = int(np.sum(np.abs(tau_arr) > tau_max + 1e-6))
    n_q_violations = int(
        np.sum((q_arr < q_min - 1e-6) | (q_arr > q_max + 1e-6))
    )
    n_dq_violations = int(
        np.sum((dq_arr < dq_min - 1e-6) | (dq_arr > dq_max + 1e-6))
    )

    constraints_ok = (
        n_tau_violations == 0
        and n_q_violations == 0
        and n_dq_violations == 0
    )

    violations = {
        "tau_violations": n_tau_violations,
        "q_violations": n_q_violations,
        "dq_violations": n_dq_violations,
        "constraints_ok": constraints_ok,
    }

    return log, solve_stats, wall_time, violations


def compute_metrics(log, solve_stats, label):
    """从 log 和统计中计算对比指标。"""
    x_arr = np.array(log["x"])
    x_ref_arr = np.array(log["x_ref"])
    tau_arr = np.array(log["tau"])
    solvetimes = np.array(solve_stats["solve_time_ms"])
    status_ok_arr = np.array(solve_stats["status_ok"])

    # 末端跟踪 RMSE
    tracking_error = x_ref_arr - x_arr
    rmse_x = np.sqrt(np.mean(tracking_error[:, 0] ** 2))
    rmse_y = np.sqrt(np.mean(tracking_error[:, 1] ** 2))
    rmse_overall = np.sqrt(np.mean(np.sum(tracking_error ** 2, axis=1)))

    # 最大跟踪误差
    max_tracking_error = np.max(np.linalg.norm(tracking_error, axis=1))

    # 力矩统计
    max_tau = np.max(np.abs(tau_arr))
    mean_tau_norm = np.mean(np.linalg.norm(tau_arr, axis=1))

    # 求解时间
    avg_solve = np.mean(solvetimes)
    max_solve = np.max(solvetimes)
    min_solve = np.min(solvetimes)
    p95_solve = np.percentile(solvetimes, 95)

    # 成功率
    success_rate = np.mean(status_ok_arr.astype(float)) * 100.0
    fallback_count = solve_stats.get("fallback_count", 0)

    return {
        "label": label,
        "rmse_x": rmse_x,
        "rmse_y": rmse_y,
        "rmse_overall": rmse_overall,
        "max_tracking_error": max_tracking_error,
        "max_tau": max_tau,
        "mean_tau_norm": mean_tau_norm,
        "avg_solve_ms": avg_solve,
        "max_solve_ms": max_solve,
        "min_solve_ms": min_solve,
        "p95_solve_ms": p95_solve,
        "success_rate": success_rate,
        "fallback_count": fallback_count,
        "total_solves": len(solvetimes),
    }


def plot_save_figures(log_scipy, log_osqp, metric_scipy, metric_osqp):
    """生成并保存 3 张对比图到 results/。"""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    t_sci = np.array(log_scipy["time"])
    t_osqp = np.array(log_osqp["time"])
    x_sci = np.array(log_scipy["x"])
    x_osqp = np.array(log_osqp["x"])
    x_ref_sci = np.array(log_scipy["x_ref"])
    x_ref_osqp = np.array(log_osqp["x_ref"])
    tau_sci = np.array(log_scipy["tau"])
    tau_osqp = np.array(log_osqp["tau"])

    tracking_err_sci = np.linalg.norm(x_ref_sci - x_sci, axis=1)
    tracking_err_osqp = np.linalg.norm(x_ref_osqp - x_osqp, axis=1)

    # ---- Figure 1: Solve Time Comparison ----
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(
        [0, 1],
        [metric_scipy["avg_solve_ms"], metric_osqp["avg_solve_ms"]],
        yerr=[
            max(metric_scipy["avg_solve_ms"] - metric_scipy["min_solve_ms"],
                metric_scipy["max_solve_ms"] - metric_scipy["avg_solve_ms"]),
            max(metric_osqp["avg_solve_ms"] - metric_osqp["min_solve_ms"],
                metric_osqp["max_solve_ms"] - metric_osqp["avg_solve_ms"]),
        ],
        color=["#1f77b4", "#ff7f0e"],
        capsize=8, width=0.5,
    )
    ax.set_xticks([0, 1])
    ax.set_xticklabels([
        f"SciPy L-BFGS-B\navg={metric_scipy['avg_solve_ms']:.1f} ms\n"
        f"max={metric_scipy['max_solve_ms']:.1f} ms  p95={metric_scipy['p95_solve_ms']:.1f} ms",
        f"OSQP QP\navg={metric_osqp['avg_solve_ms']:.3f} ms\n"
        f"max={metric_osqp['max_solve_ms']:.3f} ms  p95={metric_osqp['p95_solve_ms']:.3f} ms",
    ])
    ax.set_ylabel("Solve time [ms]")
    ax.set_title("MPC Solve Time Comparison")
    ax.grid(True, axis="y")
    fig.tight_layout()
    fig.savefig(str(RESULTS_DIR / "solve_time_comparison.png"), dpi=150)
    plt.close(fig)

    # ---- Figure 2: Tracking Error Comparison ----
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    ax1.plot(t_sci, x_sci[:, 0], label="actual x")
    ax1.plot(t_sci, x_ref_sci[:, 0], "--", label="x_ref")
    ax1.set_ylabel("X [m]")
    ax1.set_title(f"SciPy MPC — X Tracking (RMSE={metric_scipy['rmse_x']*1000:.1f} mm)")
    ax1.grid(True); ax1.legend(loc="upper right")

    ax2.plot(t_osqp, x_osqp[:, 0], label="actual x")
    ax2.plot(t_osqp, x_ref_osqp[:, 0], "--", label="x_ref")
    ax2.set_xlabel("Time [s]"); ax2.set_ylabel("X [m]")
    ax2.set_title(f"OSQP MPC — X Tracking (RMSE={metric_osqp['rmse_x']*1000:.1f} mm)")
    ax2.grid(True); ax2.legend(loc="upper right")

    fig.tight_layout()
    fig.savefig(str(RESULTS_DIR / "tracking_error_comparison.png"), dpi=150)
    plt.close(fig)

    # ---- Figure 3: Torque Comparison ----
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    ax1.plot(t_sci, tau_sci[:, 0], label="tau1")
    ax1.plot(t_sci, tau_sci[:, 1], label="tau2")
    ax1.axhline(20, ls="--", color="gray", alpha=0.5)
    ax1.axhline(-20, ls="--", color="gray", alpha=0.5)
    ax1.set_ylabel("Torque [N.m]")
    ax1.set_title(f"SciPy MPC — Torque (max|τ|={metric_scipy['max_tau']:.2f}, "
                  f"mean|τ|={metric_scipy['mean_tau_norm']:.3f})")
    ax1.grid(True); ax1.legend(loc="upper right")

    ax2.plot(t_osqp, tau_osqp[:, 0], label="tau1")
    ax2.plot(t_osqp, tau_osqp[:, 1], label="tau2")
    ax2.axhline(20, ls="--", color="gray", alpha=0.5)
    ax2.axhline(-20, ls="--", color="gray", alpha=0.5)
    ax2.set_xlabel("Time [s]"); ax2.set_ylabel("Torque [N.m]")
    ax2.set_title(f"OSQP MPC — Torque (max|τ|={metric_osqp['max_tau']:.2f}, "
                  f"mean|τ|={metric_osqp['mean_tau_norm']:.3f})")
    ax2.grid(True); ax2.legend(loc="upper right")

    fig.tight_layout()
    fig.savefig(str(RESULTS_DIR / "torque_comparison.png"), dpi=150)
    plt.close(fig)

    print(f"\nFigures saved to: {RESULTS_DIR}")


def print_comparison_table(m1, m2):
    """终端对比表。"""
    print("\n" + "=" * 72)
    print("  SciPy L-BFGS-B  vs  OSQP MPC  Comparison")
    print("=" * 72)
    rows = [
        ("End-effector RMSE X [mm]",        m1["rmse_x"]*1000,    m2["rmse_x"]*1000,    ".1f"),
        ("End-effector RMSE Y [mm]",        m1["rmse_y"]*1000,    m2["rmse_y"]*1000,    ".1f"),
        ("End-effector RMSE overall [mm]",  m1["rmse_overall"]*1000, m2["rmse_overall"]*1000, ".1f"),
        ("Max tracking error [mm]",         m1["max_tracking_error"]*1000, m2["max_tracking_error"]*1000, ".1f"),
        ("Max |torque| [N.m]",              m1["max_tau"],        m2["max_tau"],        ".2f"),
        ("Mean torque norm [N.m]",          m1["mean_tau_norm"],  m2["mean_tau_norm"],  ".3f"),
        ("Avg solve time [ms]",             m1["avg_solve_ms"],   m2["avg_solve_ms"],   ".3f"),
        ("Max solve time [ms]",             m1["max_solve_ms"],   m2["max_solve_ms"],   ".3f"),
        ("Min solve time [ms]",             m1["min_solve_ms"],   m2["min_solve_ms"],   ".3f"),
        ("P95 solve time [ms]",             m1["p95_solve_ms"],   m2["p95_solve_ms"],   ".3f"),
        ("Success rate [%]",                m1["success_rate"],   m2["success_rate"],   ".1f"),
        ("Fallback count",                  m1["fallback_count"], m2["fallback_count"], "d"),
        ("Total solves",                    m1["total_solves"],   m2["total_solves"],   "d"),
    ]
    header = f"  {'Metric':<34s} {'SciPy':>14s} {'OSQP':>14s}"
    print(header)
    print("  " + "-" * 64)
    for name, v1, v2, fmt in rows:
        s1 = f"{v1:{fmt}}" if isinstance(v1, float) else str(int(v1))
        s2 = f"{v2:{fmt}}" if isinstance(v2, float) else str(int(v2))
        print(f"  {name:<34s} {s1:>14s} {s2:>14s}")
    print("=" * 72)


def main():
    print("=" * 60)
    print("  Running SciPy L-BFGS-B MPC ...")
    print("=" * 60)
    log_scipy, stats_scipy, wall_scipy, viol_scipy = run_simulation("scipy", sim_duration=10.0)
    metric_scipy = compute_metrics(log_scipy, stats_scipy, "SciPy")

    print(f"\n  SciPy constraints OK: {viol_scipy['constraints_ok']}")
    if not viol_scipy["constraints_ok"]:
        print(f"    tau viol: {viol_scipy['tau_violations']}, "
              f"q viol: {viol_scipy['q_violations']}, "
              f"dq viol: {viol_scipy['dq_violations']}")

    print("\n" + "=" * 60)
    print("  Running OSQP MPC ...")
    print("=" * 60)
    log_osqp, stats_osqp, wall_osqp, viol_osqp = run_simulation("osqp", sim_duration=10.0)
    metric_osqp = compute_metrics(log_osqp, stats_osqp, "OSQP")

    print(f"\n  OSQP constraints OK: {viol_osqp['constraints_ok']}")
    if not viol_osqp["constraints_ok"]:
        print(f"    tau viol: {viol_osqp['tau_violations']}, "
              f"q viol: {viol_osqp['q_violations']}, "
              f"dq viol: {viol_osqp['dq_violations']}")

    print(f"\nSciPy wall time: {wall_scipy:.1f}s")
    print(f"OSQP  wall time: {wall_osqp:.1f}s")

    print_comparison_table(metric_scipy, metric_osqp)

    # ---- 保存对比 CSV ----
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = RESULTS_DIR / "mpc_comparison_summary.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Metric", "SciPy", "OSQP"])
        row_data = [
            ("RMSE X [mm]",             metric_scipy["rmse_x"]*1000,       metric_osqp["rmse_x"]*1000),
            ("RMSE Y [mm]",             metric_scipy["rmse_y"]*1000,       metric_osqp["rmse_y"]*1000),
            ("RMSE overall [mm]",       metric_scipy["rmse_overall"]*1000, metric_osqp["rmse_overall"]*1000),
            ("Max tracking error [mm]", metric_scipy["max_tracking_error"]*1000, metric_osqp["max_tracking_error"]*1000),
            ("Max torque [N.m]",        metric_scipy["max_tau"],           metric_osqp["max_tau"]),
            ("Mean torque norm [N.m]",  metric_scipy["mean_tau_norm"],     metric_osqp["mean_tau_norm"]),
            ("Avg solve time [ms]",     metric_scipy["avg_solve_ms"],      metric_osqp["avg_solve_ms"]),
            ("Max solve time [ms]",     metric_scipy["max_solve_ms"],      metric_osqp["max_solve_ms"]),
            ("Min solve time [ms]",     metric_scipy["min_solve_ms"],      metric_osqp["min_solve_ms"]),
            ("P95 solve time [ms]",     metric_scipy["p95_solve_ms"],      metric_osqp["p95_solve_ms"]),
            ("Success rate [%]",        metric_scipy["success_rate"],      metric_osqp["success_rate"]),
            ("Fallback count",          metric_scipy["fallback_count"],    metric_osqp["fallback_count"]),
            ("Total solves",            metric_scipy["total_solves"],      metric_osqp["total_solves"]),
            ("Wall time [s]",           wall_scipy,                        wall_osqp),
            ("Tau constraints OK",      1 if viol_scipy["constraints_ok"] else 0, 1 if viol_osqp["constraints_ok"] else 0),
            ("Q constraints OK",        1 if viol_scipy["q_violations"]==0 else 0, 1 if viol_osqp["q_violations"]==0 else 0),
            ("Dq constraints OK",       1 if viol_scipy["dq_violations"]==0 else 0, 1 if viol_osqp["dq_violations"]==0 else 0),
        ]
        for row in row_data:
            writer.writerow([row[0], f"{row[1]:.4f}", f"{row[2]:.4f}"])
    print(f"\nComparison CSV saved to: {csv_path}")

    # ---- 保存 OSQP per-solve stats CSV ----
    osqp_csv = RESULTS_DIR / "osqp_mpc_solve_stats.csv"
    with open(osqp_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "solve_index", "sim_time", "solve_time_ms",
            "osqp_status", "osqp_iter", "objective_value",
            "fallback_used", "tau1", "tau2",
        ])
        n_osqp = len(stats_osqp["solve_time_ms"])
        for i in range(n_osqp):
            writer.writerow([
                i,
                f"{stats_osqp['sim_time'][i]:.4f}" if i < len(stats_osqp["sim_time"]) else "0.0",
                f"{stats_osqp['solve_time_ms'][i]:.4f}",
                stats_osqp["status"][i] if i < len(stats_osqp["status"]) else 0,
                stats_osqp["iters"][i] if i < len(stats_osqp["iters"]) else 0,
                f"{stats_osqp['cost_at_solve'][i]:.4f}" if i < len(stats_osqp["cost_at_solve"]) else "0.0",
                1 if (i < len(stats_osqp["status_ok"]) and not stats_osqp["status_ok"][i]) else 0,
                f"{stats_osqp['tau_at_solve'][i][0]:.4f}" if i < len(stats_osqp["tau_at_solve"]) else "0.0",
                f"{stats_osqp['tau_at_solve'][i][1]:.4f}" if i < len(stats_osqp["tau_at_solve"]) else "0.0",
            ])
    print(f"OSQP solve stats CSV saved to: {osqp_csv}")

    # ---- 生成 3 张 PNG 图 ----
    plot_save_figures(log_scipy, log_osqp, metric_scipy, metric_osqp)


if __name__ == "__main__":
    main()
