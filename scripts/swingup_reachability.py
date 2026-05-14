#!/usr/bin/env python
"""
Reachability analysis for cart-pendulum swing-up in (θ, θ_dot) space.

For each grid point (θ₀, θ_dot₀) with x=0, ẋ=0 fixed, simulate the full
nonlinear cart-pendulum ODE under constant force u=+u_max and u=-u_max
separately. Each point is categorised as:
  0 — neither force reaches the goal (unreachable)
  1 — u=+u_max only
  2 — u=-u_max only
  3 — both forces reach the goal

The grid is saved to results/swingup_reachability.npz so the plot can be
regenerated without re-running the simulations.

Also overlays:
  - analytical separatrix for the isolated pendulum
  - two ellipses indicating regions where each force cannot reach the goal
"""

import sys
import argparse
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
from functools import partial

sys.path.insert(0, ".")
from gym_CartPole_BT.systems.cartpend import cartpend_dxdt  # noqa: E402

# Physical parameters matching CartPole-BT-r2-a1-v1
m = 1.0  # pole mass [kg]
M = 5.0  # cart mass [kg]
L = 2.0  # pole half-length [m]
g = -10.0  # gravity [m/s²] (negative = downward)
d = 1.0  # cart friction [Ns/m]
u_max = 200.0  # max force [N]

# Simulation parameters
T_sim = 5.0  # time horizon [s]
THETA_GOAL = np.pi
THETA_TOL = 0.15  # success radius [rad] (~8.6°)
MAX_STEP = 0.02  # ODE max step size [s]

# Grid resolution
N_THETA = 80
N_THETADOT = 80
theta_grid = np.linspace(0.0, 2 * np.pi, N_THETA)
thetadot_max = 7.5
thetadot_grid = np.linspace(-thetadot_max, thetadot_max, N_THETADOT)

DATA_FILE = "results/swingup_reachability.npz"


# ---------------------------------------------------------------------------


def make_goal_event(theta_goal, tol):
    def event(t, x):
        diff = (x[2] - theta_goal + np.pi) % (2 * np.pi) - np.pi
        return abs(diff) - tol

    event.terminal = True
    event.direction = 0
    return event


goal_event = make_goal_event(THETA_GOAL, THETA_TOL)


def is_reachable(theta0, thetadot0, u):
    x0 = np.array([0.0, 0.0, theta0, thetadot0])
    ode = partial(cartpend_dxdt, m=m, M=M, L=L, g=g, d=d, u=u)
    sol = solve_ivp(
        ode,
        (0.0, T_sim),
        x0,
        events=goal_event,
        max_step=MAX_STEP,
        dense_output=False,
    )
    return len(sol.t_events[0]) > 0


def compute_grid():
    print(
        f"Computing {N_THETA}×{N_THETADOT} reachability grid "
        f"(u=±{u_max:.0f} N, T={T_sim}s)..."
    )
    reach_pos = np.zeros((N_THETADOT, N_THETA), dtype=bool)  # u=+u_max
    reach_neg = np.zeros((N_THETADOT, N_THETA), dtype=bool)  # u=-u_max

    for i, td in enumerate(thetadot_grid):
        if i % 10 == 0:
            print(f"  row {i}/{N_THETADOT}  (θ_dot={td:.2f})")
        for j, th in enumerate(theta_grid):
            diff = (th - THETA_GOAL + np.pi) % (2 * np.pi) - np.pi
            if abs(diff) < THETA_TOL:
                reach_pos[i, j] = True
                reach_neg[i, j] = True
                continue
            reach_pos[i, j] = is_reachable(th, td, +u_max)
            reach_neg[i, j] = is_reachable(th, td, -u_max)

    np.savez(
        DATA_FILE,
        reach_pos=reach_pos,
        reach_neg=reach_neg,
        theta_grid=theta_grid,
        thetadot_grid=thetadot_grid,
    )
    print(f"Saved {DATA_FILE}")
    return reach_pos, reach_neg


def load_grid():
    data = np.load(DATA_FILE)
    return data["reach_pos"], data["reach_neg"]


# ---------------------------------------------------------------------------


def check_reachability(theta, theta_dot):
    """Return (not_pos, not_neg) indicating whether (theta, theta_dot) lies
    inside each ellipse.  not_neg → inside ellipse 1 (u=-u_max unlikely to
    reach goal); not_pos → inside ellipse 2 (u=+u_max unlikely to reach goal).
    Theta is treated as periodic so wrap-around is handled correctly.
    """
    a, b = 1.750, 6.36075  # semi-axes shared by both ellipses

    def _inside(cx, cy):
        dtheta = ((theta - cx + np.pi) % (2 * np.pi)) - np.pi
        return (dtheta / a) ** 2 + ((theta_dot - cy) / b) ** 2 <= 1.0

    not_neg = _inside(1.233, 0.0)  # ellipse 1: "not -ve"
    not_pos = _inside(5.050, 0.0)  # ellipse 2: "not +ve"
    return not_pos, not_neg


def make_plot(reach_pos, reach_neg):
    # Category array: 0=neither, 1=pos only, 2=neg only, 3=both
    category = reach_pos.astype(int) + 2 * reach_neg.astype(int)
    # 0 → neither, 1 → pos only, 2 → neg only, 3 → both
    # Reorder: pos only = reach_pos & ~reach_neg = 1
    #          neg only = reach_neg & ~reach_pos = 2
    # But our encoding: 1=pos, 2=neg, 3=both which is correct.

    cmap = mcolors.ListedColormap(
        [
            "#b22222",  # 0: neither — dark red
            "#90ee90",  # 1: u=+u_max only — light green
            "#228b22",  # 2: u=-u_max only — dark green
            "#32cd32",  # 3: both — medium lime green
        ]
    )
    bounds = [-0.5, 0.5, 1.5, 2.5, 3.5]
    norm = mcolors.BoundaryNorm(bounds, cmap.N)

    fig, ax = plt.subplots(figsize=(12, 6))
    TH, TD = np.meshgrid(theta_grid, thetadot_grid)
    ax.pcolormesh(
        TH, TD, category, cmap=cmap, norm=norm, shading="auto", alpha=0.85
    )

    # Analytical separatrix (isolated pendulum, no cart, no friction)
    theta_sep = np.linspace(0.0, 2 * np.pi, 1000)
    td_sep = np.sqrt(np.maximum(0.0, 10.0 * (1.0 + np.cos(theta_sep))))
    ax.plot(
        theta_sep,
        td_sep,
        "b-",
        lw=2.0,
        label=r"Separatrix: $\dot\theta^2=10(1+\cos\theta)$",
    )
    ax.plot(theta_sep, -td_sep, "b-", lw=2.0)

    # --- Ellipses as scatter dots (200 points, theta wrapped to [0, 2π])
    t_ell = np.linspace(0, 2 * np.pi, 200, endpoint=False)

    # Ellipse 1: centre (1.233, 0.0), semi-axes (1.750, 6.36075), label "not -ve"
    th_e1 = (1.233 + 1.750 * np.cos(t_ell)) % (2 * np.pi)
    td_e1 = 6.36075 * np.sin(t_ell)
    ax.scatter(th_e1, td_e1, s=4, color="lime", zorder=5, label="not -ve")

    # Ellipse 2: centre (5.050, 0.0), semi-axes (1.750, 6.36075), label "not +ve"
    th_e2 = (5.050 + 1.750 * np.cos(t_ell)) % (2 * np.pi)
    td_e2 = 6.36075 * np.sin(t_ell)
    ax.scatter(th_e2, td_e2, s=4, color="darkgreen", zorder=5, label="not +ve")

    # Key angle markers
    for ang, lbl in [
        (0.0, "θ=0\n(start)"),
        (np.pi, "θ=π\n(goal)"),
        (2 * np.pi, "θ=2π"),
    ]:
        ax.axvline(ang, color="k", lw=0.8, linestyle="--", alpha=0.5)
        ax.text(ang + 0.06, thetadot_max * 0.93, lbl, fontsize=8, va="top")

    # Legend
    patches = [
        mpatches.Patch(color="#b22222", label="Neither force reaches goal"),
        mpatches.Patch(color="#90ee90", label="u=+200 N only"),
        mpatches.Patch(color="#228b22", label="u=−200 N only"),
        mpatches.Patch(color="#32cd32", label="Either force"),
    ]
    ax.legend(
        handles=patches
        + [
            plt.Line2D(
                [0],
                [0],
                color="b",
                lw=2,
                label=r"Separatrix: $\dot\theta^2=10(1+\cos\theta)$",
            ),
            plt.Line2D(
                [0],
                [0],
                color="lime",
                marker="o",
                markersize=4,
                linestyle="None",
                label="not -ve",
            ),
            plt.Line2D(
                [0],
                [0],
                color="darkgreen",
                marker="o",
                markersize=4,
                linestyle="None",
                label="not +ve",
            ),
        ],
        fontsize=8,
        loc="upper right",
    )

    ax.set_xlim(0.0, 2 * np.pi)
    ax.set_ylim(-thetadot_max, thetadot_max)
    ax.set_xticks([0, np.pi / 2, np.pi, 3 * np.pi / 2, 2 * np.pi])
    ax.set_xticklabels(["0", "π/2", "π", "3π/2", "2π"])
    ax.set_xlabel(r"$\theta$ (rad)", fontsize=13)
    ax.set_ylabel(r"$\dot\theta$ (rad/s)", fontsize=13)
    ax.set_title(
        f"Swing-up reachability  (constant u = ±{u_max:.0f} N,  T = {T_sim} s,  "
        r"$x = \dot{x} = 0$)",
        fontsize=11,
    )
    ax.grid(True, alpha=0.25)
    plt.tight_layout()
    out = "results/swingup_reachability.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved {out}")
    plt.show()


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--reuse",
        action="store_true",
        help=f"load grid from {DATA_FILE} instead of recomputing",
    )
    args = parser.parse_args()

    if args.reuse:
        reach_pos, reach_neg = load_grid()
        print(f"Loaded grid from {DATA_FILE}")
    else:
        reach_pos, reach_neg = compute_grid()

    make_plot(reach_pos, reach_neg)

    # --- OCP solve from selected test points ---------------------------------
    sys.path.insert(0, "examples")
    from run_swingup_mpc_simple import (  # noqa: E402
        solve_swingup_ocp,
        plot_solution,
        make_u_guess,
        max_u,
        N as N_ocp,
    )

    theta0_values = np.radians(np.arange(0, 360, 15))
    test_points = [(th, 0.0) for th in theta0_values]

    results = []

    for i, (theta0, thetadot0) in enumerate(test_points):
        not_pos, not_neg = check_reachability(theta0, thetadot0)
        print(
            f"\nSim {i:2d}: θ₀={np.degrees(theta0):6.1f}°, θ̇₀={thetadot0:.1f}  →  "
            f"not_pos={not_pos}, not_neg={not_neg}"
        )

        x0 = np.array([0.0, 0.0, theta0, thetadot0])

        if not_neg:
            u_guess = make_u_guess(x0, +max_u)
        elif not_pos:
            u_guess = make_u_guess(x0, -max_u)
        else:
            u_guess = np.zeros(N_ocp)

        u_sol, solve_ok, t_solve, sim_states, t_s, theta_sol, x_sol, cost = \
            solve_swingup_ocp(x0, u_guess)

        results.append((i, x0, solve_ok, cost, t_solve))

        if solve_ok:
            final_deg = np.degrees(theta_sol[-1] % (2 * np.pi))
            print(
                f"  Solved in {t_solve:.1f} s  —  cost = {cost:.4f}"
                f"  —  final θ = {final_deg:.1f}° (target 180°)"
            )
            plot_solution(
                u_guess, sim_states, t_s, theta_sol, x_sol, u_sol,
                solve_ok, t_solve,
            )
        else:
            print(f"  FAILED in {t_solve:.1f} s")

    # --- Summary table -------------------------------------------------------
    hdr = f"{'Sim':>3}  {'x₀':>5}  {'ẋ₀':>5}  {'θ₀(°)':>7}  {'θ̇₀':>5}  " \
          f"{'OK':>5}  {'Cost':>12}  {'Time(s)':>8}"
    sep = "-" * len(hdr)
    print(f"\n{'OCP results summary':^{len(hdr)}}")
    print(sep)
    print(hdr)
    print(sep)
    for i, x0, solve_ok, cost, t_solve in results:
        x_, xd_, th_, thd_ = x0
        ok_str  = "Yes" if solve_ok else "No"
        cost_str = f"{cost:.4f}" if cost is not None else "N/A"
        print(
            f"{i:>3}  {x_:>5.2f}  {xd_:>5.2f}  {np.degrees(th_):>7.1f}  "
            f"{thd_:>5.2f}  {ok_str:>5}  {cost_str:>12}  {t_solve:>8.1f}"
        )
    print(sep)
