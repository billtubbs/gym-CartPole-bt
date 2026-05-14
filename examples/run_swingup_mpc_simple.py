#!/usr/bin/env python
# Swing-up trajectory optimisation using Rockit / CasADi / IPOPT.
#
# solve_swingup_ocp(x0, u_guess) solves a single open-loop OCP over a long
# horizon.  The environment object is used only to read physical parameters.
# The OCP is built once at module load; repeated calls only re-solve it.

import time
import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
import gym_CartPole_BT  # noqa: F401
from rockit import Ocp, MultipleShooting
import casadi as cas

# ---- Environment (parameter extraction only) --------------------------------
env = gym.make("CartPole-BT-r2-a1-v1")
env.reset()
uenv = env.unwrapped
m = uenv.masspole  # 1.0 kg
M = uenv.masscart  # 5.0 kg
L = uenv.length  # 2.0 m
g = uenv.gravity  # -10.0 m/s²
d = uenv.friction  # 1.0 Ns/m
tau = uenv.tau  # 0.05 s
max_u = float(uenv.max_force)  # 200 N
env.close()

# ---- OCP parameters ---------------------------------------------------------
N = 100  # prediction horizon (steps)
T_hor = N * tau  # horizon duration [s]

# Cost weights
Q_theta = 5.0
Q_x = 1.0
Q_E = 0.002
Q_T = 50.0
Q_thdot = 0.1
Q_xdot = 0.1
R_u = 1e-5

t_nodes = np.linspace(0, T_hor, N + 1)

# ---- Build OCP (once at module load) ----------------------------------------
ocp = Ocp(T=T_hor)

x_s = ocp.state()  # cart position  [m]
x_dot = ocp.state()  # cart velocity  [m/s]
theta = ocp.state()  # pole angle     [rad]  (0=hanging, pi=upright)
theta_dot = ocp.state()  # pole ang. vel. [rad/s]

u = ocp.control(1, order=0)

X0 = ocp.parameter(4)  # initial state parameter

# Nonlinear cart-pendulum dynamics — exact match to cartpend_dxdt (g < 0)
sin_theta = cas.sin(theta)
cos_theta = cas.cos(theta)
D = 1.0 / (L * (M + m * (1 - cos_theta**2)))
b = m * L * theta_dot**2 * sin_theta - d * x_dot + u
ocp.set_der(x_s, x_dot)
ocp.set_der(x_dot, D * (-m * L * g * cos_theta * sin_theta + L * b))
ocp.set_der(theta, theta_dot)
ocp.set_der(theta_dot, D * ((m + M) * g * sin_theta - cos_theta * b))

# Running cost
theta_cost = 4 * cas.sin((theta - np.pi) / 2) ** 2
E_sym = 0.5 * m * L**2 * theta_dot**2 - m * g * L * (1 - cos_theta)
t_frac = ocp.t / T_hor
ocp.add_objective(
    ocp.integral(
        Q_theta * t_frac * theta_cost
        + Q_x * x_s**2
        - Q_E * (1 - t_frac) * E_sym
        + R_u * u**2
    )
)

# Terminal cost
ocp.add_objective(
    Q_T
    * (
        Q_theta * 4 * cas.sin((ocp.at_tf(theta) - np.pi) / 2) ** 2
        + Q_x * ocp.at_tf(x_s) ** 2
        + Q_thdot * ocp.at_tf(theta_dot) ** 2
        + Q_xdot * ocp.at_tf(x_dot) ** 2
    )
)

ocp.subject_to(ocp.at_t0(cas.vertcat(x_s, x_dot, theta, theta_dot)) == X0)
ocp.subject_to(-max_u <= (u <= max_u))

ipopt_opts = {"ipopt": {"print_level": 5, "max_iter": 2000}}
ipopt_opts["expand"] = True
ipopt_opts["print_time"] = True
ocp.solver("ipopt", ipopt_opts)
ocp.method(MultipleShooting(N=N, M=1, intg="rk"))


# ---- Helpers ----------------------------------------------------------------


def cartpole_ode(state, u_val):
    x_, xd_, th_, thd_ = state
    sin_th = np.sin(th_)
    cos_th = np.cos(th_)
    D_ = 1.0 / (L * (M + m * (1 - cos_th**2)))
    b_ = m * L * thd_**2 * sin_th - d * xd_ + u_val
    return np.array(
        [
            xd_,
            D_ * (-m * L * g * cos_th * sin_th + L * b_),
            thd_,
            D_ * ((m + M) * g * sin_th - cos_th * b_),
        ]
    )


def simulate_rk4(x0, u_seq):
    """Simulate cart-pendulum (RK4). Returns state array of shape (len(u_seq)+1, 4)."""
    state = np.array(x0, dtype=float)
    states = [state.copy()]
    for u_k in u_seq:
        k1 = cartpole_ode(state, u_k)
        k2 = cartpole_ode(state + 0.5 * tau * k1, u_k)
        k3 = cartpole_ode(state + 0.5 * tau * k2, u_k)
        k4 = cartpole_ode(state + tau * k3, u_k)
        state = state + (tau / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        states.append(state.copy())
    return np.array(states)


# ---- API --------------------------------------------------------------------


def make_u_guess(x0, u_init, max_thetadot=4.3):
    """Build an N-step control guess: apply u_init at full force until theta
    first enters the safe zone (pi/2 < theta < 3*pi/2) OR |theta_dot| exceeds
    max_thetadot, then zero the control and let the OCP take over from there.
    """
    sim_states = simulate_rk4(x0, np.full(N, float(u_init)))
    theta_wrapped = sim_states[:, 2] % (2 * np.pi)
    in_safe = (theta_wrapped > np.pi / 2) & (theta_wrapped < 3 * np.pi / 2)
    too_fast = np.abs(sim_states[:, 3]) > max_thetadot
    crossings = np.where(in_safe | too_fast)[0]
    u_guess = np.zeros(N)
    if len(crossings) > 0:
        k = int(crossings[0])
        u_guess[:k] = u_init
    else:
        u_guess[:] = u_init  # never crossed — keep full force
    return u_guess


def solve_swingup_ocp(x0, u_guess):
    """Solve the swing-up OCP from initial state x0.

    Parameters
    ----------
    x0 : array-like, shape (4,)
        Initial state [x, x_dot, theta, theta_dot].
    u_guess : array-like, shape (N,)
        Initial guess for the N-step control sequence.

    Returns
    -------
    u_sol : ndarray (N,) or None
        Optimal control sequence; None if the solver failed.
    solve_ok : bool
    t_solve : float
        Wall-clock solve time [s].
    sim_states : ndarray (N+1, 4)
        State trajectory from simulating u_guess (useful for plotting).
    t_s : ndarray (N+1,) or None
    theta_sol : ndarray (N+1,) or None
    x_sol : ndarray (N+1,) or None
    cost : float or None
        Optimal objective value; None if the solver failed.
    """
    x0 = np.asarray(x0, dtype=float)
    u_guess = np.asarray(u_guess, dtype=float)

    sim_states = simulate_rk4(x0, u_guess)

    ocp.set_initial(x_s, sim_states[:, 0])
    ocp.set_initial(x_dot, sim_states[:, 1])
    ocp.set_initial(theta, sim_states[:, 2])
    ocp.set_initial(theta_dot, sim_states[:, 3])
    ocp.set_initial(u, u_guess)
    ocp.set_value(X0, x0)

    print(f"Solving OCP (N={N}, T_hor={T_hor:.1f} s)...")
    t0 = time.perf_counter()
    solve_ok = True
    t_s = theta_sol = x_sol = u_sol = None
    cost = None
    try:
        sol = ocp.solve()
        t_s, x_sol = sol.sample(x_s, grid="control")
        _, theta_sol = sol.sample(theta, grid="control")
        _, u_sol = sol.sample(u, grid="control")
        cost = float(ocp._method.opti.value(ocp._method.opti.f))
    except Exception as e:
        print(f"Solver failed: {e}")
        solve_ok = False
    t_solve = time.perf_counter() - t0
    print(f"{'Solved' if solve_ok else 'Failed'} in {t_solve:.2f} s")

    return u_sol, solve_ok, t_solve, sim_states, t_s, theta_sol, x_sol, cost


def plot_solution(
    u_guess, sim_states, t_s, theta_sol, x_sol, u_sol, solve_ok, t_solve
):
    """Plot initial guess and OCP solution trajectories."""
    guess_deg = np.degrees(sim_states[:, 2] % (2 * np.pi))

    fig, axes = plt.subplots(3, 1, figsize=(10, 7), sharex=True)

    axes[0].plot(
        t_nodes,
        guess_deg,
        color="C1",
        marker=".",
        linestyle="",
        label="initial guess",
    )
    if solve_ok:
        theta_deg = np.degrees(theta_sol % (2 * np.pi))
        axes[0].plot(
            t_s,
            theta_deg,
            marker=".",
            linestyle="",
            color="C0",
            label="solution",
        )
    axes[0].axhline(
        180, color="k", linestyle="--", linewidth=0.8, label="target (upright)"
    )
    axes[0].set_ylim(-15, 375)
    axes[0].set_yticks([0, 90, 180, 270, 360])
    axes[0].set_ylabel("theta (deg)")
    axes[0].legend(loc="upper left")

    axes[1].plot(
        t_nodes,
        sim_states[:, 0],
        color="C1",
        marker=".",
        linestyle="",
        label="initial guess",
    )
    if solve_ok:
        axes[1].step(t_s, x_sol, where="post", color="C0", label="solution")
    axes[1].axhline(0, color="k", linestyle="--", linewidth=0.8)
    axes[1].set_ylabel("x (m)")
    axes[1].legend(loc="upper right")

    axes[2].plot(
        t_nodes[:-1],
        u_guess,
        color="C1",
        marker=".",
        linestyle="",
        label="initial guess",
    )
    if solve_ok:
        axes[2].step(t_s, u_sol, where="post", color="C0", label="solution")
    axes[2].axhline(0, color="k", linestyle="--", linewidth=0.8)
    axes[2].set_ylabel("u (N)")
    axes[2].set_xlabel("time (s)")
    axes[2].legend(loc="upper right")

    status = (
        f"solved in {t_solve:.1f} s"
        if solve_ok
        else f"FAILED in {t_solve:.1f} s"
    )
    fig.suptitle(f"Swing-up OCP  (N={N})  —  {status}")
    plt.tight_layout()
    plt.show()


# ---- Entry point ------------------------------------------------------------

if __name__ == "__main__":
    th0 = 0.0 * np.pi / 4  # starting angle (theta=pi is upright)

    excitation_steps = [(0, 200.0), (3, -100.0), (11, 200.0), (20, 0.0)]
    u_guess = np.zeros(N)
    for start, val in excitation_steps:
        u_guess[start:] = val

    x0 = np.array([0.0, 0.0, th0, 0.0])
    u_sol, solve_ok, t_solve, sim_states, t_s, theta_sol, x_sol, cost = (
        solve_swingup_ocp(x0, u_guess)
    )
    plot_solution(
        u_guess, sim_states, t_s, theta_sol, x_sol, u_sol, solve_ok, t_solve
    )
