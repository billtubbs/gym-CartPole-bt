#!/usr/bin/env python
# Swing-up trajectory optimisation using Rockit / CasADi / IPOPT.
#
# Solves a single open-loop OCP over a long horizon starting from the
# hanging position.  The environment object is used only to read
# physical parameters.

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
Q_theta = 5.0  # running angle cost weight
Q_x = 1.0  # running cart-position cost weight
Q_E = 0.002  # running total-energy incentive weight (maximise KE+PE at each step)
Q_T = 50.0  # terminal vs. per-step running cost multiplier
Q_PE = 10.0  # terminal PE incentive (400/40: matches state-error magnitude at theta=0)
Q_thdot = 0.1  # terminal theta_dot weight
Q_xdot = 0.1  # terminal x_dot weight
R_u = 1e-5  # control effort regularisation
x_limit = 3.0  # soft cart position limit [m]
Q_xlim = 5.0  # weight on soft position limit violation

# ---- Build OCP --------------------------------------------------------------
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

# Running cost: penalise distance from upright (theta=pi) and cart centre
theta_cost = (
    4
    * cas.sin((theta - np.pi) / 2)
    ** 2  # cost=0 at upright (th=pi), cost=4 at hanging (th=0)
)
# E_ref_val = -2.0 * m * g * L  # 40 J — energy at upright rest
E_sym = 0.5 * m * L**2 * theta_dot**2 - m * g * L * (
    1 - cos_theta
)  # KE + PE; 0 at hanging rest, 40 J at upright rest
# energy_gap = cas.fmax(0.0, E_ref_val - E_sym)  # positive only when under-energised
# x_viol = cas.fmax(0.0, x_s**2 - x_limit**2)  # non-zero outside ±x_limit
t_frac = ocp.t / T_hor  # 0 at t=0, 1 at t=T_hor
ocp.add_objective(
    ocp.integral(
        Q_theta * t_frac * theta_cost  # state error: 0% → 100%
        + Q_x * x_s**2
        - Q_E * (1 - t_frac) * E_sym  # energy incentive: 100% → 0%
        # + Q_xlim * x_viol
        + R_u * u**2
    )
)

# Terminal cost: squared state error vs. goal [0, 0, pi, 0]
ocp.add_objective(
    Q_T
    * (
        Q_theta * 4 * cas.sin((ocp.at_tf(theta) - np.pi) / 2) ** 2
        + Q_x * ocp.at_tf(x_s) ** 2
        + Q_thdot * ocp.at_tf(theta_dot) ** 2
        + Q_xdot * ocp.at_tf(x_dot) ** 2
    )
)

# Terminal energy incentive: reward high total mechanical energy regardless of goal attainment.
# E = KE + PE: 0 J at hanging rest, 40 J at upright rest.
# KE_tf = 0.5 * m * L**2 * ocp.at_tf(theta_dot) ** 2
# PE_tf = -m * g * L * (1 - cas.cos(ocp.at_tf(theta)))
# E_tf = KE_tf + PE_tf
# ocp.add_objective(-Q_PE * E_tf)

# Initial condition
ocp.subject_to(ocp.at_t0(cas.vertcat(x_s, x_dot, theta, theta_dot)) == X0)

# Control constraint
ocp.subject_to(-max_u <= (u <= max_u))

ipopt_opts = {"ipopt": {"print_level": 5, "max_iter": 2000}}
ipopt_opts["expand"] = True
ipopt_opts["print_time"] = True
ocp.solver("ipopt", ipopt_opts)
ocp.method(MultipleShooting(N=N, M=1, intg="rk"))

# ---- Initial guess: open-loop simulation with resonant bang-bang u ----------
th0 = 0.0 * np.pi / 4  # starting angle (theta=pi is upright)

omega_n = np.sqrt(abs(g) / L)  # ~2.236 rad/s — pendulum natural frequency

t_nodes = np.linspace(0, T_hor, N + 1)

# Excitation schedule: list of (start_step, value) applied in order
excitation_steps = [(0, 200.0), (3, -100.0), (11, 200.0), (20, 0.0)]
u_guess = np.zeros(N + 1)
for start, val in excitation_steps:
    u_guess[start:] = val


# Simulate cart-pendulum dynamics (RK4) under u_guess to get consistent state guess
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


state = np.array([0.0, 0.0, th0, 0.0])
sim_states = [state.copy()]
for k in range(N):
    u_k = float(u_guess[k])
    k1 = cartpole_ode(state, u_k)
    k2 = cartpole_ode(state + 0.5 * tau * k1, u_k)
    k3 = cartpole_ode(state + 0.5 * tau * k2, u_k)
    k4 = cartpole_ode(state + tau * k3, u_k)
    state = state + (tau / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
    sim_states.append(state.copy())

sim_states = np.array(sim_states)  # (N+1, 4)
x_guess = sim_states[:, 0]
xd_guess = sim_states[:, 1]
th_guess = sim_states[:, 2]
thd_guess = sim_states[:, 3]

ocp.set_initial(x_s, x_guess)
ocp.set_initial(x_dot, xd_guess)
ocp.set_initial(theta, th_guess)
ocp.set_initial(theta_dot, thd_guess)
ocp.set_initial(u, u_guess[:-1])  # N control values

ocp.set_value(X0, np.array([0.0, 0.0, th0, 0.0]))  # x=0, ẋ=0, θ=th0, θ̇=0

# ---- Solve ------------------------------------------------------------------
print(f"Solving OCP (N={N}, T_hor={T_hor:.1f} s)...")
t0 = time.perf_counter()
solve_ok = True
try:
    sol = ocp.solve()
except Exception as e:
    print(f"Solver failed: {e}")
    solve_ok = False
t_solve = time.perf_counter() - t0
print(f"{'Solved' if solve_ok else 'Failed'} in {t_solve:.2f} s")

# ---- Extract solution -------------------------------------------------------
if solve_ok:
    t_s, x_sol = sol.sample(x_s, grid="control")
    _, theta_sol = sol.sample(theta, grid="control")
    _, u_sol = sol.sample(u, grid="control")

# ---- Plot -------------------------------------------------------------------
guess_deg = np.degrees(((th_guess + np.pi) % (2 * np.pi)) - np.pi)

fig, axes = plt.subplots(3, 1, figsize=(10, 7), sharex=True)

axes[0].plot(
    t_nodes, guess_deg, color="C1", linestyle="--", label="initial guess"
)
if solve_ok:
    theta_deg = np.degrees(((theta_sol + np.pi) % (2 * np.pi)) - np.pi)
    axes[0].plot(
        t_s, theta_deg, marker=".", linestyle="", color="C0", label="solution"
    )
axes[0].axhline(
    180, color="k", linestyle="--", linewidth=0.8, label="target (upright)"
)
axes[0].axhline(-180, color="k", linestyle="--", linewidth=0.8)
axes[0].set_ylim(-195, 195)
axes[0].set_yticks(np.linspace(-180, 180, 5))
axes[0].set_ylabel("theta (deg)")
axes[0].legend(loc="upper left")

axes[1].plot(
    t_nodes, x_guess, color="C1", linestyle="--", label="initial guess"
)
if solve_ok:
    axes[1].step(t_s, x_sol, where="post", color="C0", label="solution")
axes[1].axhline(0, color="k", linestyle="--", linewidth=0.8)
axes[1].set_ylabel("x (m)")
axes[1].legend(loc="upper right")

axes[2].step(
    t_nodes,
    u_guess,
    where="post",
    color="C1",
    linestyle="--",
    label="initial guess",
)
if solve_ok:
    axes[2].step(t_s, u_sol, where="post", color="C0", label="solution")
axes[2].axhline(0, color="k", linestyle="--", linewidth=0.8)
axes[2].set_ylabel("u (N)")
axes[2].set_xlabel("time (s)")
axes[2].legend(loc="upper right")

status = (
    f"solved in {t_solve:.1f} s" if solve_ok else f"FAILED in {t_solve:.1f} s"
)
fig.suptitle(f"Swing-up OCP  (N={N})  —  {status}")
plt.tight_layout()
plt.show()
