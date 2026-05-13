"""
Tests for CartPoleBT Gymnasium environments.

Run with: pytest gym_CartPole_BT/tests/

TODO:
- Test with Euler integration option
"""

import numpy as np
import pytest
import gymnasium as gym
import gym_CartPole_BT
from gym_CartPole_BT.systems.cartpend import cartpend_dxdt, cartpend_ss
from numpy.testing import assert_allclose, assert_array_equal

SEED = 0
ACTION = np.array([1.0])

ENV_NAMES = [
    "CartPole-BT-v1",
    "CartPole-BT-dL-v1",
    "CartPole-BT-dH-v1",
    "CartPole-BT-vL-v1",
    "CartPole-BT-vH-v1",
    "CartPole-BT-dL-vL-v1",
    "CartPole-BT-dH-vH-v1",
    "CartPole-BT-dL-vL-nL-v1",
    "CartPole-BT-dL-vL-nH-v1",
    "CartPole-BT-p2-v1",
    "CartPole-BT-p2-dL-v1",
    "CartPole-BT-p2-dH-v1",
    "CartPole-BT-p2-vL-v1",
    "CartPole-BT-p2-vH-v1",
    "CartPole-BT-p2-dL-nL-v1",
    "CartPole-BT-p2-dL-nH-v1",
    "CartPole-BT-x2-v1",
    "CartPole-BT-x2-dL-v1",
    "CartPole-BT-x2-dH-v1",
    "CartPole-BT-x2-dL-nL-v1",
    "CartPole-BT-x2-dL-nH-v1",
    "CartPole-BT-r2-a1-v1",
    "CartPole-BT-r2-a1-dL-v1",
    "CartPole-BT-r2-a1-dH-v1",
    "CartPole-BT-r2-a1-dL-nL-v1",
    "CartPole-BT-r2-a1-dL-nH-v1",
]

DETERMINISTIC_ENVS = [
    "CartPole-BT-v1",
    "CartPole-BT-p2-v1",
    "CartPole-BT-x2-v1",
    "CartPole-BT-r2-a1-v1",
]

STOCHASTIC_ENVS = [n for n in ENV_NAMES if n not in DETERMINISTIC_ENVS]

VARIANCE_LEVELS = {None: 0.0, "low": 0.01, "high": 0.2}

# Golden values generated with seed=0 and action=[1.0] on 2026-05-12.
# All observations and rewards are float32; literals use minimum digits for
# exact float32 round-trip (numpy repr guarantee).
GOLDEN = {
    "CartPole-BT-v1": {
        "reset_obs": np.array([0.0, 0.0, 3.1415927, 0.0], dtype=np.float32),
        "step_obs": np.array(
            [0.00024922087, 0.009954324, 3.1417174, 0.0049875826],
            dtype=np.float32,
        ),
        "step_reward": np.float32(-7.7659365e-08),
    },
    "CartPole-BT-dL-v1": {
        "reset_obs": np.array([0.0, 0.0, 3.1415927, 0.0], dtype=np.float32),
        "step_obs": np.array(
            [0.00024922087, 0.009954324, 3.1417174, 0.005050448],
            dtype=np.float32,
        ),
        "step_reward": np.float32(-7.7659365e-08),
    },
    "CartPole-BT-dH-v1": {
        "reset_obs": np.array([0.0, 0.0, 3.1415927, 0.0], dtype=np.float32),
        "step_obs": np.array(
            [0.00024922087, 0.009954324, 3.1417174, 0.006244885],
            dtype=np.float32,
        ),
        "step_reward": np.float32(-7.7659365e-08),
    },
    "CartPole-BT-vL-v1": {
        "reset_obs": np.array(
            [0.0012573022, -0.0013210486, 3.147997, 0.0010490011],
            dtype=np.float32,
        ),
        "step_obs": np.array(
            [0.001456818, 0.009287755, 3.1482224, 0.007975322],
            dtype=np.float32,
        ),
        "step_reward": np.float32(-4.6075314e-05),
    },
    "CartPole-BT-vH-v1": {
        "reset_obs": np.array(
            [0.025146045, -0.026420973, 3.2696772, 0.020980023],
            dtype=np.float32,
        ),
        "step_obs": np.array(
            [0.024395736, -0.0035970462, 3.2718108, 0.06445045],
            dtype=np.float32,
        ),
        "step_reward": np.float32(-0.017551888),
    },
    "CartPole-BT-dL-vL-v1": {
        "reset_obs": np.array(
            [0.0012573022, -0.0013210486, 3.147997, 0.0010490011],
            dtype=np.float32,
        ),
        "step_obs": np.array(
            [0.001456818, 0.009287755, 3.1482224, 0.0077074873],
            dtype=np.float32,
        ),
        "step_reward": np.float32(-4.6075314e-05),
    },
    "CartPole-BT-dH-vH-v1": {
        "reset_obs": np.array(
            [0.025146045, -0.026420973, 3.2696772, 0.020980023],
            dtype=np.float32,
        ),
        "step_obs": np.array(
            [0.024395736, -0.0035970462, 3.2718108, 0.05909376],
            dtype=np.float32,
        ),
        "step_reward": np.float32(-0.017551888),
    },
    "CartPole-BT-dL-vL-nL-v1": {
        "reset_obs": np.array(
            [-0.0040993914, 0.009526803, 3.170756, 0.0109668095],
            dtype=np.float32,
        ),
        "step_obs": np.array(
            [-0.011197397, -0.009410478, 3.1489437, -0.01672421],
            dtype=np.float32,
        ),
        "step_reward": np.float32(-4.6075314e-05),
    },
    "CartPole-BT-dL-vL-nH-v1": {
        "reset_obs": np.array(
            [-0.025526166, 0.05291821, 3.2617924, 0.050638046],
            dtype=np.float32,
        ),
        "step_obs": np.array(
            [-0.061814256, -0.084203415, 3.1518288, -0.11411487],
            dtype=np.float32,
        ),
        "step_reward": np.float32(-4.6075314e-05),
    },
    "CartPole-BT-p2-v1": {
        "reset_obs": np.array([0.0, 3.1415927], dtype=np.float32),
        "step_obs": np.array([0.00024922087, 3.1417174], dtype=np.float32),
        "step_reward": np.float32(-7.7659365e-08),
    },
    "CartPole-BT-p2-dL-v1": {
        "reset_obs": np.array([0.0, 3.1415927], dtype=np.float32),
        "step_obs": np.array([0.00024922087, 3.1417174], dtype=np.float32),
        "step_reward": np.float32(-7.7659365e-08),
    },
    "CartPole-BT-p2-dH-v1": {
        "reset_obs": np.array([0.0, 3.1415927], dtype=np.float32),
        "step_obs": np.array([0.00024922087, 3.1417174], dtype=np.float32),
        "step_reward": np.float32(-7.7659365e-08),
    },
    "CartPole-BT-p2-vL-v1": {
        "reset_obs": np.array([0.0012573022, 3.147997], dtype=np.float32),
        "step_obs": np.array([0.001456818, 3.1482224], dtype=np.float32),
        "step_reward": np.float32(-4.6075314e-05),
    },
    "CartPole-BT-p2-vH-v1": {
        "reset_obs": np.array([0.025146045, 3.2696772], dtype=np.float32),
        "step_obs": np.array([0.024395736, 3.2718108], dtype=np.float32),
        "step_reward": np.float32(-0.017551888),
    },
    "CartPole-BT-p2-dL-nL-v1": {
        "reset_obs": np.array([0.0012573022, 3.139287], dtype=np.float32),
        "step_obs": np.array([0.001298222, 3.1323683], dtype=np.float32),
        "step_reward": np.float32(-7.7659365e-08),
    },
    "CartPole-BT-p2-dL-nH-v1": {
        "reset_obs": np.array([0.006286511, 3.1300645], dtype=np.float32),
        "step_obs": np.array([0.0054942267, 3.0949714], dtype=np.float32),
        "step_reward": np.float32(-7.7659365e-08),
    },
    "CartPole-BT-x2-v1": {
        "reset_obs": np.array([-1.0, 0.0, 3.1415927, 0.0], dtype=np.float32),
        "step_obs": np.array(
            [-0.9997508, 0.009954324, 3.1417174, 0.0049875826],
            dtype=np.float32,
        ),
        "step_reward": np.float32(-3.9990034),
    },
    "CartPole-BT-x2-dL-v1": {
        "reset_obs": np.array([-1.0, 0.0, 3.1415927, 0.0], dtype=np.float32),
        "step_obs": np.array(
            [-0.9997508, 0.009954324, 3.1417174, 0.005050448], dtype=np.float32
        ),
        "step_reward": np.float32(-3.9990034),
    },
    "CartPole-BT-x2-dH-v1": {
        "reset_obs": np.array([-1.0, 0.0, 3.1415927, 0.0], dtype=np.float32),
        "step_obs": np.array(
            [-0.9997508, 0.009954324, 3.1417174, 0.006244885], dtype=np.float32
        ),
        "step_reward": np.float32(-3.9990034),
    },
    "CartPole-BT-x2-dL-nL-v1": {
        "reset_obs": np.array(
            [-0.9987427, -0.003963146, 3.1527703, 0.0010985114],
            dtype=np.float32,
        ),
        "step_obs": np.array(
            [-0.9961348, 0.049074326, 3.1582472, -0.0026497503],
            dtype=np.float32,
        ),
        "step_reward": np.float32(-3.9990034),
    },
    "CartPole-BT-x2-dL-nH-v1": {
        "reset_obs": np.array(
            [-0.9937135, -0.01981573, 3.1974802, 0.0054925573],
            dtype=np.float32,
        ),
        "step_obs": np.array(
            [-0.98167104, 0.20555434, 3.224366, -0.032127745], dtype=np.float32
        ),
        "step_reward": np.float32(-3.9990034),
    },
    "CartPole-BT-r2-a1-v1": {
        "reset_obs": np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32),
        "step_obs": np.array(
            [0.00024911689, 0.009946023, -0.00012442857, -0.0049626287],
            dtype=np.float32,
        ),
        "step_reward": np.float32(-4.0),
    },
    "CartPole-BT-r2-a1-dL-v1": {
        "reset_obs": np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32),
        "step_obs": np.array(
            [0.00024911689, 0.009946023, -0.00012442857, -0.0048997635],
            dtype=np.float32,
        ),
        "step_reward": np.float32(-4.0),
    },
    "CartPole-BT-r2-a1-dH-v1": {
        "reset_obs": np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32),
        "step_obs": np.array(
            [0.00024911689, 0.009946023, -0.00012442857, -0.0037053265],
            dtype=np.float32,
        ),
        "step_reward": np.float32(-4.0),
    },
    "CartPole-BT-r2-a1-dL-nL-v1": {
        "reset_obs": np.array(
            [0.0012573022, -0.003963146, 0.011177484, 0.0010985114],
            dtype=np.float32,
        ),
        "step_obs": np.array(
            [0.0038650674, 0.049066022, 0.016405253, -0.012599962],
            dtype=np.float32,
        ),
        "step_reward": np.float32(-4.0),
    },
    "CartPole-BT-r2-a1-dL-nH-v1": {
        "reset_obs": np.array(
            [0.006286511, -0.01981573, 0.055887416, 0.0054925573],
            dtype=np.float32,
        ),
        "step_obs": np.array(
            [0.01832887, 0.20554604, 0.08252397, -0.042077955],
            dtype=np.float32,
        ),
        "step_reward": np.float32(-4.0),
    },
}


def test_env_names_unique():
    assert len(ENV_NAMES) == len(set(ENV_NAMES))


@pytest.mark.parametrize("name", ENV_NAMES)
def test_env_properties(name):
    """Physical parameters, config flags, spaces, and initial conditions."""
    env = gym.make(name)
    u = env.unwrapped

    assert u.length == 2
    assert u.masspole == 1
    assert u.masscart == 5
    assert u.friction == 1
    assert u.tau == 0.05
    assert u.gravity == -10.0
    assert u.time_step == 0
    assert u.variance_levels == VARIANCE_LEVELS

    expected_steps = 200 if "-r2-a1-" in name else 100
    assert env.spec.max_episode_steps == expected_steps

    if "-dL" in name:
        assert u.disturbances == "low"
    elif "-dH" in name:
        assert u.disturbances == "high"
    else:
        assert u.disturbances is None

    if "-vL" in name:
        assert u.initial_state_variance == "low"
    elif "-vH" in name:
        assert u.initial_state_variance == "high"
    else:
        assert u.initial_state_variance is None

    if "-nL" in name or "-nH" in name:
        assert u.measurement_noise is not None
    else:
        assert u.measurement_noise is None

    assert u.measurement_bias is None

    if "-r2" in name:
        assert u.reward_function == "sinthetasqr_xsqr"
    else:
        assert u.reward_function == "thetasqr_xsqr"

    assert env.action_space.shape == (1,)
    assert len(u.state_bounds) == 4

    if "-p2" in name:
        assert_array_equal(u.output_matrix, ((1, 0, 0, 0), (0, 0, 1, 0)))
    else:
        assert_array_equal(u.output_matrix, np.eye(4))
    assert u.output_matrix.dtype == np.dtype("float32")

    if "-x2" in name:
        assert_allclose(u.initial_state, [-1, 0, np.pi, 0], atol=1e-6)
        assert_allclose(u.goal_state, [1, 0, np.pi, 0], atol=1e-6)
    elif "-a1" in name:
        assert_allclose(u.initial_state, [0, 0, 0, 0], atol=1e-6)
        assert_allclose(u.goal_state, [0, 0, np.pi, 0], atol=1e-6)
    else:
        assert_allclose(u.initial_state, [0, 0, np.pi, 0], atol=1e-6)
        assert_allclose(u.goal_state, [0, 0, np.pi, 0], atol=1e-6)

    assert u.initial_state.dtype == np.dtype("float32")
    assert u.goal_state.dtype == np.dtype("float32")

    env.close()


@pytest.mark.parametrize("name", ENV_NAMES)
def test_env_simulation(name):
    """Behavioral checks: obs dtypes/shapes, noisy vs deterministic reset,
    step counter, state changes after step."""
    env = gym.make(name)
    u = env.unwrapped
    noisy = "-vL" in name or "-vH" in name or "-nL" in name or "-nH" in name

    obs0, _ = env.reset(seed=SEED)
    assert obs0.dtype == np.dtype("float32")
    assert obs0.shape == env.observation_space.shape
    assert u.state.shape == (4,)
    assert u.state.dtype == np.dtype("float32")

    if noisy:
        assert not np.array_equal(obs0, u.output(u.initial_state))
    else:
        assert_array_equal(obs0, u.output(u.initial_state))

    obs1, reward, terminated, truncated, _ = env.step(ACTION)
    assert obs1.dtype == np.dtype("float32")
    expected_shape = (2,) if "-p2" in name else (4,)
    assert obs1.shape == expected_shape
    assert not (terminated or truncated)

    obs2, _, _, _, _ = env.step(np.array([-250.0]))  # clipped at max_force
    assert u.time_step == 2
    assert not np.isclose(obs1, obs2).all()

    obs3, _ = env.reset()
    assert u.time_step == 0
    if noisy:
        assert not np.isclose(obs3, obs0).all()
    else:
        assert_array_equal(obs3, u.output(u.initial_state))

    # Deterministic envs reproduce identical output after reset.
    obs4, _, _, _, _ = env.step(ACTION)
    if name in DETERMINISTIC_ENVS:
        assert_array_equal(obs4, obs1)

    env.close()


@pytest.mark.parametrize("name", STOCHASTIC_ENVS)
def test_stochastic_repeatability(name):
    """Same seed reproduces identical outputs; different seeds diverge."""
    inputs = [np.array([v]) for v in [100.0, 100.0, 10.0, -100.0, -200.0]]

    def run(seed):
        env = gym.make(name)
        obs, _ = env.reset(seed=seed)
        results = [obs] + [env.step(u) for u in inputs]
        env.close()
        return results

    r1 = run(seed=1)
    r2 = run(seed=1)
    r3 = run(seed=10)

    # Same seed: initial obs and all subsequent obs/rewards match.
    assert_array_equal(r1[0], r2[0])
    assert_array_equal(r1[-1][0], r2[-1][0])
    assert_allclose(r1[1][1], r2[1][1])
    assert_allclose(r1[-1][1], r2[-1][1])

    # Output changes after first step.
    assert not np.isclose(r1[0], r1[1][0]).all()

    # Different seeds give different outputs after a step.
    assert not np.array_equal(r1[2][0], r3[2][0])


@pytest.mark.parametrize("name", ENV_NAMES)
def test_reset_and_step(name):
    """Reset and step match golden values; second reset with same seed
    reproduces identical outputs (no state leak between episodes)."""
    golden = GOLDEN[name]
    env = gym.make(name)

    obs0, _ = env.reset(seed=SEED)
    assert_array_equal(obs0, golden["reset_obs"])

    obs1, reward, _, _, _ = env.step(ACTION)
    assert_array_equal(obs1, golden["step_obs"])
    assert_array_equal(reward, golden["step_reward"])

    obs2, _ = env.reset(seed=SEED)
    assert_array_equal(obs2, obs0)

    obs3, reward2, _, _, _ = env.step(ACTION)
    assert_array_equal(obs3, obs1)
    assert_array_equal(reward2, reward)

    env.close()


def test_reset_options():
    """reset(options={"low": l, "high": h}) perturbs around initial_state,
    not around the origin."""
    env = gym.make("CartPole-BT-x2-v1")
    initial_state = env.unwrapped.initial_state.copy()  # [-1, 0, pi, 0]
    low, high = -0.1, 0.1

    for seed in range(20):
        obs, _ = env.reset(seed=seed, options={"low": low, "high": high})
        assert np.all(obs >= initial_state + low - 1e-6), (
            f"seed={seed}: obs {obs} below initial_state+low "
            f"{initial_state + low}"
        )
        assert np.all(obs <= initial_state + high + 1e-6), (
            f"seed={seed}: obs {obs} above initial_state+high "
            f"{initial_state + high}"
        )

    env.close()


def test_measurement_noise():
    """Check measurement_noise level and noise_levels dict."""
    nL_sigma = [0.01, 0.03, np.radians(1.0), np.radians(0.6)]
    nH_sigma = [0.05, 0.15, np.radians(5.0), np.radians(3.0)]
    cases = [
        ("CartPole-BT-dL-vL-nL-v1", "low"),
        ("CartPole-BT-dL-vL-nH-v1", "high"),
        ("CartPole-BT-p2-dL-nL-v1", "low"),
        ("CartPole-BT-p2-dL-nH-v1", "high"),
        ("CartPole-BT-x2-dL-nL-v1", "low"),
        ("CartPole-BT-x2-dL-nH-v1", "high"),
    ]
    for name, expected_level in cases:
        env = gym.make(name)
        assert env.unwrapped.measurement_noise == expected_level
        assert_allclose(env.unwrapped.noise_levels["low"], nL_sigma, rtol=1e-6)
        assert_allclose(
            env.unwrapped.noise_levels["high"], nH_sigma, rtol=1e-6
        )
        assert env.unwrapped.noise_levels[None] is None
        env.close()


def test_measurement_bias():
    """Check measurement_bias is applied exactly to observations."""
    bias = [0.1, -0.05, 0.02, -0.01]
    bias_arr = np.array(bias, dtype=np.float32)
    env = gym.make("CartPole-BT-dL-v1", measurement_bias=bias)
    assert_allclose(env.unwrapped.measurement_bias, bias)
    assert env.unwrapped.measurement_bias.dtype == np.dtype("float32")
    assert env.unwrapped.measurement_noise is None

    obs, _ = env.reset(seed=0)
    true_out = env.unwrapped.output(env.unwrapped.state)
    assert_allclose(obs, true_out + bias_arr, rtol=1e-6)

    obs, reward, _, _, _ = env.step(np.array([1.0]))
    true_out = env.unwrapped.output(env.unwrapped.state)
    assert_allclose(obs, true_out + bias_arr, rtol=1e-6)
    assert isinstance(float(reward), float)
    env.close()


@pytest.mark.parametrize(
    "name",
    [
        "CartPole-BT-v1",
        "CartPole-BT-p2-v1",
        "CartPole-BT-x2-v1",
    ],
)
def test_render_rgb_array(name):
    """render() in rgb_array mode returns a (H, W, 3) uint8 array."""
    env = gym.make(name, render_mode="rgb_array")
    env.reset(seed=SEED)
    frame = env.render()
    assert isinstance(frame, np.ndarray)
    assert frame.dtype == np.uint8
    assert frame.ndim == 3 and frame.shape[2] == 3
    assert frame.shape[:2] == (
        env.unwrapped.screen_height,
        env.unwrapped.screen_width,
    )
    env.close()


@pytest.mark.parametrize(
    "name",
    [
        "CartPole-BT-v1",
        "CartPole-BT-dL-vL-nH-v1",
        "CartPole-BT-x2-v1",
        "CartPole-BT-r2-a1-v1",
    ],
)
def test_check_env(name):
    """Gymnasium's built-in env_checker passes for representative envs."""
    from gymnasium.utils.env_checker import check_env

    env = gym.make(name)
    # skip_render_check avoids opening a "human" window during testing;
    # rendering is covered separately in test_render_rgb_array.
    check_env(env.unwrapped, skip_render_check=True)
    env.close()


def test_cartpend():
    """Check calculations in cartpend_dxdt and cartpend_ss."""
    m, M, L, g, d = 1, 5, 2, -10, 1

    x_cases = {
        0: [0, 0, 0, 0],
        1: [0, 0, np.pi, 0],
        2: [0, 0, 0, 0],
        3: [0, 0, np.pi, 0],
        4: [2.260914, 0.026066, 0.484470, -0.026480],
    }
    u_cases = {0: 0.0, 1: 0.0, 2: 1.0, 3: 1.0, 4: -0.59601}
    # dy values calculated with MATLAB script from Steven L. Brunton's Control
    # Bootcamp
    dx_expected = {
        0: [0.0, 0.0, 0.0, 0.0],
        1: [0.0, -2.44929360e-16, 0.0, -7.34788079e-16],
        2: [0.0, 0.2, 0.0, -0.1],
        3: [0.0, 0.2, 0.0, 0.1],
        4: [0.026066, 0.670896, -0.026480, -2.625542],
    }
    for i, u in u_cases.items():
        x = np.array(x_cases[i])
        assert_allclose(
            cartpend_dxdt(0.0, x, m=m, M=M, L=L, g=g, d=d, u=u),
            dx_expected[i],
            atol=1e-6,
        )

    # K values calculated with MATLAB script from Steven L. Brunton's Control
    # Bootcamp
    ss_expected = {
        1: (  # pendulum up
            np.array(
                [[0, 1, 0, 0], [0, -0.2, 2, 0], [0, 0, 0, 1], [0, -0.1, 6, 0]],
                dtype=float,
            ),
            np.array([[0], [0.2], [0], [0.1]]),
        ),
        -1: (  # pendulum down
            np.array(
                [[0, 1, 0, 0], [0, -0.2, 2, 0], [0, 0, 0, 1], [0, 0.1, -6, 0]],
                dtype=float,
            ),
            np.array([[0], [0.2], [0], [-0.1]]),
        ),
    }
    for s, (A_exp, B_exp) in ss_expected.items():
        A, B = cartpend_ss(m=m, M=M, L=L, g=g, d=d, s=s)
        assert_allclose(A, A_exp)
        assert_allclose(B, B_exp)
