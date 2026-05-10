"""
Unit tests for custom cart-pole environments.

To run these tests use this at the command line:
$ python -m unittest discover -s gym_CartPole_BT/tests/

TODO:
- Test with Euler integration option
"""

import unittest
import numpy as np
import gymnasium as gym
import gym_CartPole_BT
from gym_CartPole_BT.systems.cartpend import cartpend_dxdt, cartpend_ss

from numpy.testing import assert_allclose, assert_array_equal


class TestGymCartPoleBT(unittest.TestCase):
    def test_cartpole_bt_env(self):
        """Check cartpole_bt_env environments working correctly."""

        env_names = [
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
        self.assertEqual(len(env_names), len(set(env_names)))

        variance_levels = {None: 0.0, "low": 0.01, "high": 0.2}

        for name in env_names:
            env = gym.make(name)
            self.assertEqual(env.unwrapped.length, 2)
            self.assertEqual(env.unwrapped.masspole, 1)
            self.assertEqual(env.unwrapped.masscart, 5)
            self.assertEqual(env.unwrapped.friction, 1)
            self.assertEqual(env.unwrapped.time_step, 0)
            self.assertEqual(env.unwrapped.tau, 0.05)
            self.assertEqual(env.unwrapped.gravity, -10.0)
            self.assertEqual(env.spec.max_episode_steps, 100)
            self.assertEqual(env.unwrapped.variance_levels, variance_levels)
            if "-dL" in name:
                self.assertEqual(env.unwrapped.disturbances, "low")
            elif "-dH" in name:
                self.assertEqual(env.unwrapped.disturbances, "high")
            else:
                self.assertIsNone(env.unwrapped.disturbances)
            if "-vL" in name:
                self.assertEqual(env.unwrapped.initial_state_variance, "low")
            elif "-vH" in name:
                self.assertEqual(env.unwrapped.initial_state_variance, "high")
            else:
                self.assertIsNone(env.unwrapped.initial_state_variance)
            if "-nL" in name or "-nH" in name:
                self.assertIsNotNone(env.unwrapped.measurement_noise)
            else:
                self.assertIsNone(env.unwrapped.measurement_noise)
            self.assertIsNone(env.unwrapped.measurement_bias)
            self.assertEqual(env.action_space.shape, (1,))
            self.assertEqual(len(env.unwrapped.state_bounds), 4)
            if "-p2" in name:
                assert_array_equal(
                    env.unwrapped.output_matrix, ((1, 0, 0, 0), (0, 0, 1, 0))
                )
            else:
                assert_array_equal(env.unwrapped.output_matrix, np.eye(4))
            self.assertEqual(
                env.unwrapped.output_matrix.dtype, np.dtype("float32")
            )
            if "-x2" in name:
                assert_allclose(
                    env.unwrapped.initial_state, [-1, 0, np.pi, 0], atol=1e-6
                )
                assert_allclose(
                    env.unwrapped.goal_state, [1, 0, np.pi, 0], atol=1e-6
                )
            elif "-a1" in name:
                assert_allclose(
                    env.unwrapped.initial_state, [0, 0, 0, 0], atol=1e-6
                )
                assert_allclose(
                    env.unwrapped.goal_state, [0, 0, np.pi, 0], atol=1e-6
                )
            else:
                assert_allclose(
                    env.unwrapped.initial_state, [0, 0, np.pi, 0], atol=1e-6
                )
                assert_allclose(
                    env.unwrapped.goal_state, [0, 0, np.pi, 0], atol=1e-6
                )
            if "-r2" in name:
                self.assertEqual(
                    env.unwrapped.reward_function, "sinthetasqr_xsqr"
                )
            else:
                self.assertEqual(
                    env.unwrapped.reward_function, "thetasqr_xsqr"
                )
            self.assertEqual(
                env.unwrapped.initial_state.dtype, np.dtype("float32")
            )
            self.assertEqual(
                env.unwrapped.goal_state.dtype, np.dtype("float32")
            )

            initial_output, _ = env.reset()
            self.assertEqual(initial_output.dtype, np.dtype("float32"))
            self.assertEqual(initial_output.shape, env.observation_space.shape)
            self.assertEqual(env.unwrapped.state.shape, (4,))
            self.assertEqual(env.unwrapped.state.dtype, np.dtype("float32"))
            noisy = (
                "-vL" in name
                or "-vH" in name
                or "-nL" in name
                or "-nH" in name
            )
            if noisy:
                self.assertFalse(
                    np.array_equal(
                        initial_output,
                        env.unwrapped.output(env.unwrapped.initial_state),
                    )
                )
            else:
                self.assertTrue(
                    np.array_equal(
                        initial_output,
                        env.unwrapped.output(env.unwrapped.initial_state),
                    )
                )

            # Simulate one time step
            u = np.array([1.0])
            output_1, reward, terminated, truncated, info = env.step(u)
            self.assertEqual(output_1.dtype, np.dtype("float32"))
            self.assertFalse(terminated or truncated)
            if "-p2" in name:
                self.assertEqual(output_1.shape, (2,))
            else:
                self.assertEqual(output_1.shape, (4,))

            # Simulate 2nd time step
            u = np.array([-250.0])  # Exceeds the limit
            output_2, reward, terminated, truncated, info = env.step(u)
            self.assertEqual(env.unwrapped.time_step, 2)
            self.assertFalse(np.isclose(output_1, output_2).all())

            # Check environment reset
            output_3, _ = env.reset()
            self.assertEqual(env.unwrapped.time_step, 0)
            if noisy:
                self.assertFalse(np.isclose(output_3, initial_output).all())
            else:
                self.assertTrue(
                    np.array_equal(
                        output_3,
                        env.unwrapped.output(env.unwrapped.initial_state),
                    )
                )
            u = np.array([1.0])
            output_1r, reward, terminated, truncated, info = env.step(u)

            # Check deterministic environments produce same output after reset
            deterministic_envs = [
                "CartPole-BT-v1",
                "CartPole-BT-p2-v1",
                "CartPole-BT-x2-v1",
                "CartPole-BT-r2-a1-v1",
            ]
            if name in deterministic_envs:
                self.assertTrue(np.array_equal(output_1r, output_1))

        # Check stochastic environments are repeatable when seed set
        stochastic_envs = [
            name for name in env_names if name not in deterministic_envs
        ]

        def sim_test(env, inputs, seed=None):
            obs, _ = env.reset(seed=seed)
            data = [obs]
            for u in inputs:
                data.append(env.step(u))
            return data

        for name in stochastic_envs:
            seeds = [1, 10]
            envs = [gym.make(name) for seed in seeds]
            data = {env: {} for env in envs}
            inputs = [[100.0], [100.0], [10], [-100], [-200]]
            for env, seed in zip(envs, seeds):
                # Run simulation for several steps
                data[env]["Test 1"] = sim_test(env, inputs, seed)
                # Check output changed after first step
                y1 = data[env]["Test 1"][0]  # initial obs
                y2 = data[env]["Test 1"][1][0]  # obs after 1st step
                self.assertFalse(np.isclose(y1, y2).all())
                # Reset with same seed and repeat
                data[env]["Test 2"] = sim_test(env, inputs, seed)
                # Check initial outputs match
                y2 = data[env]["Test 2"][0]
                self.assertTrue(np.array_equal(y1, y2))
                # Check final outputs match
                y1 = data[env]["Test 1"][-1][0]
                y2 = data[env]["Test 2"][-1][0]
                self.assertTrue(np.array_equal(y1, y2))
                # Check rewards match
                r1 = data[env]["Test 1"][1][1]
                r2 = data[env]["Test 2"][1][1]
                assert_allclose(r1, r2)
                r1 = data[env]["Test 1"][-1][1]
                r2 = data[env]["Test 2"][-1][1]
                assert_allclose(r1, r2)

            # Check different seeds give different outputs
            y1 = data[envs[0]]["Test 1"][2][0]
            y2 = data[envs[1]]["Test 1"][2][0]
            self.assertFalse(np.array_equal(y1, y2))

    def test_measurement_noise(self):
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
            self.assertEqual(env.unwrapped.measurement_noise, expected_level)
            assert_allclose(
                env.unwrapped.noise_levels["low"], nL_sigma, rtol=1e-6
            )
            assert_allclose(
                env.unwrapped.noise_levels["high"], nH_sigma, rtol=1e-6
            )
            self.assertIsNone(env.unwrapped.noise_levels[None])
            env.close()

    def test_measurement_bias(self):
        """Check measurement_bias is applied exactly to observations."""
        bias = [0.1, -0.05, 0.02, -0.01]
        bias_arr = np.array(bias, dtype=np.float32)
        env = gym.make("CartPole-BT-dL-v1", measurement_bias=bias)
        assert_allclose(env.unwrapped.measurement_bias, bias)
        self.assertEqual(
            env.unwrapped.measurement_bias.dtype, np.dtype("float32")
        )
        self.assertIsNone(env.unwrapped.measurement_noise)

        # Bias applied exactly on reset and step (no noise in this env)
        obs, _ = env.reset(seed=0)
        true_out = env.unwrapped.output(env.unwrapped.state)
        assert_allclose(obs, true_out + bias_arr, rtol=1e-6)

        obs, reward, _, _, _ = env.step(np.array([1.0]))
        true_out = env.unwrapped.output(env.unwrapped.state)
        assert_allclose(obs, true_out + bias_arr, rtol=1e-6)
        self.assertIsInstance(float(reward), float)
        env.close()

    def test_cartpend(self):
        """Check calculations in cartpend_dxdt function."""

        m = 1
        M = 5
        L = 2
        g = -10
        d = 1

        x_test_values = {
            0: [0, 0, 0, 0],
            1: [0, 0, np.pi, 0],
            2: [0, 0, 0, 0],
            3: [0, 0, np.pi, 0],
            4: [2.260914, 0.026066, 0.484470, -0.026480],
        }

        test_values = {0: 0.0, 1: 0.0, 2: 1.0, 3: 1.0, 4: -0.59601}

        # dy values calculated with MATLAB script from Steven L. Brunton's Control Bootcamp
        expected_results = {
            0: [0.0, 0.0, 0.0, 0.0],
            1: [0.0, -2.44929360e-16, 0.0, -7.34788079e-16],
            2: [0.0, 0.2, 0.0, -0.1],
            3: [0.0, 0.2, 0.0, 0.1],
            4: [0.026066, 0.670896, -0.026480, -2.625542],
        }

        t = 0.0
        for i, u in test_values.items():
            x = np.array(x_test_values[i])
            dx_calculated = cartpend_dxdt(t, x, m=m, M=M, L=L, g=g, d=d, u=u)
            dx_expected = np.array(expected_results[i])
            assert_allclose(dx_calculated, dx_expected, atol=1e-6)

        # K values calculated with MATLAB script from Steven L. Brunton's Control Bootcamp
        test_values = {
            5: 1,  # Pendulum up position
            6: -1,  # Pendulum down position
        }
        expected_results = {
            5: (
                np.array(
                    [
                        [0.0, 1.0, 0.0, 0.0],
                        [0.0, -0.2, 2.0, 0.0],
                        [0.0, 0.0, 0.0, 1.0],
                        [0.0, -0.1, 6.0, 0.0],
                    ]
                ),
                np.array([[0.0], [0.2], [0.0], [0.1]]),
            ),
            6: (
                np.array(
                    [
                        [0.0, 1.0, 0.0, 0.0],
                        [0.0, -0.2, 2.0, 0.0],
                        [0.0, 0.0, 0.0, 1.0],
                        [0.0, 0.1, -6.0, 0.0],
                    ]
                ),
                np.array([[0.0], [0.2], [0.0], [-0.1]]),
            ),
        }
        for i, s in test_values.items():
            A_calculated, B_calculated = cartpend_ss(
                m=m, M=M, L=L, g=g, d=d, s=s
            )
            A_expected, B_expected = expected_results[i]
            assert_allclose(A_calculated, A_expected)
            assert_allclose(B_calculated, B_expected)


if __name__ == "__main__":
    unittest.main()
