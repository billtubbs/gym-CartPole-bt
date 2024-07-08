"""
Modified cart-pole environment for use with the Gymnasium
project maintained by the Farama Foundation.

https://github.com/Farama-Foundation/Gymnasium


Modified cart-pole environment for use with OpenAI Gymnaisum.

This version of the classic cart-pole or cart-and-pendulum
control problem offers more variations on the basic OpenAI
Gym version (CartPole-v1).

It is based on a MATLAB implementation by Steven L. Brunton
as part of his Control Bootcamp series of videos on YouTube.

Features of this version include:
- More challenging control objectives (e.g. to stabilize
  the cart x-position as well as the pendulum angle)
- Continuously varying control actions
- Random disturbance to the state
- Measurement noise
- Reduced set of measured state variables

The goal of building this environment was to test different
control engineering and reinfircement learning methods on
a problem that is more challenging than the simple cart-pole
environment provided by OpenAI but still simple enough to
understand and use to help us learn about the relative
strengths and weaknesses of control/RL approaches.
"""

import math
from typing import Optional, Tuple, Union
from functools import partial
import numpy as np
from scipy.integrate import solve_ivp

import gymnasium as gym
from gymnasium import logger, spaces
from gymnasium.envs.classic_control import utils
from gymnasium.error import DependencyNotInstalled
from gymnasium.vector import VectorEnv
from gymnasium.vector.utils import batch_space

from gym_CartPole_BT.systems.cartpend import cartpend_dxdt


def angle_normalize(theta):
    return theta % (2*np.pi)


class CartPoleBTEnv(gym.Env[np.ndarray, Union[int, np.ndarray]]):
    """
    ## Description

    A pole is attached by an un-actuated joint to a cart, which moves along a 
    track. The goal is to move the cart and the pole to a goal position and 
    angle and stabilize it.

    ## Action Space

    The action is a `ndarray` with shape `(1,)` indicating the horizontal force
    on the cart in the x direction.

    | Num | Action                | Min                 | Max               |
    |-----|-----------------------|---------------------|-------------------|
    | 0   | Force on Cart         | -200                | 200               |

    **Note**: The velocity that is reduced or increased by the applied force is not fixed and it depends on the angle
     the pole is pointing. The center of gravity of the pole varies the amount of energy needed to move the cart underneath it

    ## Observation Space

    The observation is a `ndarray` with shape `(4,)` with the values corresponding to the following positions and velocities:

    | Num | Observation           | Min                 | Max               |
    |-----|-----------------------|---------------------|-------------------|
    | 0   | Cart Position         | -Inf                | Inf               |
    | 1   | Cart Velocity         | -Inf                | Inf               |
    | 2   | Pole Angle            | -Inf                | Inf               |
    | 3   | Pole Angular Velocity | -Inf                | Inf               |

    ## Rewards
    The reward is calculated each time step and is a negative cost.
    The cost function is the sum of the squared differences between
        (i) the cart x-position and the goal x-position
        (ii) the pole angle and the goal angle.

    ## Starting State
    Each episode, the system starts in a random state.

    ## Episode End
    The episode ends after 100 timesteps.

    ## Arguments

    Cartpole only has `render_mode` as a keyword for `gymnasium.make`.
    On reset, the `options` parameter allows the user to change the bounds used to determine the new random state.

    ```python
    >>> import gymnasium as gym
    >>> env = gym.make("CartPoleBT-v1", render_mode="rgb_array")
    >>> env
    <TimeLimit<OrderEnforcing<PassiveEnvChecker<CartPoleEnv<CartPole-v1>>>>>
    >>> env.reset(seed=123, options={"low": -0.1, "high": 0.1})  # default low=-0.05, high=0.05
    (array([ 0.03647037, -0.0892358 , -0.05592803, -0.06312564], dtype=float32), {})

    ```

    ## Vectorized environment

    To increase steps per seconds, users can use a custom vector environment or with an environment vectorizor.

    ```python
    >>> import gymnasium as gym
    >>> envs = gym.make_vec("CartPole-v1", num_envs=3, vectorization_mode="vector_entry_point")
    >>> envs
    CartPoleVectorEnv(CartPole-v1, num_envs=3)
    >>> envs = gym.make_vec("CartPole-v1", num_envs=3, vectorization_mode="sync")
    >>> envs
    SyncVectorEnv(CartPole-v1, num_envs=3)

    ```

    ## Version History
    * v1: TODO: Describe
    * v0: Initial release.
    """

    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 50,
    }

    def __init__(
        self, 
        goal_state=(0.0, 0.0, np.pi, 0.0), 
        disturbances=None, 
        initial_state='goal', 
        initial_state_variance=None, 
        measurement_error=None, 
        output_matrix=None, 
        n_steps=100, 
        render_mode=None
    ):  # TODO: Add type annotations

        self.goal_state = goal_state
        self.disturbances = disturbances
        self.initial_state = initial_state
        self.initial_state_variance = initial_state_variance
        self.measurement_error = measurement_error
        self.output_matrix = output_matrix
        self.n_steps = n_steps

        self.gravity = 10.0  # TODO: Note sign change
        self.masscart = 5.0
        self.masspole = 1.0
        self.length = 2.0
        self.friction = 1.0
        self.max_force = 200.0  # TODO: What is this for?

        # DELETE
        #self.total_mass = self.masspole + self.masscart
        #self.polemass_length = self.masspole * self.length
        #self.force_mag = 10.0

        # Set initial state and goal state
        self.goal_state = np.array(goal_state, dtype=np.float32)
        if isinstance(initial_state, str) and initial_state == 'goal':
            self.initial_state = self.goal_state.copy()
        else:
            self.initial_state = np.array(initial_state, dtype=np.float32)

        # Other features
        self.disturbances = disturbances
        self.initial_state_variance = initial_state_variance
        self.measurement_error = measurement_error
        if output_matrix is None:
            self.output_matrix = np.eye(4).astype(np.float32)
        else:
            self.output_matrix = np.array(output_matrix).astype(np.float32)
        self.variance_levels = {
            None: 0.0,
            'low': 0.01,
            'high': 0.2
        }

        # Details of simulation
        self.tau = 0.05  # seconds between state updates
        self.n_steps = n_steps
        self.time_step = 0
        self.kinematics_integrator = 'RK45'

        # Maximum and minimum thresholds for pole angle and cart position
        inf = np.finfo(np.float32).max
        self.theta_threshold_radians = inf
        self.x_threshold = inf

        # Episode terminates early if these limits are exceeded
        self.state_bounds = np.array([
            [-self.x_threshold, self.x_threshold],
            [-inf, inf],
            [-self.theta_threshold_radians, self.theta_threshold_radians],
            [-inf, inf]
        ])

        # Translate state constraints into output bounds
        output_bounds = self.output_matrix.dot(self.state_bounds)
        low = output_bounds[:, 0].astype(np.float32)
        high = output_bounds[:, 1].astype(np.float32)
        self.observation_space = spaces.Box(low, high, dtype=np.float32)
        self.action_space = spaces.Box(np.float32(-self.max_force),
                                       np.float32(self.max_force),
                                       shape=(1,), dtype=np.float32)

        self.render_mode = render_mode

        self.screen_width = 600
        self.screen_height = 400
        self.screen = None
        self.clock = None
        self.isopen = True
        self.state = None

        self.steps_beyond_terminated = None

    def cost_function(self, state, goal_state):
        """Evaluates the cost based on the current state y and
        the goal state.
        """

        return ((state[0] - self.goal_state[0])**2 +
                (angle_normalize(state[2]) - self.goal_state[2])**2)

    def output(self, state):
        return self.output_matrix.dot(state)

    def step(self, u):

        u = np.clip(u, -self.max_force, self.max_force)[0].astype('float32')
        x = self.state
        t = self.time_step * self.tau

        if self.kinematics_integrator == 'Euler':
            # Calculate time derivative
            x_dot = cartpend_dxdt(t, x,
                                  m=self.masspole,
                                  M=self.masscart,
                                  L=self.length,
                                  g=self.gravity,
                                  d=self.friction,
                                  u=u)

            # Simple state update (Euler method)
            self.state += self.tau * x_dot.astype('float32')
            output = self.output(self.state)

        else:
            # Create a partial function for use by solver
            f = partial(cartpend_dxdt,
                        m=self.masspole,
                        M=self.masscart,
                        L=self.length,
                        g=self.gravity,
                        d=self.friction,
                        u=u)

            # Integrate using numerical solver
            tf = t + self.tau
            sol = solve_ivp(f, t_span=[t, tf], y0=x,
                            method=self.kinematics_integrator, 
                            t_eval=[tf])
            self.state = sol.y.reshape(-1).astype('float32')
            output = self.output(self.state)

        # Add disturbance to pendulum angular velocity (theta_dot)
        if self.disturbances is not None:
            v = self.variance_levels[self.disturbances]
            self.state[3] += 0.05 * self.np_random.normal(scale=v)

        terminated = bool(
            self.state[0] < -self.x_threshold
            or self.state[0] > self.x_threshold
            or self.state[2] < -self.theta_threshold_radians
            or self.state[2] > self.theta_threshold_radians
        )

        reward = -self.cost_function(self.state, self.goal_state)

        if terminated:
            if self.steps_beyond_terminated is None:
                # Pole just fell!
                self.steps_beyond_terminated = 0
                reward -= 100  # TODO: Set this penalty amount
            else:
                if self.steps_beyond_terminated == 0:
                    logger.warn(
                        "You are calling 'step()' even though this "
                        "environment has already returned terminated = True. You "
                        "should always call 'reset()' once you receive 'terminated = "
                        "True' -- any further steps are undefined behavior."
                    )
                self.steps_beyond_terminated += 1

        if self.render_mode == "human":
            self.render()
        # truncation=False as the time limit is handled by the `TimeLimit` wrapper added during `make`
        return np.array(self.state, dtype=np.float32), reward, terminated, False, {}

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ):
        super().reset(seed=seed)
        # Note that if you use custom reset bounds, it may lead to out-of-bound
        # state/observations.
        low, high = utils.maybe_parse_reset_bounds(
            options, -0.05, 0.05  # default low
        )  # default high
        self.state = self.np_random.uniform(low=low, high=high, size=(4,))
        self.steps_beyond_terminated = None

        if self.render_mode == "human":
            self.render()
        return np.array(self.state, dtype=np.float32), {}

    def render(self):
        if self.render_mode is None:
            assert self.spec is not None
            gym.logger.warn(
                "You are calling render method without specifying any render mode. "
                "You can specify the render_mode at initialization, "
                f'e.g. gym.make("{self.spec.id}", render_mode="rgb_array")'
            )
            return

        try:
            import pygame
            from pygame import gfxdraw
        except ImportError as e:
            raise DependencyNotInstalled(
                'pygame is not installed, run `pip install "gymnasium[classic-control]"`'
            ) from e

        if self.screen is None:
            pygame.init()
            if self.render_mode == "human":
                pygame.display.init()
                self.screen = pygame.display.set_mode(
                    (self.screen_width, self.screen_height)
                )
            else:  # mode == "rgb_array"
                self.screen = pygame.Surface((self.screen_width, self.screen_height))
        if self.clock is None:
            self.clock = pygame.time.Clock()

        world_width = self.length * 2.4
        scale = self.screen_width / world_width
        carty = 160 # TOP OF CART
        polewidth = 10.0
        polelen = scale * (0.5 * self.length)
        cartwidth = 50.0
        cartheight = 30.0

        if self.state is None:
            return None

        x = self.state

        self.surf = pygame.Surface((self.screen_width, self.screen_height))
        self.surf.fill((255, 255, 255))

        l, r, t, b = (
            -cartwidth/2, 
            cartwidth/2, 
            cartheight/2,
            -cartheight/2
        )
        axleoffset = cartheight / 4.0
        cartx = x[0] * scale + self.screen_width / 2.0  # MIDDLE OF CART
        carty = 100  # TOP OF CART

        # Draw cart
        cart_coords = [(l, b), (l, t), (r, t), (r, b)]
        cart_coords = [(c[0] + cartx, c[1] + carty) for c in cart_coords]
        gfxdraw.aapolygon(self.surf, cart_coords, (0, 0, 0))
        gfxdraw.filled_polygon(self.surf, cart_coords, (0, 0, 0))

        l, r, t, b = (
            -polewidth / 2,
            polewidth / 2,
            polelen - polewidth / 2,
            -polewidth / 2,
        )

        # Draw pole
        pole_coords = []
        for coord in [(l, b), (l, t), (r, t), (r, b)]:
            # TODO: Fix orientation of pole
            coord = pygame.math.Vector2(coord).rotate_rad(-x[2])  # -x[2] or x[2] + np.pi
            coord = (coord[0] + cartx, coord[1] + carty + axleoffset)
            pole_coords.append(coord)
        gfxdraw.aapolygon(self.surf, pole_coords, (202, 152, 101))
        gfxdraw.filled_polygon(self.surf, pole_coords, (202, 152, 101))

        # Draw axle
        gfxdraw.aacircle(
            self.surf,
            int(cartx),
            int(carty + axleoffset),
            int(polewidth / 2),
            (129, 132, 203),
        )
        gfxdraw.filled_circle(
            self.surf,
            int(cartx),
            int(carty + axleoffset),
            int(polewidth / 2),
            (129, 132, 203),
        )

        # Draw track
        gfxdraw.hline(self.surf, 0, self.screen_width, carty, (0, 0, 0))

        self.surf = pygame.transform.flip(self.surf, False, True)
        self.screen.blit(self.surf, (0, 0))
        if self.render_mode == "human":
            pygame.event.pump()
            self.clock.tick(self.metadata["render_fps"])
            pygame.display.flip()

        # TODO: Draw goal line

        # TODO: Draw initial state position

        elif self.render_mode == "rgb_array":
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self.screen)), axes=(1, 0, 2)
            )

    def close(self):
        if self.screen is not None:
            import pygame

            pygame.display.quit()
            pygame.quit()
            self.isopen = False


class CartPoleVectorEnv(VectorEnv):
    metadata = {
        "render_modes": ["rgb_array"],
        "render_fps": 50,
    }

    def __init__(
        self,
        num_envs: int = 1,
        max_episode_steps: int = 500,
        render_mode: Optional[str] = None
    ):

        self.num_envs = num_envs
        self.max_episode_steps = max_episode_steps
        self.render_mode = render_mode

        self.gravity = 9.8
        self.masscart = 1.0
        self.masspole = 0.1
        self.total_mass = self.masspole + self.masscart
        self.length = 0.5  # actually half the pole's length
        self.polemass_length = self.masspole * self.length
        self.force_mag = 10.0
        self.tau = 0.02  # seconds between state updates
        self.kinematics_integrator = "euler"

        self.state = None

        self.steps = np.zeros(num_envs, dtype=np.int32)
        self.prev_done = np.zeros(num_envs, dtype=np.bool_)

        # Angle at which to fail the episode
        self.theta_threshold_radians = 12 * 2 * math.pi / 360
        self.x_threshold = 2.4

        # Angle limit set to 2 * theta_threshold_radians so failing observation
        # is still within bounds.
        high = np.array(
            [
                self.x_threshold * 2,
                np.inf,
                self.theta_threshold_radians * 2,
                np.inf,
            ],
            dtype=np.float32,
        )

        self.low = -0.05
        self.high = 0.05

        self.single_action_space = spaces.Discrete(2)
        self.action_space = batch_space(self.single_action_space, num_envs)
        self.single_observation_space = spaces.Box(-high, high, dtype=np.float32)
        self.observation_space = batch_space(self.single_observation_space, num_envs)

        self.screen_width = 600
        self.screen_height = 400
        self.screens = None
        self.surf = None

        self.steps_beyond_terminated = None

    def step(
        self, action: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
        assert self.action_space.contains(
            action
        ), f"{action!r} ({type(action)}) invalid"
        assert self.state is not None, "Call reset before using step method."

        x, x_dot, theta, theta_dot = self.state
        force = np.sign(action - 0.5) * self.force_mag
        costheta = np.cos(theta)
        sintheta = np.sin(theta)

        # For the interested reader:
        # https://coneural.org/florian/papers/05_cart_pole.pdf
        temp = (
            force + self.polemass_length * theta_dot**2 * sintheta
        ) / self.total_mass
        thetaacc = (self.gravity * sintheta - costheta * temp) / (
            self.length * (4.0 / 3.0 - self.masspole * costheta**2 / self.total_mass)
        )
        xacc = temp - self.polemass_length * thetaacc * costheta / self.total_mass

        if self.kinematics_integrator == "euler":
            x = x + self.tau * x_dot
            x_dot = x_dot + self.tau * xacc
            theta = theta + self.tau * theta_dot
            theta_dot = theta_dot + self.tau * thetaacc
        else:  # semi-implicit euler
            x_dot = x_dot + self.tau * xacc
            x = x + self.tau * x_dot
            theta_dot = theta_dot + self.tau * thetaacc
            theta = theta + self.tau * theta_dot

        self.state = np.stack((x, x_dot, theta, theta_dot))

        terminated: np.ndarray = (
            (x < -self.x_threshold)
            | (x > self.x_threshold)
            | (theta < -self.theta_threshold_radians)
            | (theta > self.theta_threshold_radians)
        )

        self.steps += 1

        truncated = self.steps >= self.max_episode_steps

        reward = np.ones_like(terminated, dtype=np.float32)

        # Reset all environments which terminated or were truncated in the last step
        # TODO: Why is this here?
        self.state[:, self.prev_done] = self.np_random.uniform(
            low=self.low, high=self.high, size=(4, self.prev_done.sum())
        )
        self.steps[self.prev_done] = 0
        reward[self.prev_done] = 0.0
        terminated[self.prev_done] = False
        truncated[self.prev_done] = False

        self.prev_done = terminated | truncated

        return self.state.T.astype(np.float32), reward, terminated, truncated, {}

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ):
        super().reset(seed=seed)
        self.state = self.initial_state.copy()
        assert self.state.shape[0] == 4

        # Add random variance to initial state
        v = self.variance_levels[self.initial_state_variance]
        self.state += self.np_random.normal(scale=v, size=(4, )).astype('float32')

        output = self.output(self.state)

        self.time_step = 0
        self.steps_beyond_terminated = None
        self.steps = np.zeros(self.num_envs, dtype=np.int32)
        self.prev_done = np.zeros(self.num_envs, dtype=np.bool_)

        return output, {}

    def render(self):
        if self.render_mode is None:
            assert self.spec is not None
            gym.logger.warn(
                "You are calling render method without specifying any render mode. "
                "You can specify the render_mode at initialization, "
                f'e.g. gym.make_vec("{self.spec.id}", render_mode="rgb_array")'
            )
            return

        try:
            import pygame
            from pygame import gfxdraw
        except ImportError:
            raise DependencyNotInstalled(
                'pygame is not installed, run `pip install "gymnasium[classic_control]"`'
            )

        if self.screens is None:
            pygame.init()

            self.screens = [
                pygame.Surface((self.screen_width, self.screen_height))
                for _ in range(self.num_envs)
            ]

        world_width = self.length * 2.4
        scale = self.screen_width / world_width
        carty = 160 # TOP OF CART
        polewidth = 10.0
        polelen = scale * (0.5 * self.length)
        cartwidth = 50.0
        cartheight = 30.0

        if self.state is None:
            raise ValueError(
                "Cartpole's state is None, it probably hasn't be reset yet."
            )

        for x, screen in zip(self.state.T, self.screens):
            assert isinstance(x, np.ndarray) and x.shape == (4,)

            self.surf = pygame.Surface((self.screen_width, self.screen_height))
            self.surf.fill((255, 255, 255))

            l, r, t, b = (
                -cartwidth/2, 
                cartwidth/2, 
                cartheight/2,
                -cartheight/2
            )
            axleoffset = cartheight / 4.0
            cartx = x[0] * scale + self.screen_width / 2.0  # MIDDLE OF CART
            carty = 100  # TOP OF CART

            # Draw cart
            cart_coords = [(l, b), (l, t), (r, t), (r, b)]
            cart_coords = [(c[0] + cartx, c[1] + carty) for c in cart_coords]
            gfxdraw.aapolygon(self.surf, cart_coords, (0, 0, 0))
            gfxdraw.filled_polygon(self.surf, cart_coords, (0, 0, 0))

            l, r, t, b = (
                -polewidth/2, 
                polewidth/2, 
                polelen - polewidth/2,
                -polewidth/2
            )

            # Draw pole
            pole_coords = []
            for coord in [(l, b), (l, t), (r, t), (r, b)]:
                coord = pygame.math.Vector2(coord).rotate_rad(-x[2])  # -x[2] or x[2] + np.pi
                coord = (coord[0] + cartx, coord[1] + carty + axleoffset)
                pole_coords.append(coord)
            gfxdraw.aapolygon(self.surf, pole_coords, (202, 152, 101))
            gfxdraw.filled_polygon(self.surf, pole_coords, (202, 152, 101))

            # Draw axle
            gfxdraw.aacircle(
                self.surf,
                int(cartx),
                int(carty + axleoffset),
                int(polewidth / 2),
                (129, 132, 203),
            )
            gfxdraw.filled_circle(
                self.surf,
                int(cartx),
                int(carty + axleoffset),
                int(polewidth / 2),
                (129, 132, 203),
            )

            # Draw track
            gfxdraw.hline(self.surf, 0, self.screen_width, carty, (0, 0, 0))

        # TODO: Draw goal line

        # TODO: Draw initial state position

            self.surf = pygame.transform.flip(self.surf, False, True)
            screen.blit(self.surf, (0, 0))

        return [
            np.transpose(np.array(pygame.surfarray.pixels3d(screen)), axes=(1, 0, 2))
            for screen in self.screens
        ]

    def close(self):
        if self.screens is not None:
            import pygame

            pygame.quit()