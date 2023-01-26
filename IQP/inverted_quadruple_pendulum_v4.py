import numpy as np
import sys

from gymnasium import utils
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.spaces import Box


DEFAULT_CAMERA_CONFIG = {
    "trackbodyid": 0,
    "distance": 4.1225,
    "lookat": np.array((0.0, 0.0, 0.12250000000000005)),
}

class InvertedQuadruplePendulumEnv(MujocoEnv, utils.EzPickle):
    """
    ## Description

    This environment originates from control theory and builds on the cartpole
    environment based on the work done by Barto, Sutton, and Anderson in
    ["Neuronlike adaptive elements that can solve difficult learning control problems"](https://ieeexplore.ieee.org/document/6313077),
    powered by the Mujoco physics simulator - allowing for more complex experiments
    (such as varying the effects of gravity or constraints). This environment involves a cart that can
    moved linearly, with a pole fixed on it and a second pole fixed on the other end of the first one
    (leaving the second pole as the only one with one free end). The cart can be pushed left or right,
    and the goal is to balance the second pole on top of the first pole, which is in turn on top of the
    cart, by applying continuous forces on the cart.

    ## Action Space
    The agent take a 1-element vector for actions.
    The action space is a continuous `(action)` in `[-1, 1]`, where `action` represents the
    numerical force applied to the cart (with magnitude representing the amount of force and
    sign representing the direction)

    | Num | Action                    | Control Min | Control Max | Name (in corresponding XML file) | Joint | Unit      |
    |-----|---------------------------|-------------|-------------|----------------------------------|-------|-----------|
    | 0   | Force applied on the cart | -1          | 1           | slider                           | slide | Force (N) |

    ## Observation Space

    The state space consists of positional values of different body parts of the pendulum system,
    followed by the velocities of those individual parts (their derivatives) with all the
    positions ordered before all the velocities.

    The observation is a `ndarray` with shape `(11,)` where the elements correspond to the following:

    | Num | Observation                                                       | Min  | Max | Name (in corresponding XML file) | Joint | Unit                     |
    | --- | ----------------------------------------------------------------- | ---- | --- | -------------------------------- | ----- | ------------------------ |
    | 1   | position of the cart along the linear surface                     | -Inf | Inf | slider                           | slide | position (m)             |
    | 2   | sine of the angle between the cart and the first pole             | -Inf | Inf | sin(hinge)                       | hinge | unitless                 |
    | 3   | sine of the angle between the two poles                           | -Inf | Inf | sin(hinge2)                      | hinge | unitless                 |
    | 4   | sine of the angle between the third poles                         | -Inf | Inf | sin(hinge3)                      | hinge | unitless                 |
    | 5   | sine of the angle between the forth poles                         | -Inf | Inf | sin(hinge4)                      | hinge | unitless                 |
    | 6   | cosine of the angle between the cart and the first pole           | -Inf | Inf | cos(hinge)                       | hinge | unitless                 |
    | 7   | cosine of the angle between the two poles                         | -Inf | Inf | cos(hinge2)                      | hinge | unitless                 |
    | 8   | cosine of the angle between the third poles                       | -Inf | Inf | cos(hinge3)                      | hinge | unitless                 |
    | 9   | cosine of the angle between the forth poles                       | -Inf | Inf | cos(hinge4)                      | hinge | unitless                 |
    | 10  | velocity of the cart                                              | -Inf | Inf | slider                          | slide | velocity (m/s)           |
    | 11  | angular velocity of the angle between the cart and the first pole | -Inf | Inf | hinge                            | hinge | angular velocity (rad/s) |
    | 12  | angular velocity of the angle between the two poles               | -Inf | Inf | hinge2                           | hinge | angular velocity (rad/s) |
    | 13  | angular velocity of the angle between the third poles             | -Inf | Inf | hinge3                           | hinge | angular velocity (rad/s) |
    | 14  | angular velocity of the angle between the forth poles             | -Inf | Inf | hinge4                           | hinge | angular velocity (rad/s) |
    | 15  | constraint force - 1                                              | -Inf | Inf |                                  |       | Force (N)                |
    | 16  | constraint force - 2                                              | -Inf | Inf |                                  |       | Force (N)                |
    | 17  | constraint force - 3                                              | -Inf | Inf |                                  |       | Force (N)                |
    | 18  | constraint force - 4                                              | -Inf | Inf |                                  |       | Force (N)                |
    | 19  | constraint force - 5                                              | -Inf | Inf |                                  |       | Force (N)                |

    There is physical contact between the robots and their environment - and Mujoco
    attempts at getting realistic physics simulations for the possible physical contact
    dynamics by aiming for physical accuracy and computational efficiency.

    There is one constraint force for contacts for each degree of freedom (3).
    The approach and handling of constraints by Mujoco is unique to the simulator
    and is based on their research. Once can find more information in their
    [*documentation*](https://mujoco.readthedocs.io/en/latest/computation.html)
    or in their paper
    ["Analytically-invertible dynamics with contacts and constraints: Theory and implementation in MuJoCo"](https://homes.cs.washington.edu/~todorov/papers/TodorovICRA14.pdf).


    ## Rewards

    The reward consists of two parts:
    - *alive_bonus*: The goal is to make the second inverted pendulum stand upright
    (within a certain angle limit) as long as possible - as such a reward of +10 is awarded
     for each timestep that the second pole is upright.
    - *distance_penalty*: This reward is a measure of how far the *tip* of the second pendulum
    (the only free end) moves, and it is calculated as
    *0.01 * x<sup>2</sup> + (y - 2)<sup>2</sup>*, where *x* is the x-coordinate of the tip
    and *y* is the y-coordinate of the tip of the second pole.
    - *velocity_penalty*: A negative reward for penalising the agent if it moves too
    fast *0.001 *  v<sub>1</sub><sup>2</sup> + 0.005 * v<sub>2</sub> <sup>2</sup>*

    The total reward returned is ***reward*** *=* *alive_bonus - distance_penalty - velocity_penalty*

    ## Starting State
    All observations start in state
    (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0) with a uniform noise in the range
    of [-0.1, 0.1] added to the positional values (cart position and pole angles) and standard
    normal force with a standard deviation of 0.1 added to the velocity values for stochasticity.

    ## Episode End
    The episode ends when any of the following happens:

    1.Truncation:  The episode duration reaches 1000 timesteps.
    2.Termination: Any of the state space values is no longer finite.
    3.Termination: The y_coordinate of the tip of the second pole *is less than or equal* to 1. The maximum standing height of the system is 1.196 m when all the parts are perpendicularly vertical on top of each other).

    ## Arguments

    No additional arguments are currently supported.

    ```python
    import gymnasium as gym
    env = gym.make('InvertedDoublePendulum-v4')
    ```
    There is no v3 for InvertedPendulum, unlike the robot environments where a v3 and
    beyond take `gymnasium.make` kwargs such as `xml_file`, `ctrl_cost_weight`, `reset_noise_scale`, etc.

    ```python
    import gymnasium as gym
    env = gym.make('InvertedDoublePendulum-v2')
    ```

    ## Version History

    * v4: All MuJoCo environments now use the MuJoCo bindings in mujoco >= 2.1.3
    * v3: Support for `gymnasium.make` kwargs such as `xml_file`, `ctrl_cost_weight`, `reset_noise_scale`, etc. rgb rendering comes from tracking camera (so agent does not run away from screen)
    * v2: All continuous control environments now use mujoco-py >= 1.50
    * v1: max_time_steps raised to 1000 for robot based tasks (including inverted pendulum)
    * v0: Initial versions release (1.0.0)
    """

    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "depth_array",
        ],
        "render_fps": 20,
    }

    # ----------------------------------------------------------------------------------------- shape #
    # joint 2개 추가 -> sin, cos, angular_vel 추가
    def __init__(self, **kwargs):
        observation_space = Box(low=-np.inf, high=np.inf, shape=(19,), dtype=np.float64)
        MujocoEnv.__init__(
            self,
            frame_skip=5,
            observation_space=observation_space,
            default_camera_config=DEFAULT_CAMERA_CONFIG,
            **kwargs
        )
        utils.EzPickle.__init__(self, **kwargs)

    def step(self, action):
        self.do_simulation(action, self.frame_skip)
        ob = self._get_obs()

        x, _, y = self.data.site_xpos[0] # Tip의 위치

        """
        print("Observation : ", ob)
        print("Tip : ", self.data.site_xpos[0])
        print("Q Vel : ", self.data.qvel)
        print("Q Cos Pos : ", np.cos(self.data.qpos[1:])) # 각 펜듈럼의 cos 값
        """

        dist_penalty = 0.01 * x**2 + (y - 2) ** 2
        v1, v2, v3, v4 = self.data.qvel[1:5] # 봉의 각속도
        vel_penalty = 1e-3 * v1**2 + 5e-3 * v2**2 + 5e-3 * v3**2 + 5e-3 * v4**2
        alive_bonus = 10
        r = alive_bonus - dist_penalty - vel_penalty

        # Terminal condition
        terminated = False #bool(y <= 1) # True/False
        if self.render_mode == "human":
            self.render()
        return ob, r, terminated, False, {}

    def _get_obs(self):
        return np.concatenate(
            [
                self.data.qpos[:1],  # cart x pos
                np.sin(self.data.qpos[1:]),  # link angles
                np.cos(self.data.qpos[1:]),
                np.clip(self.data.qvel, -10, 10),
                np.clip(self.data.qfrc_constraint, -10, 10),
            ]
        ).ravel()

    def reset_model(self):
        self.set_state(
            # 1번 째 joint가 아래를 향하도록
            self.init_qpos + np.array([0, np.pi, 0, 0, 0]) +
            + self.np_random.uniform(low=-0.1, high=0.1, size=self.model.nq),
            self.init_qvel + self.np_random.standard_normal(self.model.nv) * 0.1,
        )
        return self._get_obs()
