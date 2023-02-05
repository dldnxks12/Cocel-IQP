import sys
import numpy as np
from gymnasium import utils
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.spaces import Box

DEFAULT_CAMERA_CONFIG = {
    "trackbodyid": 0,
    "distance": 4.1225,
    "lookat": np.array((0.0, 0.0, 0.12250000000000005)),
}

class InvertedSinglePendulumEnv(MujocoEnv, utils.EzPickle):

    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "depth_array",
        ],
        "render_fps": 20,
    }

    def __init__(self, **kwargs):
        observation_space = Box(low=-np.inf, high=np.inf, shape=(7,), dtype=np.float64)
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
        """
        ob : 
        
        [-0.08409434  0.00870807 - 0.99996208 - 0.46542945 - 0.76157032  0.
         0.]
         """
        x, _, y = self.data.site_xpos[0]
        dist_penalty = 0.01 * x**2 + (y - 2) ** 2
        # print(self.data.qvel) # [0.136002   0.20637791]
        v1 = self.data.qvel[1]
        vel_penalty = 1e-3 * (v1 ** 2)
        alive_bonus = 10
        r = alive_bonus - dist_penalty - vel_penalty

        weight = (1+y) # give some wight according to current y position with respect to 1

        r = r*weight
        #terminated = (y <= -0.6)
        terminated = False
        if self.data.qpos[0] <= -0.99 or self.data.qpos[0] >= 0.99:
            terminated = True

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
            self.init_qpos + np.array([0, np.pi]) +
            + self.np_random.uniform(low=-0.1, high=0.1, size=self.model.nq),
            self.init_qvel + self.np_random.standard_normal(self.model.nv) * 0.1,
        )
        return self._get_obs()
