import numpy as np
from rand_param_envs.base import RandomEnv
from rand_param_envs.gym import utils
import os

class Pr2EnvLego(RandomEnv, utils.EzPickle):

    FILE = os.path.join(os.dirname(__file__), 'assets/pr2.xml')

    def __init__(self,log_scale_limit=0.):

        RandomEnv.__init__(self, log_scale_limit, self.FILE, 4)
        utils.EzPickle.__init__(self)


    def _get_obs(self):
        dim = self.model.data.qpos.shape[0]

        idxpos = list(range(7))  # TODO: Hacky
        idxvel = list(range(7))
        return np.concatenate([
            self.model.data.qpos.flat[idxpos],
            self.model.data.qvel.flat[idxvel],  # Do not include the velocity of the target (should be 0).
            self.get_tip_position(),
            self.get_vec_tip_to_goal(),
        ]).reshape(-1)


    def get_tip_position(self):
        return self.model.data.site_xpos[0]

    def get_vec_tip_to_goal(self):
        tip_position = self.get_tip_position()
        goal_position = self.goal
        vec_tip_to_goal = goal_position - tip_position
        return vec_tip_to_goal

    def _step(self, action):

        self.forward_dynamics(action)

        vec_tip_to_goal = self.get_vec_tip_to_goal()
        distance_tip_to_goal = np.linalg.norm(vec_tip_to_goal)

        reward = - distance_tip_to_goal

        state = self._state
        notdone = np.isfinite(state).all()
        done = not notdone

        ob = self._get_obs()

        return ob, reward, done, {}

    def reset_model(self):
        qpos = self.init_qpos + self.np_random.uniform(low=-.005, high=.005,
                size=self.model.nq)
        qvel = self.init_qvel + self.np_random.uniform(low=-.005, high=.005,
                size=self.model.nv)
        self.goal = np.random.uniform(0, 1, 3)
        self.set_state(qpos, qvel)
