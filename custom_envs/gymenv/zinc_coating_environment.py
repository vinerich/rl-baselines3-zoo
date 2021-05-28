import gym
from gym import spaces

import numpy as np

from .zinc_coating.base import ZincCoatingBase as base


class ZincCoatingV0(gym.Env):
    """Simple continous zinc coating environment"""

    def __init__(self, steps_per_episode=5000, coating_reward_time_offset=0, random_coating_targets=False, random_coil_characteristics=False, random_coil_lengths=False, random_coil_speed=False, coating_dist_mean=0.0, coating_dist_std=0.0, coating_dist_reward=False):
        super(ZincCoatingV0, self).__init__()

        self.steps_per_episode = steps_per_episode
        self.action_space = spaces.Box(
            np.array([0]), np.array([700]), dtype=np.float32)
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(70,), dtype=np.float32)
        self.base = base(
            coating_reward_time_offset=coating_reward_time_offset,
            random_coating_targets=random_coating_targets,
            random_coil_characteristics=random_coil_characteristics,
            random_coil_lengths=random_coil_lengths,
            random_coil_speed=random_coil_speed,
            coating_dist_mean=coating_dist_mean,
            coating_dist_std=coating_dist_std,
            coating_dist_reward=coating_dist_reward,
        )
        self.seed()

    def seed(self, seed=None):
        if seed is None:
            seed = np.random.randint(0, 10000000)
        self.base.seed(seed)
        return [seed]

    def step(self, nozzle_pressure):
        self.current_step += 1
        if (self.current_step >= self.steps_per_episode):
            self.done = True
        else:
            self.done = False

        observation, reward, zinc_coating_real = self.base.step(nozzle_pressure[0])

        return self._transform_observation(observation), reward, self.done, {zinc_coating_real: zinc_coating_real}

    def reset(self):
        self.current_step = 0
        observation, _, _ = self.base.reset()

        return self._transform_observation(observation)

    def render(self, mode='human', close=False):
        print("hey")

    def _transform_observation(self, observation):
        coating_delta = observation.zinc_coating - observation.current_coating_target
        return ((observation.coil_speed - 1, 3) / 2 ,
                (observation.current_coating_target - 40) / 20,
                (observation.next_coating_target - 40) / 20,
                (observation.coil_length / 100),
                (observation.zinc_coating - 8) / 172,
                (observation.nozzle_pressure / 700),
                (coating_delta + 100) / 200,
                (1 if coating_delta < 0 else 0),
                (1 if coating_delta >= 0 and coating_delta <= 20 else 0),
                (1 if coating_delta > 20 else 0)) + one_hot_encode(observation.current_coil_type, 30) + one_hot_encode(observation.next_coil_type, 30)


def one_hot_encode(to_encode, discrete_states):
    output = [0] * discrete_states
    output[to_encode] = 1
    return tuple(output)
