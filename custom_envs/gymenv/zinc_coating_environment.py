import gym
from gym import spaces

import numpy as np

from .zinc_coating.base import ZincCoatingBase as base


class ZincCoatingV0(gym.Env):
    """Simple continous zinc coating environment"""

    def __init__(self, steps_per_episode=5000, seed=420, coating_reward_time_offset=0, use_randomized_coils=True, use_randomized_coil_lengths=False, use_changing_coil_speed=False):
        super(ZincCoatingV0, self).__init__()

        self.steps_per_episode = steps_per_episode
        self.action_space = spaces.Box(
            np.array([0]), np.array([700]), dtype=np.float32)
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(70,), dtype=np.float32)
        self.base = base(seed, coating_reward_time_offset, use_randomized_coils,
                         use_randomized_coil_lengths, use_changing_coil_speed)

    def step(self, nozzle_pressure):
        self.current_step += 1
        if (self.current_step >= self.steps_per_episode):
            self.done = True
        else:
            self.done = False

        observation, reward = self.base.step(nozzle_pressure[0])
        transformed = self._transform_observation(observation)

        return self._transform_observation(observation), reward, self.done, {}

    def reset(self):
        self.current_step = 0
        observation, _ = self.base.reset()

        return self._transform_observation(observation)

    def render(self, mode='human', close=False):
        print("hey")

    def _transform_observation(self, observation):
        coating_delta = observation.zinc_coating - observation.current_coating_target
        return ((observation.coil_speed * 3 / 10),
                (observation.current_coating_target - 40) / 20,
                (observation.next_coating_target - 40)/20,
                (observation.coil_length / 100),
                (observation.zinc_coating - 20) / 110,
                (observation.nozzle_pressure / 700),
                (coating_delta + 100) / 200,
                (1 if coating_delta < 0 else 0),
                (1 if coating_delta >= 0 and coating_delta <= 20 else 0),
                (1 if coating_delta > 20 else 0)) + one_hot_encode(observation.current_coil_type, 30) + one_hot_encode(observation.next_coil_type, 30)


def one_hot_encode(to_encode, discrete_states):
    output = [0]*discrete_states
    output[to_encode] = 1
    return tuple(output)
