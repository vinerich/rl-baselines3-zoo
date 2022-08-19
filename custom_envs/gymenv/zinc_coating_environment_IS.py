import gym
from gym import spaces

import numpy as np

from .zinc_coating_environment import ZincCoatingV0


class ZincCoatingV0_IS(ZincCoatingV0):
    """Simple continous zinc coating environment"""

    def __init__(self, steps_per_episode=5000, coating_reward_time_offset=0, random_coating_targets=True, random_coil_characteristics=True, random_coil_lengths=True, random_coil_speed=True):
        super(ZincCoatingV0_IS, self).__init__(steps_per_episode, coating_reward_time_offset, random_coating_targets, random_coil_characteristics, random_coil_lengths, random_coil_speed)

        if coating_reward_time_offset == 0:
            raise Exception("Not usable with offset 0")

        self.coating_reward_time_offset = coating_reward_time_offset
        self.observation_dim = 14 + coating_reward_time_offset
        self.action_buffer = []
        for _ in range (coating_reward_time_offset):
            self.action_buffer.append(0)

        self.observation_space = spaces.Box(
            low=0, high=1, shape=(self.observation_dim,), dtype=np.float32)

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
        self.action_buffer.pop(0)
        self.action_buffer.append(nozzle_pressure[0] / 700)

        return self._transform_observation(observation) + tuple(self.action_buffer), reward, self.done, {"coating": zinc_coating_real}

    def reset(self):
        self.current_step = 0
        observation, _, _ = self.base.reset()

        self.action_buffer = []
        for _ in range (self.coating_reward_time_offset):
            self.action_buffer.append(0)

        return self._transform_observation(observation) + tuple(self.action_buffer)

    def render(self, mode='human', close=False):
        print("hey")

    def _transform_observation(self, observation):
        coating_delta = observation.zinc_coating - observation.current_coating_target
        return ((observation.coil_speed - 1.3) / 2,
                (observation.current_coating_target - 8) / 202,
                (observation.zinc_coating - 8) / 202,
                (observation.nozzle_pressure / 700),
                (coating_delta + 50) / 220,
                (1 if coating_delta < 0 else 0),
                (1 if coating_delta >= 0 and coating_delta <= 20 else 0),
                (1 if coating_delta > 20 else 0)) + one_hot_encode(observation.next_coil_type if observation.coil_switch_next_tick else observation.current_coil_type, 6)


def one_hot_encode(to_encode, discrete_states):
    output = [0] * discrete_states
    output[to_encode] = 1
    return tuple(output)
