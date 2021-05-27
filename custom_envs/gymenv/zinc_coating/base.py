import gym
from .coil import Coil
from .nozzle import Nozzle
from .zinc_bath import ZincBath
from .observation import Observation
import numpy as np
import math
import queue


class Constants:
    TIMESTEP = 100  # ms


class ZincCoatingBase():
    def __init__(self, coating_reward_time_offset=0, random_coating_targets=False, random_coil_characteristics=False, random_coil_lengths=False, random_coil_speed=False, coating_dist_mean=0.0, coating_dist_std=0.0, coating_dist_reward=False):
        self.use_randomized_coil_targets = random_coating_targets
        self.use_randomized_coil_characteristics = random_coil_characteristics
        self.use_changing_coil_speed = random_coil_speed
        self.use_randomized_coil_lengths = random_coil_lengths
        self.coating_reward_time_offset = np.max(
            [0, coating_reward_time_offset])
        self.coating_dist = np.random.normal(coating_dist_mean, coating_dist_std)
        self.coating_dist_mean = coating_dist_mean
        self.coating_dist_std = coating_dist_std
        self.coating_dist_reward = coating_dist_reward
        self.reward_queue = queue.Queue()

        self.nozzle = Nozzle()
        self.zinc_bath = ZincBath()

    def seed(self, seed=None):
        if seed is None:
            seed = np.random.randint(0, 10000000)
        np.random.seed(seed)

        print("Seed to use: ", seed)

    def reset(self):
        self.timestep = 0
        self.coil_speed = 160 / 60
        self.coil_types = [0] * 30

        if(self.use_changing_coil_speed):
            self.coil_speed_target = np.random.randint(80, 200) / 60
        else:
            self.coil_speed_target = self.coil_speed

        self.reward_queue = queue.Queue()
        for i in range(self.coating_reward_time_offset):
            self.reward_queue.put(0)

        self.nozzle = Nozzle()
        self.zinc_bath = ZincBath()
        self.zinc_bath.step()

        self.current_coil = self.getNewCoil()
        self.current_coil.start(self.timestep, self.coil_speed)
        self.next_coil = self.getNewCoil()

        zinc_bath_coating = self.zinc_bath.getZincCoatingForCoil(
            self.current_coil.getZincCoatingCharacteristic(), self.coil_speed)
        zinc_coating = self.nozzle.getZincCoating(
            zinc_bath_coating, self.coil_speed)
        zinc_coating_dist = self._apply_coating_dist(zinc_coating)

        return Observation(self.timestep, self.coil_speed, self.current_coil.type, self.current_coil.getZincCoatingTarget(), self.next_coil.type, self.next_coil.getZincCoatingTarget(), self.current_coil.max_length, zinc_bath_coating, zinc_coating_dist, self.nozzle.getPressure()), 0, zinc_coating

    def step(self, new_pressure):
        self.timestep += Constants.TIMESTEP
        self.nozzle.setPressure(new_pressure)
        self.zinc_bath.step()

        if(self.use_changing_coil_speed):
            if(self.coil_speed_target < self.coil_speed):
                self.coil_speed -= 0.2 / 60
            else:
                self.coil_speed += 0.2 / 60

            if(np.absolute(self.coil_speed - self.coil_speed_target) < 0.01):
                self.coil_speed_target = np.random.randint(80, 200) / 60

        coil_length = self.current_coil.getLength(
            self.timestep, self.coil_speed)
        if(coil_length < 0):
            self.current_coil = self.next_coil
            self.current_coil.start(self.timestep, self.coil_speed)
            self.next_coil = self.getNewCoil()

            coil_length = self.current_coil.max_length

        zinc_bath_coating = self.zinc_bath.getZincCoatingForCoil(
            self.current_coil.getZincCoatingCharacteristic(), self.coil_speed)
        zinc_coating = self.nozzle.getZincCoating(
            zinc_bath_coating, self.coil_speed)
        zinc_coating_dist = self._apply_coating_dist(zinc_coating)

        zinc_coating_diff = zinc_coating - self.current_coil.getZincCoatingTarget()
        zinc_coating_dist_diff = zinc_coating_dist - self.current_coil.getZincCoatingTarget()

        if self.coating_dist_reward:
            self.reward_queue.put(self._get_reward(zinc_coating_dist_diff))
        else:
            self.reward_queue.put(self._get_reward(zinc_coating_diff))

        return Observation(self.timestep, self.coil_speed, self.current_coil.type, self.current_coil.getZincCoatingTarget(), self.next_coil.type, self.next_coil.getZincCoatingTarget(), coil_length, zinc_bath_coating, zinc_coating_dist, self.nozzle.getPressure()), self.reward_queue.get(), zinc_coating

    def getNewCoil(self):
        coil_type = 0
        if(self.use_randomized_coil_targets or self.use_randomized_coil_characteristics):
            if(np.max(self.coil_types) == 0):
                coil_type = np.random.randint(30)
            else:
                probabilities = np.max(self.coil_types) - self.coil_types
                probabilities = probabilities / np.sum(probabilities)
                coil_type = np.random.choice(30, p=probabilities)

        coil_length = 100
        if(self.use_randomized_coil_lengths):
            coil_length = np.random.randint(50, 150)

        self.coil_types[coil_type] += 1
        return Coil(type=coil_type, length=coil_length, rand_target=self.use_randomized_coil_targets, rand_characteristic=self.use_randomized_coil_characteristics)

    def _get_reward(self, zinc_coating_diff):
        if zinc_coating_diff < 0:
            reward = -100
        elif zinc_coating_diff > 20:
            reward = -10
        else:
            reward = 1 / (zinc_coating_diff + 1)

        return reward

    def _apply_coating_dist(self, true_value):
        return true_value + np.random.default_rng().normal(self.coating_dist_mean, self.coating_dist_std)
