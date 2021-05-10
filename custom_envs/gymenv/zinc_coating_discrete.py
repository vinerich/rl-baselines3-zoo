import gym
from gym import spaces
from enum import IntEnum
import numpy as np
import math
import queue
import plotly.graph_objects as graph_object
from plotly.subplots import make_subplots


class Actions(IntEnum):
    DECREASE_NOZZLE_PRESSURE = 0
    INCREASE_NOZZLE_PRESSURE = 1
    DO_NOTHING = 2


class Constants:
    NOZZLE_PRESSURE_MAX = 700
    NOZZLE_PRESSURE_MIN = 0
    NOZZLE_PRESSURE_INIT = 350
    NOZZLE_PRESSURE_STEP_SIZE = 6
    NOZZLE_PRESSURE_TO_ZINC_SCRUB = 0.0009

    ZINC_BATH_COATING_MAX = 100
    ZINC_BATH_COATING_MIN = 75
    ZINC_BATH_COATING_INIT = 85
    ZINC_BATH_JITTER_STEP_SIZE = 1

    ZINC_COATING_DISCRETE_INTERVAL = 20
    ZINC_COATING_DISCRETE_STEP_SIZE = 1
    ZINC_COATING_DISCRETE_STEPS = 2 + \
        math.floor(ZINC_COATING_DISCRETE_INTERVAL /
                   ZINC_COATING_DISCRETE_STEP_SIZE)

    ZINC_COATING_TARGET = 50


class Plotter:
    def __init__(self):
        self.nozzle = []
        self.zinc_coating = []
        self.target_coating = []
        self.zinc_bath_coating = []
        self.reward = []

    def update(self, nozzle, thickness, target_thickness, bath, reward):
        self.nozzle.append(nozzle)
        self.zinc_coating.append(thickness)
        self.target_coating.append(target_thickness)
        self.zinc_bath_coating.append(bath)
        if (len(self.reward) > 0):
            self.reward.append(self.reward[-1] + reward)
        else:
            self.reward.append(reward)

    def show(self):
        plot = make_subplots(rows=3)
        plot.add_trace(graph_object.Scatter(
            y=self.zinc_bath_coating, mode="lines", name="zinc_bath_coating"), row=1, col=1)
        plot.add_trace(graph_object.Scatter(
            y=self.zinc_coating, mode="lines", name="zinc_coating"), row=1, col=1)
        plot.add_trace(graph_object.Scatter(
            y=self.target_coating, mode="lines", name="target_coating"), row=1, col=1)
        plot.add_trace(graph_object.Scatter(
            y=self.nozzle, mode="lines", name="nozzle_pressure"), row=2, col=1)
        plot.add_trace(graph_object.Scatter(
            y=self.reward, mode="lines", name="cum. reward"), row=3, col=1)
        plot.show()


class ZincCoatingV0(gym.Env):
    """Simple zinc coating environment"""
    metadata = {'render.modes': ['human']}

    def __init__(self, reward_offset=100):
        super(ZincCoatingV0, self).__init__()

        self.action_space = spaces.Discrete(len(Actions))
        self.observation_space = spaces.Discrete(
            Constants.ZINC_COATING_DISCRETE_STEPS)

        self.reward_offset = reward_offset
        self.reward_queue = queue.SimpleQueue()
        for i in range(self.reward_offset):
            self.reward_queue.put(0)

        self.plotter = Plotter()

    def step(self, action):
        # Simulating episodes if Monte Carlo Methods are used
        if (self.current_step >= 1000):
            self.done = True
        else:
            self.done = False

        self.reward = self._get_reward()
        self.reward_queue.put(self.reward)
        self._take_action(action)
        return self._next_observation(), self.reward_queue.get(), self.done, {}

    def reset(self):
        self.nozzle_pressure = Constants.NOZZLE_PRESSURE_INIT + \
            (np.random.rand() - 0.5) * 50
        self.zinc_bath_coating = Constants.ZINC_BATH_COATING_INIT + \
            (np.random.rand() - 0.5) * 10
        self.target_zinc_coating = Constants.ZINC_COATING_TARGET

        self.zinc_coating = self._get_zinc_coating()
        self.current_step = 0
        self.reward = 0

        self.plotter = Plotter()

        return self._next_observation()

    def render(self, mode='human', close=False):
        self.plotter.update(self.nozzle_pressure,
                            self.zinc_coating, self.target_zinc_coating, self.zinc_bath_coating, self.reward)
        if (self.done):
            self.plotter.show()

    def _get_reward(self):
        if self.zinc_coating < self.target_zinc_coating:
            reward = -100
        elif self.zinc_coating >= self.target_zinc_coating + Constants.ZINC_COATING_DISCRETE_INTERVAL:
            reward = -1
        else:
            reward = 1/(self.zinc_coating - self.target_zinc_coating + 1)

        return reward

    def _take_action(self, action):
        if(action == int(Actions.INCREASE_NOZZLE_PRESSURE)):
            self.nozzle_pressure = np.min([
                self.nozzle_pressure + Constants.NOZZLE_PRESSURE_STEP_SIZE, Constants.NOZZLE_PRESSURE_MAX])

        elif(action == int(Actions.DECREASE_NOZZLE_PRESSURE)):
            self.nozzle_pressure = np.max([
                self.nozzle_pressure - Constants.NOZZLE_PRESSURE_STEP_SIZE, Constants.NOZZLE_PRESSURE_MIN])

    def _next_observation(self):
        if(self.current_step % Constants.ZINC_BATH_JITTER_STEP_SIZE == 0):
            self.zinc_bath_coating = np.min([np.max([self.zinc_bath_coating +
                                                     (np.random.rand() - 0.5) * 5, Constants.ZINC_BATH_COATING_MIN]), Constants.ZINC_BATH_COATING_MAX])
        self.zinc_coating = self._get_zinc_coating()
        self.current_step += 1

        return self._get_state()

    def _get_zinc_coating(self):
        zinc_scrub = self.nozzle_pressure * Constants.NOZZLE_PRESSURE_TO_ZINC_SCRUB
        zinc_thickness = (1 - zinc_scrub) * self.zinc_bath_coating
        return zinc_thickness

    def _get_state(self):
        if(self.zinc_coating < self.target_zinc_coating):
            return 0
        elif self.zinc_coating >= self.target_zinc_coating + Constants.ZINC_COATING_DISCRETE_INTERVAL:
            return Constants.ZINC_COATING_DISCRETE_STEPS - 1
        else:
            return math.floor(
                (self.zinc_coating - self.target_zinc_coating) / Constants.ZINC_COATING_DISCRETE_STEP_SIZE) + 1
