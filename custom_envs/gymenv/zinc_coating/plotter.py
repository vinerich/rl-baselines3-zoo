import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


class Plotter():
    """A class used to plot the zinc coating environment.
    """

    def __init__(self):
        """Create a new plotter.
        """

        self.timesteps = []
        self.coil_speed = []
        self.coil_type = []
        self.coil_length = []
        self.zinc_bath_coating = []
        self.zinc_coating = []
        self.zinc_coating_target = []
        self.zinc_coating_diff = []
        self.nozzle_pressure = []
        self.reward = []

        self.figure, self.subplots = plt.subplots(2, 1)
        self.coil_bath_line,  = self.subplots[0].plot([], [])

    def addValues(self, timestep, coil_speed, coil_type, coil_length, zinc_bath_coating, zinc_coating, zinc_coating_target, zinc_coating_diff, nozzle_pressure, reward):
        self.timesteps.append(timestep)
        self.coil_speed.append(coil_speed)
        self.coil_type.append(coil_type)
        self.coil_length.append(coil_length)
        self.zinc_bath_coating.append(zinc_bath_coating)
        self.zinc_coating.append(zinc_coating)
        self.zinc_coating_target.append(zinc_coating_target)
        self.zinc_coating_diff.append(zinc_coating_diff)
        self.nozzle_pressure.append(nozzle_pressure)
        self.reward.append(reward)

    def updateFigure(self):
        self.subplots[0].plot(self.timesteps, self.zinc_coating)

    def getZincCoatingForCoil(self, coil_characteristic, current_speed):
        speed_characteristic = current_speed * 0.125 + 0.875
        return self.zinc_coating * coil_characteristic / speed_characteristic

    def show(self):
        self.updateFigure()
        plt.show()
