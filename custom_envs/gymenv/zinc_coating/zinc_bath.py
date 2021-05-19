import numpy as np


class ZincBath():
    """A class to represent a zinc bath.
    """

    def __init__(self):
        """Create a new zinc bath.
        """
        self.min = 120
        self.max = 160
        self.zinc_coating = 140

    def step(self):
        self.zinc_coating += np.random.rand() * 2 - 1  # [-1, 1)
        self.zinc_coating = np.clip(self.zinc_coating, self.min, self.max)  # [120, 160]

    def getZincCoatingForCoil(self, coil_characteristic, current_speed):
        speed_characteristic = current_speed * 0.125 + 0.875
        return self.zinc_coating * coil_characteristic / speed_characteristic
