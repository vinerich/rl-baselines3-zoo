import numpy as np


class ZincBath():
    """A class to represent a zinc bath.
    """

    def __init__(self):
        """Create a new zinc bath.
        """
        self.min = 130
        self.max = 170
        self.zinc_coating = np.random.rand() * 40 + 130  # [130, 170)

    def step(self):
        self.zinc_coating += np.random.rand() * 2 - 1  # [-1, 1)
        self.zinc_coating = np.clip(self.zinc_coating, self.min, self.max)  # [130, 170]

    # characteristic: [0.6, 1.0]
    # speed: [1,333, 3,333]
    def getZincCoatingForCoil(self, coil_characteristic, current_speed):
        speed_characteristic = current_speed * 0.1 + 0.87  # [1.0, 1.2]
        return self.zinc_coating * coil_characteristic / speed_characteristic  # [65, 170]
