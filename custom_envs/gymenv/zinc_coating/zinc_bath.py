import numpy as np


class ZincBath():
    """A class to represent a zinc bath.
    """

    def __init__(self, random_noise=False):
        """Create a new zinc bath.
        """
        self.min = 130
        self.max = 210
        self.zinc_coating = 170
        if random_noise:
            self.zinc_coating = np.random.rand() * 80 + 130  # [130, 210)
        self.random_noise = random_noise

    def step(self):
        if self.random_noise:
            self.zinc_coating += np.random.rand() * 4 - 2  # [-2, 2)
            self.zinc_coating = np.clip(self.zinc_coating, self.min, self.max)  # [130, 210]

    # characteristic: [0.6, 1.0]
    # speed: [1,333, 3,333]
    def getZincCoatingForCoil(self, coil_characteristic, current_speed):
        speed_characteristic = current_speed * 0.1 + 0.87  # [1.0, 1.2]
        return self.zinc_coating * coil_characteristic / speed_characteristic  # [65, 210]
