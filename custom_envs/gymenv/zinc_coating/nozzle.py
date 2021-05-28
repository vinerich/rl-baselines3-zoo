import numpy as np


class Nozzle():
    """A class to represent a nozzle used to blow excess zinc of a coil.
    """

    def __init__(self, pressure=350):
        self.max_pressure = 700
        self.min_pressure = 0
        self.pressure_to_zinc_scrub = 0.00125

        self.setPressure(pressure)

    def setPressure(self, pressure):
        self.pressure = np.clip(pressure, self.min_pressure, self.max_pressure)  # [0, 700]

    def getPressure(self):
        return self.pressure

    # [65, 150]
    # speed: [1,333, 3,333]
    def getZincCoating(self, zinc_coating, current_speed):
        speed_characteristic = current_speed * 0.1 + 0.87  # [1.0, 1.2]
        return (1 - self.pressure_to_zinc_scrub * self.pressure) * zinc_coating * speed_characteristic  # [8, 180]
