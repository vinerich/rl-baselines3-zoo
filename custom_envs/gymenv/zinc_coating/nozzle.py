import numpy as np


class Nozzle():
    """A class to represent a nozzle used to blow excess zinc of a coil.
    """

    def __init__(self, pressure=350):
        self.max_pressure = 700
        self.min_pressure = 0
        self.pressure_to_zinc_scrub = 0.001

        self.setPressure(pressure)

    def setPressure(self, pressure):
        self.pressure = np.max(
            [self.min_pressure, np.min([pressure, self.max_pressure])])

    def getPressure(self):
        return self.pressure

    def getZincCoating(self, zinc_coating, current_speed):
        speed_characteristic = current_speed * 0.15 + 0.7
        return (1 - self.pressure_to_zinc_scrub * self.pressure) * zinc_coating * speed_characteristic
