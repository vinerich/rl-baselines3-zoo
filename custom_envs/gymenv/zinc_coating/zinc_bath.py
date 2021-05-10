class ZincBath():
    """A class to represent a zinc bath.
    """

    def __init__(self):
        """Create a new zinc bath.
        """
        self.zinc_coating = 140

    def getZincCoatingForCoil(self, coil_characteristic, current_speed):
        speed_characteristic = current_speed * 0.125 + 0.875
        return self.zinc_coating * coil_characteristic / speed_characteristic
