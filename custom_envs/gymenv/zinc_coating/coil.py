import math


class Coil():
    """A class to represent a coil.
    """

    def __init__(self, type=0, length=100, rand_target=False, rand_characteristic=False):
        """Create a new coil with given type and length.

        Args:
            type (int, optional): Coil type. Varies the coating target and characteristics. Range 0-29. Defaults to 0.
            length (int, optional): Coil length. Defaults to 100.
        """
        if(type < 30 and type >= 0):
            self.type = type
        else:
            raise ValueError(
                "Coil type should be in the range of 0 to 29. Got: " + type)
        self.max_length = length
        self.length = length
        self.started = False
        self.rand_target = rand_target
        self.rand_characteristic = rand_characteristic

    def start(self, timestep, speed):
        """Starts the unrolling of this coil.

        Args:
            timestep (int): Current time in ms.
            speed (float): Current speed in m/s.
        """
        self.current_time = timestep
        self.current_speed = speed
        self.started = True

    def getLength(self, timestep, speed):
        """Gets the current length of the coil.

        Args:
            timestep (int): Current time in ms.
            speed (float): Current speed in m/s.

        Returns:
            float: Current length in m.
        """

        if(not self.started):
            return self.max_length

        time_passed = timestep - self.current_time
        time_passed_in_seconds = time_passed / 1000
        remaining_length = self.length - self.current_speed * time_passed_in_seconds

        remaining_length_next = remaining_length - self.current_speed * time_passed_in_seconds

        self.current_speed = speed
        self.current_time = timestep
        self.length = remaining_length
        return remaining_length, True if remaining_length_next < 0 else False

    def getZincCoatingTarget(self):
        """Gets the zinc coating target of this coil type.

        Returns:
            int: Coating target. Varies between 40 and 60.
        """
        if(self.rand_target):
            return math.floor(self.type / 3) * 2 + 40.0  # [40, 58]
        else:
            return 40.0

    def getZincCoatingCharacteristic(self):
        """Gets the zinc coating characteristic of this coil type. Coil types vary in there zinc bond.

        Returns:
            float: Coating characteristic. How much zinc bonds with the coil. Varies between 0.6 and 1.0.
        """
        if(self.rand_characteristic):
            return 1.0 - self.type % 3 * 0.2  # [0.6, 1.0]
        else:
            return 1.0
