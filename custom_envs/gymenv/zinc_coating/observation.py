class Observation:
    def __init__(self, timestep, coil_speed, current_coil_type, current_coating_target, next_coil_type, next_coating_target, coil_length, zinc_bath_coating, zinc_coating, nozzle_pressure):
        self.timestep = timestep
        self.coil_speed = coil_speed
        self.current_coil_type = current_coil_type
        self.current_coating_target = current_coating_target
        self.next_coil_type = next_coil_type
        self.next_coating_target = next_coating_target
        self.coil_length = coil_length
        self.zinc_bath_coating = zinc_bath_coating
        self.zinc_coating = zinc_coating
        self.nozzle_pressure = nozzle_pressure
