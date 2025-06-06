import logging
import os
import sys
from datetime import datetime

import jsbsim
import numpy as np
import polars as pl

# Add parent directory to path so we can import from sibling directories
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.plotting import plot_path, plot_trajectory
from config.f16_ic_config import ic, type_randomization_variance


jsbsim.FGJSBBase().debug_lvl = 0


DT = 0.1
RAD2DEG = 180.0 / np.pi
EARTH_RADIUS = 20925646.3  # Earth radius in feet (mean radius)

class FDM:
    def __init__(self, aircraft_model):
        self.aircraft = jsbsim.FGFDMExec(None)
        self.aircraft.load_model(aircraft_model)
        self.aircraft.set_dt(DT)  # Set the simulation time step

    def configure_turbulence(self, turbulence_strength=15.0, wind_speed=30.0):
        self.aircraft["atmosphere/turb-type"] = 1
        self.aircraft["atmosphere/turb-seed"] = 42
        self.aircraft["atmosphere/turbulence/magnitude"] = turbulence_strength
        self.aircraft["atmosphere/tubulence/wind-speed"] = wind_speed
        self.aircraft["atmosphere/turbulence/lu"] = 1000.0  # Scale of turbulence
        self.aircraft["atmosphere/turbulence/lv"] = 1000.0  # Length scale of turbulence
        self.aircraft["atmosphere/turbulence/lw"] = 1000.0  # Vertical scale of turbulence
        self.aircraft["atmosphere/turbulence/sigma-u"] = 20.0
        self.aircraft["atmosphere/turbulence/sigma-v"] = 10.0
        self.aircraft["atmosphere/turbulence/sigma-w"] = 10.0

    def initialize(self, initial_condition, randomization_factor=2.0):
        """Load initial conditions from a predefined configuration."""

        if randomization_factor > 0:
            # Randomize initial conditions within a specified range
            for subtype in initial_condition.keys():
                if subtype in type_randomization_variance.keys():
                    for key in initial_condition[subtype].keys():
                        random_offset = randomization_factor * np.random.normal(0, type_randomization_variance[subtype])
                        initial_condition[subtype][key] += random_offset

        final_ic = {}
        for sub_ic in initial_condition.values():
            final_ic.update(sub_ic)

        for ic_name in final_ic.keys():
            self.aircraft[ic_name] = final_ic[ic_name]

        self.aircraft.run_ic()


    def propagate_dynamics(self):
        self.aircraft.run()

    def set_state(self, state):
        """Set the state of the aircraft using a dictionary of state variables.

        Args:
            state (dict): Dictionary containing state variables to set.
                Expected keys include 'time', 'altitude', 'x', 'y', 'z',
                'u', 'v', 'w', 'phi', 'theta', 'psi', 'p', 'q', 'r', and 'gamma'.
        """
        for key, value in state.items():
            if key in self.aircraft:
                self.aircraft[key] = value
            else:
                raise KeyError(f"State variable '{key}' not found in aircraft model.")

    def get_state_dict(self, exclude=None):
        def geodetic_to_local(lat_rad, lon_rad):
            x = EARTH_RADIUS * lon_rad * np.cos(np.radians(lat_rad / 2))
            y = EARTH_RADIUS * lat_rad
            return x, y

        x_local, y_local = geodetic_to_local(
            self.aircraft["position/lat-gc-rad"], self.aircraft["position/long-gc-rad"]
        )
        observed_states = {
            "altitude": self.aircraft["atmosphere/density-altitude"],
            "u": self.aircraft["velocities/u-fps"],
            "v": self.aircraft["velocities/v-fps"],
            "w": self.aircraft["velocities/w-fps"],
            "phi": self.aircraft["attitude/phi-deg"],
            "theta": self.aircraft["attitude/theta-deg"],
            "psi": self.aircraft["attitude/psi-deg"],
            "p": self.aircraft["velocities/p-rad_sec"] * RAD2DEG,
            "q": self.aircraft["velocities/q-rad_sec"] * RAD2DEG,
            "r": self.aircraft["velocities/r-rad_sec"] * RAD2DEG,
        }
        unobserved_states = {
            "time": self.aircraft["simulation/sim-time-sec"],
            "x-dist": self.aircraft["position/distance-from-start-lat-mt"] * 3.2804,
            "y-dist": self.aircraft["position/distance-from-start-lon-mt"] * 3.2804,
            "x": x_local,
            "y": y_local,
            "z": self.aircraft["position/h-agl-ft"],
            "alpha": self.aircraft["aero/alpha-rad"] * RAD2DEG,
            "beta": self.aircraft["aero/beta-rad"] * RAD2DEG,
            "alphadot": self.aircraft["aero/alphadot-rad_sec"] * RAD2DEG,
            "betadot": self.aircraft["aero/betadot-rad_sec"] * RAD2DEG,
            "gamma": self.aircraft["flight-path/gamma-rad"] * RAD2DEG,
        }
        full_state = {**observed_states, **unobserved_states}

        if exclude is not None:
            for key in exclude:
                full_state.pop(key, None)

        return observed_states, full_state
    
    def get_observation(self, exclude=None):
        """Get the current observation from the aircraft's state.

        Args:
            exclude (list, optional): List of keys to exclude from the observation.
                If provided, these keys will be removed from the observation dictionary.

        Returns:
            dict: Dictionary containing the current observation state.
        """
        observed_states, _ = self.get_state_dict(exclude=exclude)
        return np.array(list(observed_states.values()), dtype=np.float32)

    def set_input(self, action):
        """Set the control inputs for the aircraft based on the action vector.

        Args:
            action (list or array): Control inputs in the following order:
                [aileron, elevator, rudder, throttle]
                - aileron: Aileron control input (normalized)
                - elevator: Elevator control input (normalized)
                - rudder: Rudder control input (normalized)
                - throttle: Throttle control input (normalized)

        Note:
            All control inputs are expected to be normalized values that will be
            converted to float and assigned to the aircraft's flight control system.
        """
        self.aircraft["fcs/aileron-cmd-norm"] = float(action[0])
        self.aircraft["fcs/elevator-cmd-norm"] = float(action[1])
        self.aircraft["fcs/rudder-cmd-norm"] = float(action[2])
        # self.aircraft["fcs/throttle-cmd-norm"] = float(action[3])

    def get_input_dict(self):
        """Get the current control inputs from the aircraft's flight control system."""
        return {
            "aileron": self.aircraft["fcs/aileron-cmd-norm"],
            "elevator": self.aircraft["fcs/elevator-cmd-norm"],
            "rudder": self.aircraft["fcs/rudder-cmd-norm"],
            # "throttle": self.aircraft["fcs/throttle-cmd-norm"],
        }
    
    def get_input(self):
        """Get the current control inputs as a list."""
        return [
            self.aircraft["fcs/aileron-cmd-norm"],
            self.aircraft["fcs/elevator-cmd-norm"],
            self.aircraft["fcs/rudder-cmd-norm"],
            # self.aircraft["fcs/throttle-cmd-norm"],
        ].to_numpy(dtype=np.float32)


if __name__ == "__main__":
    """Run a simple open loop test of the FlightDynamics class"""
    # Configure logging
    log_filename = f"logs/flight_sim_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_filename),
        ],
    )
    logger = logging.getLogger(__name__)

    fd = FDM("f16")
    fd.initialize(ic, randomization_factor=0.0)  # Load initial conditions

    state_trajectory = []
    action_trajectory = []
    sim_time = 100  # total simulation time in seconds

    logger.info(f"Aircraft model: {fd.aircraft.get_model_name()}")
    logger.info(f"Simulation time: {sim_time} seconds")
    logger.info(f"Time step: {DT} seconds")
    logger.info(f"initial fuel: {fd.aircraft['propulsion/total-fuel-lbs']}")
    for _ in np.arange(0, sim_time, DT):
        _, full_state = fd.get_state_dict()
        state_trajectory.append(full_state)

        fd.set_input([0.0, 0.0, 0.0, 0.0])  # this will be provided by the RL agent eventually
        action_trajectory.append(fd.get_input_dict())

        # current = fd.get_state() + fd.get_input()  # this is just to trigger the computation of the state
        logger.info({**fd.get_input_dict(), **full_state})

        fd.propagate_dynamics()

        if fd.aircraft["atmosphere/density-altitude"] < 50 or fd.aircraft["atmosphere/density-altitude"] > 100000:
            print("Aircraft has crashed!")
            break

    logger.info(f"final fuel: {fd.aircraft['propulsion/total-fuel-lbs']}")

    state_trajectory = pl.DataFrame(state_trajectory)
    action_trajectory = pl.DataFrame(action_trajectory)

    plot_trajectory(state_trajectory, action_trajectory, rewards=None)
    plot_path(state_trajectory, interactive=False)
