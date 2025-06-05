import jsbsim
import ipdb
import polars as pl
import numpy as np
import matplotlib.pyplot as plt

jsbsim.FGJSBBase().debug_lvl=0

DT = 0.01
RAD2DEG = 180.0 / np.pi

class FlightDynamics:
    def __init__(self, aircraft_model):
        self.aircraft = jsbsim.FGFDMExec(None)
        self.aircraft.load_model(aircraft_model)
        self.aircraft.set_dt(DT)  # Set the simulation time step

    def initialize(self, initial_condition=None, randomization_factor=0.0):
        # self.set_state(initial_condition)
        # Set the initial conditions
        self.aircraft['ic/h-sl-ft'] = 10000        # Altitude (ft)
        self.aircraft['ic/u-fps'] = 420            # Calibrated airspeed (knots)
        self.aircraft['ic/psi-true-deg'] = 0       # Heading (degrees)
        self.aircraft['ic/phi-deg'] = 0            # Roll angle (level)
        self.aircraft['ic/theta-deg'] = 0          # Pitch angle (level)
        self.aircraft['ic/alpha-deg'] = 0          # Angle of attack
        self.aircraft['ic/beta-deg'] = 0           # Sideslip angle
        self.aircraft['gear/gear-cmd-norm'] = 0.0  # Gear position (0.0 for retracted)

        # Set engine-related conditions
        self.aircraft['propulsion/engine/set-running'] = 1.0
        # self.aircraft['propulsion/set-running'] = 1.0

        if self.aircraft.run_ic():
            print("Initial conditions loaded successfully.")

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

    def get_state(self, exclude=None):
        state_vec = {
            'time': self.aircraft['simulation/sim-time-sec'],
            'altitude': self.aircraft['atmosphere/density-altitude'],
            'x': self.aircraft['position/distance-from-start-lat-mt']*3.2804,
            'y': self.aircraft['position/distance-from-start-lon-mt']*3.2804,
            'z': self.aircraft['position/h-agl-ft'],

            'u': self.aircraft['velocities/u-fps'],
            'v': self.aircraft['velocities/v-fps'],
            'w': self.aircraft['velocities/w-fps'],

            'phi': self.aircraft['attitude/phi-deg'],
            'theta': self.aircraft['attitude/theta-deg'],
            'psi': self.aircraft['attitude/psi-deg'],

            'p': self.aircraft['velocities/p-rad_sec']*RAD2DEG,
            'q': self.aircraft['velocities/q-rad_sec']*RAD2DEG,
            'r': self.aircraft['velocities/r-rad_sec']*RAD2DEG,

            'alpha': self.aircraft['aero/alpha-rad']*RAD2DEG,
            'alphadot': self.aircraft['aero/alphadot-rad_sec']*RAD2DEG,
            'beta': self.aircraft['aero/beta-rad']*RAD2DEG,
            'betadot': self.aircraft['aero/betadot-rad_sec']*RAD2DEG,
            'gamma': self.aircraft['flight-path/gamma-rad']* RAD2DEG,

            'gust_north': self.aircraft['atmosphere/gust-north-fps'],
            'gust_east': self.aircraft['atmosphere/gust-east-fps'],
            'gust_down': self.aircraft['atmosphere/gust-down-fps'],
        }
        if exclude is not None:
            for key in exclude:
                state_vec.pop(key, None)
        return state_vec

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
        self.aircraft['fcs/aileron-cmd-norm'] = float(action[0])
        self.aircraft['fcs/elevator-cmd-norm'] = float(action[1])
        self.aircraft['fcs/rudder-cmd-norm'] = float(action[2])
        self.aircraft['fcs/throttle-cmd-norm'] = float(action[3])

    def get_input(self):
        """Get the current control inputs from the aircraft's flight control system."""
        return {
            'aileron': self.aircraft['fcs/aileron-cmd-norm'],
            'elevator': self.aircraft['fcs/elevator-cmd-norm'],
            'rudder': self.aircraft['fcs/rudder-cmd-norm'],
            'throttle': self.aircraft['fcs/throttle-cmd-norm']
        }

    def plot_trajectory(self, states, actions):
        """Plot the state trajectory using polars DataFrame."""        
        # Convert polars DataFrame to pandas for plotting
        figure, axes = plt.subplots(3, 2, figsize=(12, 10))        
        axes[0,0].plot(states['time'], states['x'], label='x position (ft)')
        axes[0,0].plot(states['time'], states['y'], label='y position (ft)')
        axes[0,0].plot(states['time'], states['altitude'], label='altitude (ft)')

        # axes[0,0].plot(states['time'], states['z'], label='z position (ft)')
        axes[0,0].set_xlabel('time (s)')
        axes[0,0].set_ylabel('distance (ft)')
        axes[0,0].set_title('position')
        axes[0,0].legend()
        axes[0,0].grid()

        axes[0,1].plot(states['time'], states['u'], label='u velocity (fps)')
        axes[0,1].plot(states['time'], states['v'], label='v velocity (fps)')
        axes[0,1].plot(states['time'], -states['w'], label='w velocity (fps)')
        axes[0,1].set_xlabel('time (s)')
        axes[0,1].set_ylabel('velocity components')
        axes[0,1].set_title('aircraft velocities')
        axes[0,1].legend()
        axes[0,1].grid()

        ax_right1 = axes[1,0].twinx()
        axes[1,0].plot(states['time'], states['phi'], label='phi (deg)')
        axes[1,0].plot(states['time'], states['theta'], label='theta (deg)')
        ax_right1.plot(states['time'], states['gamma'], label='gamma (deg)', linestyle='--', color='black')
        ax_right1.set_ylabel('flight path angle (deg)')
        ax_right1.legend(loc='upper right')
        # plot a horizontal line at 180 degrees for level flight
        ax_right1.axhline(y=180, color='gray', linewidth=2)
        # plot a horizontal line at -180 degrees for level flight
        ax_right1.axhline(y=-180, color='gray', linewidth=2)
        ax_right1.set_yticks(np.arange(-180, 181, 60))
        axes[1,0].set_xlabel('time (s)')
        axes[1,0].set_ylabel('attitude angles (deg)')
        axes[1,0].set_title('attitude')
        axes[1,0].legend(loc='upper left')
        axes[1,0].grid()

        axes[1,1].plot(states['time'], states['p'], label='p (deg/s)')
        axes[1,1].plot(states['time'], states['q'], label='q (deg/s)')
        axes[1,1].plot(states['time'], states['r'], label='r (deg/s)')
        axes[1,1].set_xlabel('time (s)')
        axes[1,1].set_ylabel('angular rates (deg/s)')
        axes[1,1].set_title('angular rates')
        axes[1,1].legend()
        axes[1,1].set_ylim([-30, 30])  # Adjust y-limits for better visibility
        axes[1,1].grid()
        
        ax_right2 = axes[2,0].twinx()
        ax_right2.plot(states['time'], states['psi'], label='psi (deg) (heading)', linestyle='--', color='black')
        # plot a horizontal line at 180 degrees for level flight
        ax_right2.axhline(y=180, color='gray', linewidth=2)
        # plot a horizontal line at -180 degrees for level flight
        ax_right2.axhline(y=-180, color='gray', linewidth=2)
        ax_right2.set_yticks(np.arange(-180, 181, 60))
        ax_right2.set_ylabel('heading (deg)')
        ax_right2.legend(loc='upper right')
        axes[2,0].plot(states['time'], states['alpha'], label='alpha (deg)')
        axes[2,0].plot(states['time'], states['beta'], label='beta (deg)')
        axes[2,0].set_xlabel('time (s)')
        axes[2,0].set_ylabel('angle (deg)')
        axes[2,0].legend(loc='upper left')
        axes[2,0].grid()

        axes[2,1].plot(states['time'], actions['throttle'], label='throttle input')
        axes[2,1].plot(states['time'], actions['elevator'], label='elevator input')
        axes[2,1].plot(states['time'], actions['aileron'], label='aileron input')
        axes[2,1].plot(states['time'], actions['rudder'], label='rudder input')
        axes[2,1].set_xlabel('time (s)')
        axes[2,1].set_ylabel('normalized control inputs')
        axes[2,1].set_title('control input history')
        axes[2,1].legend()
        axes[2,1].grid()

        figure.tight_layout()
        figure.savefig('../plots/flight_dynamics_trajectory_sample.png')

    def plot_path(self, states, interactive=False):
        # plot the path of the aircraft in 3D space
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        ax.plot(states['x'], states['y'], states['altitude'], label='Flight Path')
        ax.set_xlabel('X Position (m)')
        ax.set_ylabel('Y Position (m)')
        ax.set_zlabel('Altitude (ft)')
        ax.set_title('3D Flight Path')
        ax.set_box_aspect([1, 1, 0.5])  # Aspect ratio for better visualization
        max_vals = np.array([states['x'].max(), states['y'].max(), states['altitude'].max()]).max()
        min_vals = np.array([states['x'].min(), states['y'].min(), states['altitude'].min()]).min()
        ax.set_xlim([min_vals, max_vals])
        ax.set_ylim([min_vals, max_vals])
        ax.set_zlim([min_vals, max_vals])

        if interactive:
            plt.show()
        else:
            fig.savefig('../plots/flight_path_sample.png')

if __name__== "__main__":
    """Run a simple open loop test of the FlightDynamics class"""
    fd = FlightDynamics("f16")
    fd.initialize()

    state_trajectory = [] 
    action_trajectory = []
    sim_time = 100 # total simulation time in seconds
    print(f"initial fuel: {fd.aircraft['propulsion/total-fuel-lbs']}")

    for _ in np.arange(0, sim_time, DT):
        state_trajectory.append(fd.get_state())

        fd.set_input([0, -0.5, 0.0, 0.5])
        action_trajectory.append(fd.get_input())

        fd.propagate_dynamics()

        if fd.aircraft['atmosphere/density-altitude'] < 100 or fd.aircraft['atmosphere/density-altitude'] > 100000:
            print("Aircraft has crashed!")
            break

    print(f"final fuel: {fd.aircraft['propulsion/total-fuel-lbs']}")

    for name in fd.aircraft.get_property_catalog():
        if "gear" in name:
            print(name)

    state_trajectory = pl.DataFrame(state_trajectory)
    action_trajectory = pl.DataFrame(action_trajectory)

    fd.plot_trajectory(state_trajectory, action_trajectory)
    fd.plot_path(state_trajectory, interactive=True)