import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

# individual initial conditions for the F16 aircraft - these are the starting values for the simulation
attitude_ic = {
    "ic/psi-true-deg": 180,  # Heading (degrees)
    "ic/phi-deg": 0,  # Roll angle (level)
    "ic/theta-deg": 0,  # Pitch angle (level)
}

attitude_rate_ic = {
    "ic/p-rad_s": 0.0,  # Roll rate (rad/s)
    "ic/q-rad_s": 0.0,  # Pitch rate (rad/s)
    "ic/r-rad_s": 0.0,  # Yaw rate (rad/s)
}

rate_ic = {
    "ic/u-fps": 200.0,  # Forward velocity (m/s)
    "ic/v-fps": 0.0,  # Sideways velocity (m/s)
    "ic/w-fps": 0.0,  # Vertical velocity (m/s)
}

position_ic = {
    "ic/h-agl-ft": 10000.0,  # Z position (ft)
}

aero_ic = {
    "ic/alpha-deg": 0.0,  # True angle of attack (degrees)
    "ic/beta-deg": 0.0,  # True sideslip angle (degrees)
}

# these are other initial conditions, typically needed to start the simulation
other_ic = {
    "gear/gear-cmd-norm": 0.0,  # Gear position (0.0 for retracted)
    "propulsion/engine[0]/set-running": 1.0,  # start with the engine running (needed for thrust command)
    "propulsion/engine[1]/set-running": 1.0,  # start with the engine running (needed for thrust command)
    "fcs/throttle-cmd-norm": 0.0,  # Thrust command (0.0 for idle)
}

# master initial conditions dictionary combining all types
ic = {
    "attitude": attitude_ic,
    "attitude_rate": attitude_rate_ic,
    "translation_rate": rate_ic,
    "position": position_ic,
    # 'aero': aero_ic,
    "other": other_ic,
}

# amount of randomization for each type
type_randomization_variance = {
    "attitude": 15,
    "attitude_rate": 5,
    "translation_rate": 20,
    "position": 1000,
    "aero": 10,
}

if __name__ == "__main__":
    print("F16 IC Configuration Loaded")
    print("Initial Conditions:", ic)
    print("Type Randomization Variance:", type_randomization_variance)

    figure, ax = plt.subplots()
    for key, value in sorted(type_randomization_variance.items(), key=lambda x: x[1]):
        if key != "position":
            # plotting the normal distribution of each type with 0 mean
            x = np.linspace(-3 * value, 3 * value, 100)
            y = norm.pdf(x, 0, value)
            ax.plot(x, y, label=f"{key} (σ={value})", linewidth=2)

    ax.set_xlabel("Value")
    ax.set_ylabel("Probability Density")
    ax.set_title("Normal Distributions for Type Randomization Variance")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.show()
