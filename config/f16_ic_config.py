import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

# individual initial conditions for the F16 aircraft - these are the starting values for the simulation
attitude_ic = {
    "ic/psi-true-deg": 180,  # Heading (degrees)
    "ic/phi-deg": 0,  # Roll angle (level)
    "ic/theta-deg": 10,  # Pitch angle (level)
}

attitude_rate_ic = {
    "ic/p-rad_s": 0.0,  # Roll rate (rad/s)
    "ic/q-rad_s": 0.0,  # Pitch rate (rad/s)
    "ic/r-rad_s": 0.0,  # Yaw rate (rad/s)
}

rate_ic = {
    "ic/u-fps": 500.0,  # Forward velocity (ft/s)
    "ic/v-fps": 0.0,  # Sideways velocity (ft/s)
    "ic/w-fps": -10.0,  # Vertical velocity (ft/s)
}

position_ic = {
    "ic/h-agl-ft": 5000.0,  # Z position (ft)
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
    "attitude": 30,
    "attitude_rate": 10,
    "translation_rate": 40,
    "position": 500,
    "aero": 7,
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
    ax.set_title("Distribution of initial condition randomization per type")
    ax.legend(frameon=False, loc="upper right")
    # remove the top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    figure.savefig("type_randomization_variance.png", dpi=300, bbox_inches="tight")
