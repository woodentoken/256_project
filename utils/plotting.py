import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import ipdb

plt.rcParams['lines.linewidth'] = 2

def plot_trajectory(states, actions, rewards):
    """Plot the state trajectory using polars DataFrame."""
    # Convert polars DataFrame to pandas for plotting
    figure, axes = plt.subplots(3, 3, figsize=(15, 10))
    states = states.with_columns(time = states["time"] - states["time"][0])  # Normalize time to start from 0
    axes[0, 0].plot(states["time"], states["x"], label="x position (ft)")
    axes[0, 0].plot(states["time"], states["y"], label="y position (ft)")
    axes[0, 0].plot(states["time"], states["altitude"], label="altitude (ft)")

    # axes[0,0].plot(states['time'], states['z'], label='z position (ft)')
    axes[0, 0].set_xlabel("time (s)")
    axes[0, 0].set_ylabel("distance (ft)")
    axes[0, 0].set_title("position")
    axes[0, 0].legend()

    axes[0, 1].plot(states["time"], states["u"], label="u velocity (fps)")
    axes[0, 1].plot(states["time"], states["v"], label="v velocity (fps)")
    axes[0, 1].plot(states["time"], -states["w"], label="w velocity (fps)")
    axes[0, 1].set_xlabel("time (s)")
    axes[0, 1].set_ylabel("velocity components")
    axes[0, 1].set_title("aircraft velocities")
    axes[0, 1].legend()

    axes[1, 0].plot(states["time"], states["phi"], label="phi (deg)")
    axes[1, 0].plot(states["time"], states["theta"], label="theta (deg)")

    axes[1, 0].set_xlabel("time (s)")
    axes[1, 0].set_ylabel("attitude angles (deg)")
    axes[1, 0].set_title("attitude")
    axes[1, 0].legend(loc="lower right")

    axes[1, 1].plot(states["time"], states["p"], label="p (deg/s)")
    axes[1, 1].plot(states["time"], states["q"], label="q (deg/s)")
    axes[1, 1].plot(states["time"], states["r"], label="r (deg/s)")
    axes[1, 1].set_xlabel("time (s)")
    axes[1, 1].set_ylabel("angular rates (deg/s)")
    axes[1, 1].set_title("angular rates")
    axes[1, 1].legend()

    axes[2, 0].plot(states["time"], states["alpha"], label="alpha (deg)")
    axes[2, 0].plot(states["time"], states["beta"], label="beta (deg)")
    axes[2, 0].set_xlabel("time (s)")
    axes[2, 0].set_ylabel("angle (deg)")
    axes[2, 0].set_title("aerodynamic angles")
    axes[2, 0].legend(loc="lower right")

    axright = axes[2, 1].twinx()
    axright.plot(states["time"], states["psi"], label="psi (deg)", color="blue")
    axright.set_ylabel("heading angle (deg)")
    axright.legend(loc="upper right")
    axright.yaxis.label.set_color("blue")
    axright.axhline(y=0, color="gray", linewidth=2)
    axright.axhline(y=360, color="gray", linewidth=2)
    axright.set_yticks(np.arange(0, 360, 60))
    axright.set_ylim(0, 360)
    axes[2, 1].plot(states["time"], states["gamma"], label="gamma (deg)", color="green")
    axes[2, 1].set_xlabel("time (s)")
    axes[2, 1].set_ylabel("heading angle (deg)")
    axes[2, 1].set_title("navigation angles")
    axes[2, 1].yaxis.label.set_color("green")
    axes[2, 1].legend(loc="lower right")

    axes[2, 2].plot(states["time"], actions["aileron"], label="aileron input")
    axes[2, 2].plot(states["time"], actions["elevator"], label="elevator input")
    axes[2, 2].plot(states["time"], actions["rudder"], label="rudder input")
    if "throttle" in actions:
        axes[2, 2].plot(states["time"], actions["throttle"], label="throttle input")

    axes[2, 2].set_xlabel("time (s)")
    axes[2, 2].set_ylabel("normalized control inputs")
    axes[2, 2].set_title("control input history")
    axes[2, 2].legend()

    for ax in axes.flat:
        ax.axhline(y=0, color="black", linewidth=0.5)
        ax.grid(True, alpha=0.3)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    # axes[1, 2].plot(states["time"], rewards["total"], label="airspeed (fps)", color="black")
    # # Create stacked bar chart for reward components
    # bar_width = states["time"][1] - states["time"][0] if len(states["time"]) > 1 else 1
    
    # axes[1, 2].bar(states["time"], rewards["preservation_bonus"], 
    #                width=bar_width, label="preservation bonus", color="green", alpha=0.7)
    # axes[1, 2].bar(states["time"], rewards["smoothness_bonus"], 
    #                bottom=rewards["preservation_bonus"],
    #                width=bar_width, label="smoothness bonus", color="blue", alpha=0.7)
    # axes[1, 2].bar(states["time"], rewards["collision_penalty"], 
    #                bottom=rewards["preservation_bonus"] + rewards["smoothness_bonus"],
    #                width=bar_width, label="collision penalty", color="red", alpha=0.7)
    # axes[1, 2].bar(states["time"], rewards["control_penalty"], 
    #                bottom=rewards["preservation_bonus"] + rewards["smoothness_bonus"] + rewards["collision_penalty"],
    #                width=bar_width, label="control penalty", color="orange", alpha=0.7)
    axes[1, 2].plot(states["time"], rewards["total"], label="total reward", color="black")
    axes[1, 2].plot(states["time"], rewards["preservation_bonus"], label="preservation bonus", color="green")
    axes[1, 2].plot(states["time"], rewards["smoothness_bonus"], label="smoothness bonus", color="blue")
    axes[1, 2].plot(states["time"], rewards["collision_penalty"], label="collision penalty", color="red")
    axes[1, 2].plot(states["time"], rewards["control_penalty"], label="control penalty", color="orange")
    axes[1, 2].plot(states["time"], rewards["attitude_penalty"], label="attitude penalty", color="purple")
    axes[1, 2].set_xlabel("time (s)")
    axes[1, 2].set_ylabel("reward components")
    axes[1, 2].set_title("reward components")
    axes[1, 2].legend()

    axes[0, 2].axis("off")  # Hide the empty subplot

    figure.tight_layout()
    figure.savefig("plots/flight_dynamics_trajectory_sample.png")


def plot_path(states, interactive=False):
    # plot the path of the aircraft in 3D space
    fig = plt.figure(figsize=(15, 10))
    ax = fig.add_subplot(111, projection="3d")

    ax.plot(
        states["x"][0],
        states["y"][0],
        states["altitude"][0],
        marker="o",
        markersize=5,
        color="green",
        label="Start Point",
    )
    ax.plot(
        states["x"][-1],
        states["y"][-1],
        states["altitude"][-1],
        marker="o",
        markersize=5,
        color="red",
        label="End Point",
    )
    ax.plot(states["x"], states["y"], states["altitude"], label="Flight Path", color="black", linewidth=1.5)

    ax.set_xlabel("\n\nX Position (ft)")
    ax.set_ylabel("\n\nY Position (ft)")
    ax.set_zlabel("\n\nAltitude (ft)")
    ax.set_title("3D Flight Path")
    ax.set_box_aspect([1, 1, 1])  # Aspect ratio for better visualization
    ax.set_aspect("equal")
    ax.xaxis.set_major_locator(plt.MaxNLocator(3))
    ax.yaxis.set_major_locator(plt.MaxNLocator(3))
    ax.zaxis.set_major_locator(plt.MaxNLocator(5))

    if interactive:
        plt.show()
    else:
        fig.savefig("plots/flight_path_sample.png")

if __name__ == "__main__":
    # load data from logs using pickle
    import pickle
    with open("logs/state_action_reward_history_episode_400.pkl", "rb") as f:
        data = pickle.load(f)
    states = data["state"]
    actions = data["action"]
    rewards = data["reward"]
    ipdb.set_trace()  # Debugging breakpoint
    plot_trajectory(states, actions, rewards)
    plot_path(states, interactive=False)
