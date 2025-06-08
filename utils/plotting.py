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
    
    axright00 = axes[0, 0].twinx()
    axright00.set_ylabel("xy position (ft)")
    axright00.legend(loc="lower right")
    axright00.plot(states["time"], states["x"], label="x position (ft)")
    axright00.plot(states["time"], states["y"], label="y position (ft)")
    axright00.legend(loc="lower right")

    # axes[0,0].plot(states['time'], states['z'], label='z position (ft)')
    axes[0, 0].plot(states["time"], states["altitude"], label="altitude (ft)", linewidth=4, color="black")
    # axes[0, 0].set_xlabel("time (s)")
    axes[0,0].set_xticklabels([])  # Hide x-tick labels

    axes[0, 0].set_ylabel("altitude (ft)")
    axes[0, 0].set_title("position")
    axes[0, 0].legend()

    axes[0, 1].plot(states["time"], states["u"], label="u velocity (fps)")
    axes[0, 1].plot(states["time"], states["v"], label="v velocity (fps)")
    axes[0, 1].plot(states["time"], -states["w"], label="w velocity (fps)")
    # axes[0, 1].set_xlabel("time (s)")
    axes[0,1].set_xticklabels([])  # Hide x-tick labels

    axes[0, 1].set_ylabel("velocity components")
    axes[0, 1].set_title("aircraft velocities")
    axes[0, 1].legend()

    axes[1, 0].plot(states["time"], states["phi"], label="phi (deg)")
    axes[1, 0].plot(states["time"], states["theta"], label="theta (deg)")

    # axes[1, 0].set_xlabel("time (s)")
    axes[1,0].set_xticklabels([])  # Hide x-tick labels

    axes[1, 0].set_ylabel("attitude angles (deg)")
    axes[1, 0].set_title("attitude")
    axes[1, 0].legend(loc="lower right")

    axes[1, 1].plot(states["time"], states["p"], label="p (deg/s)")
    axes[1, 1].plot(states["time"], states["q"], label="q (deg/s)")
    axes[1, 1].plot(states["time"], states["r"], label="r (deg/s)")
    # axes[1, 1].set_xlabel("time (s)")
    axes[1,1].set_xticklabels([])  # Hide x-tick labels

    axes[1, 1].set_ylabel("angular rates (deg/s)")
    axes[1, 1].set_title("angular rates")
    axes[1, 1].legend()

    axes[2, 0].plot(states["time"], states["alpha"], label="alpha (deg)")
    axes[2, 0].plot(states["time"], states["beta"], label="beta (deg)")
    axes[2, 0].set_xlabel("time (s)")
    axes[2, 0].set_ylabel("angle (deg)")
    axes[2, 0].set_title("aerodynamic angles")
    axes[2, 0].legend(loc="lower right")

    axright21 = axes[2, 1].twinx()
    axright21.plot(states["time"], states["psi"], label="psi (deg)", color="blue")
    axright21.set_ylabel("heading angle (deg)")
    axright21.legend(loc="upper right")
    axright21.yaxis.label.set_color("blue")
    axright21.axhline(y=0, color="gray", linewidth=2)
    axright21.axhline(y=360, color="gray", linewidth=2)
    axright21.set_yticks(np.arange(0, 360, 60))
    axright21.set_ylim(0, 360)
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

    #                width=bar_width, label="control penalty", color="orange", alpha=0.7)
    if rewards is None:
        axes[1, 2].axis("off")
        axes[0, 2].axis("off")  # Hide the empty subplot
    else:            
        axes[1, 2].plot(states["time"], rewards["total"], label="total reward", color="black", linewidth=2)
        axes[1, 2].plot(states["time"], rewards["preservation_bonus"], label="preservation bonus", color="green")
        axes[1, 2].plot(states["time"], rewards["smoothness_penalty"], label="smoothness bonus", color="red")
        axes[1, 2].plot(states["time"], rewards["control_penalty"], label="control penalty", color="orange")
        # axes[1, 2].set_xlabel("time (s)")
        axes[1,2].set_xticklabels([])  # Hide x-tick labels
        axes[1, 2].set_ylabel("reward components")
        axes[1, 2].set_title("reward components")
        axes[1, 2].legend()
        axes[1,2].set_ylim(-20, 100)

        # get rewards
        axes[0, 2].plot(states["time"], rewards["total"].cum_sum(), label="total reward", color="black", linewidth=2)
        axes[0, 2].plot(states["time"], rewards["preservation_bonus"].cum_sum(), label="preservation bonus", color="green")
        axes[0, 2].plot(states["time"], rewards["smoothness_penalty"].cum_sum(), label="smoothness bonus", color="red")
        axes[0, 2].plot(states["time"], rewards["control_penalty"].cum_sum(), label="control penalty", color="orange")
        # axes[0, 2].set_xlabel("time (s)")
        axes[0,2].set_xticklabels([])  # Hide x-tick labels
        axes[0, 2].set_ylabel("return")
        axes[0, 2].set_title("accumulated rewards")
        axes[0, 2].legend()
        # axes[0,2].plot(states["time"], rewards["total"].rolling_mean(100), label="smoothed total reward", color="blue")


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

def plot_smoothed_rewards(rewards, window_size=100):
    """Plot smoothed rewards over time."""
    smoothed_rewards = rewards.rolling(window_size).mean()
    plt.figure(figsize=(10, 5))
    plt.plot(smoothed_rewards, label=f"Smoothed Rewards (window={window_size})", color="blue")
    plt.xlabel("Episode Count")
    plt.ylabel("Smoothed Total Reward")
    plt.title("Smoothed Rewards Over Time")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("plots/smoothed_rewards.png")

if __name__ == "__main__":
    # load data from logs using pickle
    import pickle
    with open("logs/state_action_reward_history_episode_1000.pkl", "rb") as f:
        data = pickle.load(f)
    states = data["state"]
    actions = data["action"]
    rewards = data["reward"]
    plot_trajectory(states, actions, rewards)
    plot_path(states, interactive=False)

    # ipdb.set_trace()  # Set a breakpoint for debugging
    learning = pl.read_csv("training_logs/ppo_log.csv.monitor.csv", skip_rows=1)
    figure, ax = plt.subplots(figsize=(10, 5))
    # get smoothed rewards
    smoothed_rewards = learning["r"].rolling_mean(100)
    ax.plot(learning["episode_count"], learning["r"], label="Total Reward", color="black")
    ax.plot(learning["episode_count"], smoothed_rewards, label="Smoothed Total Reward", color="blue")
    ax.set_xlabel("Episode Count")
    ax.set_ylabel("Total Reward")
    ax.set_title("Learning Progress")
    ax.legend()
    figure.savefig("plots/learning_progress.png")
