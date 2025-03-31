import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy.stats import vonmises

# -------------------------
# Simulation Classes
# -------------------------

class MigratingCell:
    def __init__(self, mode="Random", persistency=0.9, bias_direction=(1,0), step_size=1.0):
        """
        mode: "Random", "Persistent", "Biased", or "Bias-Persistent"
        persistency: for "Persistent" and "Bias-Persistent" modes, the probability to retain the previous angle.
        bias_direction: a tuple (bx, by) for biased motions.
        step_size: step size per update.
        """
        self.mode = mode
        self.persistency = persistency
        self.bias_direction = np.array(bias_direction, dtype=float)
        if np.linalg.norm(self.bias_direction) != 0:
            self.bias_direction = self.bias_direction / np.linalg.norm(self.bias_direction)
        self.step_size = step_size
        self.position = np.array([0.0, 0.0])
        # Initialize with a random angle (for all modes)
        self.angle = np.random.uniform(0, 2*np.pi)
        self.trajectory = [self.position.copy()]
    
    def step(self):
        if self.mode == "Random":
            # Always choose a new random angle
            self.angle = np.random.uniform(0, 2*np.pi)
        elif self.mode == "Persistent":
            # With probability persistency, keep previous angle; otherwise choose uniformly at random.
            if np.random.rand() >= self.persistency:
                self.angle = np.random.uniform(0, 2*np.pi)
        elif self.mode == "Biased":
            # Always choose new angle from biased distribution
            # Use a von Mises distribution with fixed concentration kappa (e.g. 4)
            if np.linalg.norm(self.bias_direction) == 0:
                # If no bias provided, fallback to uniform random.
                self.angle = np.random.uniform(0, 2*np.pi)
            else:
                bias_angle = np.arctan2(self.bias_direction[1], self.bias_direction[0])
                self.angle = vonmises.rvs(kappa=4, loc=bias_angle)
        elif self.mode == "Bias-Persistent":
            # With probability persistency, keep previous angle;
            # otherwise, choose a new biased angle as in "Biased" mode.
            if np.random.rand() >= self.persistency:
                if np.linalg.norm(self.bias_direction) == 0:
                    self.angle = np.random.uniform(0, 2*np.pi)
                else:
                    bias_angle = np.arctan2(self.bias_direction[1], self.bias_direction[0])
                    self.angle = vonmises.rvs(kappa=4, loc=bias_angle)
        else:
            raise ValueError("Invalid mode")
        
        # Compute displacement based on the chosen angle.
        delta = self.step_size * np.array([np.cos(self.angle), np.sin(self.angle)])
        self.position += delta
        self.trajectory.append(self.position.copy())
    
    def get_trajectory(self):
        return np.array(self.trajectory)
    
    def get_velocities(self):
        traj = self.get_trajectory()
        return np.diff(traj, axis=0)
    
    def get_angles(self):
        velocities = self.get_velocities()
        angles = np.arctan2(velocities[:,1], velocities[:,0])
        return angles

class MigrationSimulation:
    def __init__(self, num_cells=50, mode="Random", persistency=0.9, bias_direction=(1,0), step_size=1.0, num_steps=200):
        self.num_cells = num_cells
        self.mode = mode
        self.persistency = persistency
        self.bias_direction = bias_direction
        self.step_size = step_size
        self.num_steps = num_steps
        self.cells = [MigratingCell(mode, persistency, bias_direction, step_size) for _ in range(num_cells)]
    
    def run(self):
        for step in range(self.num_steps):
            for cell in self.cells:
                cell.step()
    
    def get_all_trajectories(self):
        return [cell.get_trajectory() for cell in self.cells]
    
    def get_all_velocities(self):
        return [cell.get_velocities() for cell in self.cells]
    
    def get_all_angles(self):
        return [cell.get_angles() for cell in self.cells]
    
    def compute_MSD(self):
        trajs = self.get_all_trajectories()
        trajs = np.array(trajs)  # shape: (num_cells, num_steps+1, 2)
        squared_displacements = np.sum(trajs**2, axis=2)
        MSD = np.mean(squared_displacements, axis=0)
        return MSD

    def compute_velocity_autocorrelation(self):
        all_v = self.get_all_velocities()
        max_len = min(v.shape[0] for v in all_v)
        autocorr = np.zeros(max_len)
        count = 0
        for v in all_v:
            v = v[:max_len]
            for lag in range(max_len):
                products = np.sum(v[:max_len-lag] * v[lag:max_len], axis=1)
                autocorr[lag] += np.mean(products)
            count += 1
        autocorr /= count
        return autocorr

    def compute_speed_distribution(self):
        all_v = self.get_all_velocities()
        speeds = np.concatenate([np.linalg.norm(v, axis=1) for v in all_v])
        return speeds

    def compute_angle_distribution(self):
        all_angles = self.get_all_angles()
        angles = np.concatenate(all_angles)
        return angles

# -------------------------
# Streamlit App Layout
# -------------------------

st.title("Cell Migration Simulation & Population Statistics")

# Sidebar: Simulation Settings
st.sidebar.header("Simulation Settings")
mode = st.sidebar.selectbox("Select Motion Mode:", 
                            ["Random", "Persistent", "Biased", "Bias-Persistent"])
num_cells = st.sidebar.number_input("Number of Cells", min_value=1, max_value=200, value=50, step=1)
num_steps = st.sidebar.slider("Number of Steps", min_value=10, max_value=1000, value=200, step=10)
step_size = st.sidebar.slider("Step Size", min_value=0.1, max_value=10.0, value=1.0, step=0.1)
if mode in ["Persistent", "Bias-Persistent"]:
    persistency = st.sidebar.slider("Persistence Factor (0-1)", min_value=0.0, max_value=1.0, value=0.8, step=0.05)
else:
    persistency = 0.0
if mode in ["Biased", "Bias-Persistent"]:
    bias_x = st.sidebar.number_input("Bias X Component", value=1.0)
    bias_y = st.sidebar.number_input("Bias Y Component", value=0.0)
    bias_direction = (bias_x, bias_y)
else:
    bias_direction = (0, 0)

# Run Simulation
if st.sidebar.button("Run Simulation"):
    sim = MigrationSimulation(num_cells=num_cells,
                              mode=mode,
                              persistency=persistency,
                              bias_direction=bias_direction,
                              step_size=step_size,
                              num_steps=num_steps)
    sim.run()
    trajectories = sim.get_all_trajectories()
    
    # Panel 1: Final Migration Process
    st.header("Migration Process (Final Trajectories)")
    fig_proc, ax_proc = plt.subplots(figsize=(6, 6))
    cmap = cm.get_cmap("viridis", num_cells)
    for idx, traj in enumerate(trajectories):
        ax_proc.plot(traj[:,0], traj[:,1], marker="o", markersize=2, color=cmap(idx))
        ax_proc.plot(traj[0,0], traj[0,1], marker="s", color="black")  # starting point
        ax_proc.plot(traj[-1,0], traj[-1,1], marker="*", color="red")  # final point
    ax_proc.set_title("Final Trajectories")
    ax_proc.set_xlabel("X Position")
    ax_proc.set_ylabel("Y Position")
    ax_proc.axis("equal")
    st.pyplot(fig_proc)
    
    # Panel 2: Population Statistics
    st.header("Population Statistics")
    
    # MSD Plot (log-log)
    MSD = sim.compute_MSD()
    time_array = np.arange(len(MSD))
    fig_msd, ax_msd = plt.subplots()
    ax_msd.loglog(time_array[1:], MSD[1:], marker="o")
    ax_msd.grid(which="major", linestyle="--")
    ax_msd.set_title("Mean Squared Displacement (MSD)")
    ax_msd.set_xlabel("Time Step (log scale)")
    ax_msd.set_ylabel("MSD (log scale)")
    st.pyplot(fig_msd)
    
    # Velocity Autocorrelation
    autocorr = sim.compute_velocity_autocorrelation()
    fig_auto, ax_auto = plt.subplots()
    ax_auto.plot(autocorr, marker="o")
    ax_auto.set_title("Velocity Autocorrelation Function")
    ax_auto.set_xlabel("Lag")
    ax_auto.set_ylabel("Autocorrelation")
    st.pyplot(fig_auto)
    
    # Angle Distribution Histogram
    angles = sim.compute_angle_distribution()
    fig_angle, ax_angle = plt.subplots()
    ax_angle.hist(angles, bins=30, density=True, color="skyblue", edgecolor="black")
    mean_angle = np.mean(angles)
    ax_angle.axvline(mean_angle, color='k', linestyle='dashed', linewidth=1)
    ax_angle.set_title("Turning Angle Distribution")
    ax_angle.set_xlabel("Angle (radians)")
    ax_angle.set_ylabel("Probability Density")
    st.pyplot(fig_angle)
    
    # Velocity Distribution Histogram
    speeds = sim.compute_speed_distribution()
    fig_speed, ax_speed = plt.subplots()
    ax_speed.hist(speeds, bins=30, density=True, color="lightgreen", edgecolor="black")
    ax_speed.set_title("Instantaneous Speed Distribution")
    ax_speed.set_xlabel("Speed")
    ax_speed.set_ylabel("Probability Density")
    st.pyplot(fig_speed)
