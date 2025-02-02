import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.integrate import odeint
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.gridspec import GridSpec

plt.style.use('dark_background')

# ----------------------------
# Physical and Rocket Parameters
# ----------------------------
G = 6.67430e-11       # Gravitational constant (m^3 kg^-1 s^-2)
mass = 500.0          # kg (dry mass)
length = 10.0         # meters (rocket length)
radius = 0.5          # meters (rocket body radius)
Cd = 0.5              # Drag coefficient
A = np.pi * radius**2 # Cross-sectional area (m²)
rho0 = 1.225          # Air density at sea level (kg/m³)
H = 8000.0            # Scale height for Earth's atmosphere (m)
M_planet = 5.972e24   # kg (mass of Earth)
R_planet = 6_371_000  # meters (Earth's radius)

# ----------------------------
# Quaternion Math for Rotation
# ----------------------------
def quaternion_multiply(q1, q2):
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2
    ])

def rotation_matrix(q):
    w, x, y, z = q
    return np.array([
        [1 - 2*y**2 - 2*z**2, 2*x*y - 2*z*w, 2*x*z + 2*y*w],
        [2*x*y + 2*z*w, 1 - 2*x**2 - 2*z**2, 2*y*z - 2*x*w],
        [2*x*z - 2*y*w, 2*y*z + 2*x*w, 1 - 2*x**2 - 2*y**2]
    ])

# ----------------------------
# Derivatives for ODE Integration
# ----------------------------
def derivatives(state, t):
    x, y, z, vx, vy, vz, qw, qx, qy, qz, ωx, ωy, ωz = state
    dxdt = vx
    dydt = vy
    dzdt = vz
    r = np.sqrt(x**2 + y**2 + z**2)
    altitude = r - R_planet

    if altitude < 0:
        F_gravity = np.zeros(3)
        rho = 0.0
    else:
        F_gravity = (-G * M_planet * mass / r**3) * np.array([x, y, z])
        rho = rho0 * np.exp(-altitude / H)

    vel = np.array([vx, vy, vz])
    speed = np.linalg.norm(vel)
    F_drag = -0.5 * rho * Cd * A * speed * vel if speed > 1e-6 else np.zeros(3)
    a_total = (F_gravity + F_drag) / mass
    ax, ay, az = a_total

    omega_quat = np.array([0.0, ωx, ωy, ωz])
    current_q = np.array([qw, qx, qy, qz])
    q_deriv = 0.5 * quaternion_multiply(omega_quat, current_q)
    return [dxdt, dydt, dzdt, ax, ay, az, *q_deriv, 0.0, 0.0, 0.0]

# ----------------------------
# Initial Conditions and ODE Integration
# ----------------------------
initial_pos = [0.0, 0.0, R_planet]
initial_vel = [0.0, 0.0, 100000]  # initial upward velocity in m/s
initial_q = [1.0, 0.0, 0.0, 0.0]
initial_omega = [0.5, 0.0, 0.0]
state0 = initial_pos + initial_vel + initial_q + initial_omega
t = np.linspace(0, 360, 1000)
solution = odeint(derivatives, state0, t)

x, y, z = solution[:, 0], solution[:, 1], solution[:, 2]
qw, qx, qy, qz = solution[:, 6], solution[:, 7], solution[:, 8], solution[:, 9]
altitude = np.sqrt(x**2 + y**2 + z**2) - R_planet
vx, vy, vz = solution[:, 3], solution[:, 4], solution[:, 5]
speed = np.sqrt(vx**2 + vy**2 + vz**2)
g_acc = G * M_planet / (x**2 + y**2 + z**2)
apogee_idx = np.argmax(altitude)

# ----------------------------
# Visualization Setup
# ----------------------------
fig = plt.figure(figsize=(20, 12))
gs = GridSpec(3, 2, figure=fig, width_ratios=[2, 1])
ax_3d = fig.add_subplot(gs[:, 0], projection='3d')
ax_alt = fig.add_subplot(gs[0, 1])
ax_speed = fig.add_subplot(gs[1, 1])
ax_acc = fig.add_subplot(gs[2, 1])

# 3D Plot Configuration
ax_3d.set_title('Rocket Trajectory (Ascent: Cyan, Descent: Red)')
ax_3d.set_xlabel('X (m)')
ax_3d.set_ylabel('Y (m)')
ax_3d.set_zlabel('Altitude (m)')
ax_3d.xaxis.pane.set_facecolor((0.0, 0.0, 0.2))
ax_3d.yaxis.pane.set_facecolor((0.0, 0.0, 0.2))
ax_3d.zaxis.pane.set_facecolor((0.0, 0.0, 0.2))

# Set custom zoom limits to see a smaller area
zoom_range = 1e6  # adjust this value (in meters) as needed
ax_3d.set_xlim(-zoom_range, zoom_range)
ax_3d.set_ylim(-zoom_range, zoom_range)
ax_3d.set_zlim(R_planet - 1e5, R_planet + zoom_range)

# Initialize Rocket and Trajectory Elements
rocket_line, = ax_3d.plot([], [], [], 'r-', linewidth=3)
ascent_line, = ax_3d.plot([], [], [], 'c-', linewidth=1)
descent_line, = ax_3d.plot([], [], [], 'r-', linewidth=1)
rocket_points = np.array([[0, 0, -length/2], [0, 0, length/2]])

# Telemetry Plot Setup
for ax, title, color in zip([ax_alt, ax_speed, ax_acc],
                            ['Altitude vs Time', 'Speed vs Time', 'Gravitational Acceleration vs Time'],
                            ['c', 'm', 'y']):
    ax.set_title(title)
    ax.grid(True)
    ax.set_facecolor((0.0, 0.0, 0.1))
    ax.set_xlim(0, t[-1])
    # Set an initial y-axis limit. It will autoscale during the animation.
    ax.set_ylim(0, max([altitude.max(), speed.max(), g_acc.max()]))

alt_line, = ax_alt.plot([], [], 'c-')
speed_line, = ax_speed.plot([], [], 'm-')
acc_line, = ax_acc.plot([], [], 'y-')

# ----------------------------
# Animation Function
# ----------------------------
def update(frame):
    # Update Rocket Position
    current_pos = [x[frame], y[frame], z[frame]]
    R = rotation_matrix([qw[frame], qx[frame], qy[frame], qz[frame]])
    transformed = rocket_points.dot(R.T) + current_pos
    rocket_line.set_data(transformed[:, 0], transformed[:, 1])
    rocket_line.set_3d_properties(transformed[:, 2])

    # Update Trajectory
    if frame <= apogee_idx:
        ascent_line.set_data(x[:frame+1], y[:frame+1])
        ascent_line.set_3d_properties(z[:frame+1])
    else:
        ascent_line.set_data(x[:apogee_idx+1], y[:apogee_idx+1])
        ascent_line.set_3d_properties(z[:apogee_idx+1])
        descent_line.set_data(x[apogee_idx:frame+1], y[apogee_idx:frame+1])
        descent_line.set_3d_properties(z[apogee_idx:frame+1])

    # Update Telemetry
    alt_line.set_data(t[:frame+1], altitude[:frame+1])
    speed_line.set_data(t[:frame+1], speed[:frame+1])
    acc_line.set_data(t[:frame+1], g_acc[:frame+1])
    
    # Autoscale telemetry axes for current data
    for ax, data in zip([ax_alt, ax_speed, ax_acc], [altitude, speed, g_acc]):
        ax.set_ylim(0, data[:frame+1].max()*1.1)
    
    return rocket_line, ascent_line, descent_line, alt_line, speed_line, acc_line

# ----------------------------
# Create Animation
# ----------------------------
ani = FuncAnimation(fig, update, frames=len(t), interval=20, blit=True)
plt.show()
