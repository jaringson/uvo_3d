import numpy as np

''' Sim '''
t_start = 0
dt = 0.01
sim_t = 100
# t_plot = 1.0

cvo_dt = 0.1

num_quads = 40

outfile_name = 'outfile.json'

''' CVO '''
slack_gamme = 10

collision_radius = 10
buffer_power = 50
buffer_on = True
collision_range = 100
start_radius = 100

add_kalman = True
add_uncertianty = True
drop_prob = 0.5

gps_pos_stdev = 3.0
gps_vel_stdev = 1.0


''' Gen Kalman Filter '''
sigmaQ_vel = 3
alphaQ_vel = 0.5
sigmaQ_jrk = 0.0075
alphaQ_jrk = 0.5

sigmaR_pos = 1.0
sigmaR_vel = 0.1

''' Control '''
tau = 0.05

x_dot_P = 0.5
x_dot_I = 0.0
x_dot_D = 0.05

y_dot_P = 0.5
y_dot_I = 0.0
y_dot_D = 0.05

z_dot_P = 0.4
z_dot_I = 0.25
z_dot_D = 0.1

psi_P = 2.0
psi_I = 0.0
psi_D = 0.0

throttle_eq = 0.5

max_roll = 0.1 #np.pi/4 #0.196
max_pitch = 0.1 #np.pi/4 #
max_yaw_rate = 1.5 #0.785
max_throttle = 1.0 #0.85

roll_P = 10.0 #10.0
roll_I = 0.0
roll_D = 5.0
pitch_P = 10.0
pitch_I = 0.0
pitch_D = 5.0
yaw_rate_P = 10.0
yaw_rate_I = 0.0
yaw_rate_D = 0.0

max_tau_x = 200.0
max_tau_y = 200.0
max_tau_z = 50.0




''' Dynamics '''
# x0 = np.array([[0], [0], [0]])
v0 = np.array([[0], [0], [0]])
q0 = np.array([[1], [0], [0], [0]])
omega0 = np.array([[0], [0], [0]])

max_thrust = 98.0665
mass = 5.0
linear_drag = np.diag([0.1, 0.1, 0.001])
angular_drag = np.diag([0.001, 0.001, 0.001])
inertia_matrix = np.diag([0.6271, 0.6271, 1.25])
K_v = np.diag([1.5, 1.5, 1.5])

gravity = 9.80665

''' Waypoint Manager '''
Kp =  [0.5, 0.5, 0.5]
max_vel = 5.0
waypoint_threshold = 0.1
waypoint_velocity_threshold = 0.5
