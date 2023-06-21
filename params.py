import numpy as np

''' Sim '''
t_start = 0
dt = 0.01
sim_t =  20
# t_plot = 1.0
is_2d = True

cvo_dt = 0.1

num_quads = 20

outfile_name = 'outfile.json'

''' CVO '''
collision_radius = 1
buffer_power = 2
buffer_on = False
collision_range = 1500
start_radius = 50

add_kalman = True
add_uncertianty = True
drop_prob = 0.8

gps_pos_stdev = 1.0
gps_vel_stdev = 0.1


''' Gen Kalman Filter '''
sigmaQ_vel = 3
alphaQ_vel = 2.5
sigmaQ_jrk = 0.75 #0075
alphaQ_jrk = 0.75

sigmaR_pos = 2.0
sigmaR_vel = 2.0

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

max_roll = np.pi/4 #0.196
max_pitch = np.pi/4 #
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
max_vel = 10.0
waypoint_threshold = 0.1
waypoint_velocity_threshold = 0.5
