import numpy as np
from numpy.linalg import norm
from utils import roll, pitch, yaw, exp, rota, rotp, from_euler, R_v_to_v1, R_v_to_b
from simple_pid import SimplePID

import copy

from IPython.core.debugger import set_trace

M_PI = 3.14159265359

class Controller:
    def __init__(self, params, id):
        self.id = id
        self.throttle_eq_ = params.throttle_eq

        self.gravity_ = params.gravity
        self.K_v_ = params.K_v

        self.max_roll_ = params.max_roll
        self.max_pitch_ = params.max_pitch
        self.max_yaw_rate_ = params.max_yaw_rate
        self.max_throttle_ = params.max_throttle

        self.e3 = np.array([[0.0], [0.0], [1.0]])


        ''' Calculate max accelerations. Assuming that equilibrium throttle produces
         1 g of acceleration and a linear thrust model, these max acceleration
         values are computed in g's as well. '''
        max_accel_xy = np.sin(np.arccos(self.throttle_eq_)) / self.throttle_eq_ / np.sqrt(2.) + 10

        tau = params.tau

        P = params.x_dot_P
        I = params.x_dot_I
        D = params.x_dot_D
        self.PID_x_dot_ = SimplePID(P, I, D, max_accel_xy, -max_accel_xy, tau)

        P = params.y_dot_P
        I = params.y_dot_I
        D = params.y_dot_D
        self.PID_y_dot_ = SimplePID(P, I, D, max_accel_xy, -max_accel_xy, tau)

        P = params.z_dot_P
        I = params.z_dot_I
        D = params.z_dot_D
        self.PID_z_dot_ = SimplePID(P, I, D, max_accel_xy, -max_accel_xy, tau)

        P = params.psi_P
        I = params.psi_I
        D = params.psi_D
        self.PID_psi_ = SimplePID(P, I, D, np.Inf, -np.Inf, tau)

        max_tau_x = params.max_tau_x
        max_tau_y = params.max_tau_y
        max_tau_z = params.max_tau_z

        P = params.roll_P
        I = params.roll_I
        D = params.roll_D
        self.PID_roll_ = SimplePID(P, I, D, max_tau_x, -max_tau_x, tau)

        P = params.pitch_P
        I = params.pitch_I
        D = params.pitch_D
        self.PID_pitch_ = SimplePID(P, I, D, max_tau_y, -max_tau_y, tau)

        P = params.yaw_rate_P
        I = params.yaw_rate_I
        D = params.yaw_rate_D
        self.PID_yaw_rate_ = SimplePID(P, I, D, max_tau_z, -max_tau_z, tau)

    def saturate(self, x, max, min):
        if x > max:
            return max
        if x < min:
            return min
        return x

    # def computeControl(self, state, dt, vel_c, psi_c = 0):
    #     if dt <= 1e-8:
    #         return
    #
    #     x_ = copy.copy(state[0:3])
    #     v_ = copy.copy(state[3:6])
    #     q_ = copy.copy(state[6:10])
    #     omega_ = copy.copy(state[10:13])
    #
    #     phi = roll(q_)
    #     theta = pitch(q_)
    #     psi = yaw(q_)
    #
    #     R_v_to_v1_ = R_v_to_v1(psi)
    #     R_v1_to_b_ = R_v_to_b(phi, theta, 0.0)
    #
    #     if np.abs(psi_c + 2*M_PI - psi) < np.abs(psi_c - psi):
    #         psi_c += 2*M_PI
    #
    #     elif np.abs(psi_c - 2*M_PI - psi) < np.abs(psi_c - psi):
    #         psi_c -= 2*M_PI
    #
    #
    #     vhat = R_v1_to_b_.T @ v_ # vehicle-1 velocity estimate
    #     vel_c = R_v_to_v1_.T @ vel_c
    #
    #     k_tilde = self.throttle_eq_ * (self.e3 - (1.0 / self.gravity_) * self.K_v_ @ (vel_c - vhat))
    #
    #     # pack up throttle command
    #     throttle_c = self.e3.T @ R_v1_to_b_ @ k_tilde
    #
    #     kd = (1.0 / throttle_c) * k_tilde # desired body z direction
    #     kd = kd / norm(kd) # need direction only
    #     tilt_angle = np.arccos(self.e3.T @ kd) # desired tilt
    #
    #     q_c = np.array([[1.0],[0.0],[0.0],[0.0]])
    #     if tilt_angle > 1e-6:
    #         k_cross_kd = np.cross(self.e3.T, kd.T).T
    #         q_c = exp(tilt_angle * k_cross_kd / norm(k_cross_kd))
    #
    #     # q_c = from_euler(roll(q_c), pitch(q_c), psi_c)
    #
    #     phi_c = roll(q_c)
    #     theta_c = pitch(q_c)
    #     r_c = psi_c
    #
    #     phi_c = self.saturate(phi_c, self.max_roll_, -self.max_roll_)
    #     theta_c = self.saturate(theta_c, self.max_pitch_, -self.max_pitch_)
    #     r_c = self.saturate(r_c, self.max_yaw_rate_, -self.max_yaw_rate_)
    #     throttle_c = self.saturate(throttle_c[0,0], self.max_throttle_, 0)
    #
    #     tau_x = 0 #self.PID_roll_.computePID(phi_c, phi, dt, omega_[0,0])
    #     tau_y = self.PID_pitch_.computePID(theta_c, theta, dt, omega_[1,0])
    #     tau_z = 0 #self.PID_yaw_rate_.computePID(r_c, omega_[2,0], dt)
    #
    #     ret_u = np.array([[tau_x], [tau_y], [tau_z], [throttle_c]])
    #
    #     # set_trace()
    #     # print('theta_c: ', theta_c)
    #     # print('tau_y: ', tau_y)
    #     # print('k_tilde: ', k_tilde)
    #     # print('throttle_c: ', throttle_c)
    #     # print('v_: ', v_)
    #     # print('vel_c: ', vel_c)
    #     print('ret_u: ', ret_u)
    #     # set_trace()
    #     return ret_u


    def computeControl(self, state, dt, vel_c, t, psi_c = 0):
        if dt <= 1e-8:
            return

        x_ = copy.copy(state[0:3])
        v_ = copy.copy(state[3:6])
        q_ = copy.copy(state[6:10])
        omega_ = copy.copy(state[10:13])


        phi = roll(q_)
        theta = pitch(q_)
        psi = yaw(q_)


        ''' Calculate desired yaw rate
        First, determine the shortest direction to the commanded psi '''
        if np.abs(psi_c + 2*M_PI - psi) < np.abs(psi_c - psi):
            psi_c += 2*M_PI

        elif np.abs(psi_c - 2*M_PI - psi) < np.abs(psi_c - psi):
            psi_c -= 2*M_PI

        # set_trace()

        r_c = self.PID_psi_.computePID(psi_c, psi, dt)


        ''' Compute desired accelerations (in terms of g's) in the vehicle 1 frame
        Rotate body frame velocities to vehicle 1 frame velocities '''
        # u = v_[0,0]
        # v = v_[1,0]
        # w = v_[2,0]
        #
        sinp = np.sin(phi)
        cosp = np.cos(phi)
        sint = np.sin(theta)
        cost = np.cos(theta)
        # pxdot = cost * u + sinp * sint * v + cosp * sint * w
        # pydot = cosp * v - sinp * w
        # pddot = -sint * u + sinp * cost * v + cosp * cost * w

        v_ = rota(q_, v_)
        pxdot = v_[0,0]
        pydot = v_[1,0]
        pddot = v_[2,0]

        R_v_to_v1_ = R_v_to_v1(psi)
        vel_c = R_v_to_v1_ @ vel_c
        x_dot_c = vel_c[0,0]
        y_dot_c = vel_c[1,0]
        z_dot_c = vel_c[2,0]

        # print('vel_c: ', vel_c)
        # print('v_: ', v_)
        ax = self.PID_x_dot_.computePID(x_dot_c, pxdot, dt)
        ay = self.PID_y_dot_.computePID(y_dot_c, pydot, dt)
        az = self.PID_z_dot_.computePID(z_dot_c, pddot, dt)
        # print('ax: ', ax)
        # set_trace()

        ''' Compute throttle commanded
         Model inversion (m[ax;ay;az] = m[0;0;g] + R'[0;0;-T] '''
        total_acc_c = np.sqrt((1.0 - az) * (1.0 - az) + \
                                  ax * ax + ay * ay)  # (in g's)
        phi_c = 0
        theta_c = 0
        if total_acc_c > 0.001:
            phi_c = np.arcsin(ay / total_acc_c)
            theta_c = -1.0*np.arcsin(ax / total_acc_c)



        # Compute desired thrust based on current pose
        throttle_c = (1.0 - az) * self.throttle_eq_ / cosp / cost

        phi_c = self.saturate(phi_c, self.max_roll_, -self.max_roll_)
        theta_c = self.saturate(theta_c, self.max_pitch_, -self.max_pitch_)
        r_c = self.saturate(r_c, self.max_yaw_rate_, -self.max_yaw_rate_)
        throttle_c = self.saturate(throttle_c, self.max_throttle_, 0)

        tau_x = self.PID_roll_.computePID(phi_c, phi, dt, omega_[0,0])
        tau_y = self.PID_pitch_.computePID(theta_c, theta, dt, omega_[1,0])
        tau_z = self.PID_yaw_rate_.computePID(r_c, omega_[2,0], dt)

        # ret_u = np.array([[tau_x], [tau_y], [tau_z], [throttle_c]])
        ret_u = np.array([[throttle_c], [tau_x], [tau_y], [tau_z]])

        # ret_u = np.array([[ax], [ay], [r_c], [throttle_c]])


        # print('theta: ', theta)
        # print('theta_c: ', theta_c, theta)
        # print('tau: ', tau_y)
        # set_trace()
        # print(ret_u)
        return ret_u
