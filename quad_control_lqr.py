import numpy as np
from numpy.linalg import norm, inv
import control
from copy import copy as cp
# from simple_pid import SimplePID

from utils import roll, pitch, yaw, exp, rota, rotp, from_euler, R_v_to_v1, R_v_to_b
from utils import skew, from_two_unit_vectors, boxminus

from commanded_state import CommandState

from IPython.core.debugger import set_trace

M_PI = 3.14159265359

class Controller:
    def __init__(self, params, id, _):
        self.id = id
        self.throttle_eq_ = params.throttle_eq

        self.gravity_ = params.gravity

        self.max_thrust_ = params.max_thrust
        self.mass_ = params.mass
        self.linear_drag_ = params.linear_drag
        self.angular_drag_ = params.angular_drag
        self.inertia_matrix_ = params.inertia_matrix
        self.inertia_inv_ = inv(self.inertia_matrix_)

        self.e3 = params.e3


        self.dt = params.dt


        self.pos_max_err = params.lqr_max_pos_error
        self.vel_max_err = params.lqr_max_vel_error
        self.q_max_err = params.lqr_max_ang_error
        self.s_max_err = params.lqr_max_throttle_error
        self.omega_max_err = params.lqr_max_omega_error

        self.CommandState = CommandState(params)

        self.Q = np.eye(12)
        self.R = np.eye(4)

        self.Q[0,0] = 0
        self.Q[1,1] = 0
        self.Q[2,2] = 0

        self.Q[3,3] = 5
        self.Q[4,4] = 5
        self.Q[5,5] = 5


        # self.P = np.eye(12)*0.01

        self.J = 0

    def saturate(self, x, min, max):
        # if x > max:
        #     return max
        # if x < min:
        #     return min
        x = np.clip(x, min, max)
        return x

    # def computeControl(self, state, t):
    def computeControl(self, state, dt, vel_c, t, psi_c = 0):
        # if dt <= 1e-8:
        #     return

        x_ = cp(state[0:3])
        v_ = cp(state[3:6])
        q_ = cp(state[6:10])
        omega_ = cp(state[10:13])


        phi = roll(q_)
        theta = pitch(q_)
        psi = yaw(q_)

        R_vb = R_v_to_b(phi, theta, psi)

        A = np.zeros((12,12))
        B = np.zeros((12,4))


        ### Position
        A[0:3,3:6] = R_vb.T # d/dv
        A[0:3,6:9] = -R_vb.T @ skew(v_) # d/dq
        ### Velocity
        A[3:6,3:6] = -self.linear_drag_ - skew(omega_) # d/dv
        A[3:6,6:9] = skew(self.gravity_ * R_vb @ self.e3) # d/dq
        A[3:6,9:12] = skew(v_) #d/(d omega)
        ### Attitude
        A[6:9,6:9] = -skew(omega_) # d/dq
        A[6:9,9:12] = np.eye(3) # d /(d omega)
        ### Attitude Rate
        A[9:12,9:12] = -self.inertia_inv_ @ ( skew(omega_)@self.inertia_matrix_ + \
                                            -skew(self.inertia_matrix_@omega_) + \
                                            self.angular_drag_) # d/(d omega)

        ### Velocity
        B[3:6,0] = (-self.e3*self.max_thrust_/self.mass_).T # d/ds
        # B[3:6,1:4] = skew(v_)
        ### Attitude Rate
        B[9:12,1:4] = self.inertia_inv_ # d/d tau


        # A = A[2:-1,2:-1]
        # B = B[2:-1]
        # set_trace()

        K, P, E = control.lqr(A, B, self.Q, self.R)

        ### Get Commanded States
        pos_c, v_c, q_c, omega_c, u_s, u_tau = \
                self.CommandState.get_commanded_state(cp(state), cp(t), cp(vel_c))

        u_c = np.block([[u_s],
                        [u_tau]])

        pos_err = x_ - pos_c
        # pos_err =  np.zeros((3,1)) #
        pos_err = self.saturate(pos_err, -self.pos_max_err, self.pos_max_err)

        vel_err = v_ - v_c
        vel_err = self.saturate(vel_err, -self.vel_max_err, self.vel_max_err)

        q_err = boxminus(cp(q_), cp(q_c))
        q_err = self.saturate(q_err, -self.q_max_err, self.q_max_err)

        omega_err = omega_ - omega_c
        omega_err = self.saturate(omega_err, -self.omega_max_err, self.omega_max_err)

        x_err = np.block([[pos_err],
                          [vel_err],
                          [q_err],
                          [omega_err]])

        x_c = np.block([[pos_c],
                          [v_c],
                          [q_c],
                          [omega_c]])

        # x_err = np.block([[vel_err],
        #                   [q_err],
        #                   [omega_err]])

        # K_P = -inv(R) @ B.T @ P

        # print(K-K_P)
        u_err = -K@x_err
        # u2 = K@x_err

        u = u_err + u_c
        # u = np.zeros((4,1))
        # u[0,0] = self.gravity_ * self.mass_ / self.max_thrust_

        # print(u)
        # set_trace()
        self.J = x_err.T@self.Q@x_err + u_err.T@self.R@u_err
        return u #, x_c, self.J
