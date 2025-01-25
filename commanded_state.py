import numpy as np
from numpy.linalg import norm, inv
from copy import copy as cp
from utils import roll, pitch, yaw, exp, rota, rotp, from_euler, R_v_to_v1, R_v_to_b
from utils import skew, boxminus, from_two_unit_vectors

from IPython.core.debugger import set_trace

class CommandState:
    def __init__(self, params):

        self.gravity_ = params.gravity
        self.e3 = params.e3
        self.dt = params.dt

        self.max_thrust_ = params.max_thrust
        self.mass_ = params.mass
        self.inertia_matrix_ = params.inertia_matrix


        self.angular_drag_ = params.angular_drag
        # self.linear_drag_ = params.linear_drag



    def get_commanded_state(self, state, t, vel_c):

        x_ = cp(state[0:3])
        v_ = cp(state[3:6])
        q_ = cp(state[6:10])
        omega_ = cp(state[10:13])


        pos_c = vel_c * self.dt + cp(x_)
        # pos_c = np.zeros((3,1))

        # pddot_I = self.get_pddot8(t)
        # a_I = self.gravity_*self.e3 - pddot_I
        # quat_c = np.array([[1.0],[0.0],[0.0],[0.0]])

        v_c = rotp(q_, vel_c)

        acc_c = self.gravity_*self.e3 - (vel_c - rota(q_,v_)) / (self.dt)
        # acc_c = (rota(q_,v_) - vel_c) / self.dt + self.gravity_*self.e3
        # quat_c = from_two_unit_vectors(self.e3, acc_c/norm(acc_c))
        quat_c = cp(q_)
        # v_c = np.zeros((3,1))

        # omega_c = boxminus(quat_c, q_) / self.dt
        # omega_c = boxminus(q_, quat_c) / self.dt
        # omega_c = np.zeros((3,1))
        omega_c = cp(omega_)

        ## ----------------------------------- ##
        # u_s = norm(a_I) * self.mass_ / self.max_thrust_
        u_s = acc_c[2,0] * self.mass_ / self.max_thrust_
        # u_s =

        u_tau = self.angular_drag_@omega_c + skew(omega_c)*self.inertia_matrix_@omega_c
        # u_tau = self.inertia_matrix_@omega_c + skew(omega_c)*self.inertia_matrix_@omega_c

        # print((vel_c - rota(q_,v_)) / (self.dt))
        # print(roll(quat_c), pitch(quat_c), yaw(quat_c))
        # print(u_s, u_tau)
        # set_trace()

        return pos_c, v_c, quat_c, omega_c, u_s, u_tau
