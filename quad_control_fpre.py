import numpy as np
from numpy.linalg import norm, inv, lstsq
import control
from copy import copy as cp
# from simple_pid import SimplePID

from utils import roll, pitch, yaw, exp, rota, rotp, from_euler, R_v_to_v1, R_v_to_b
from utils import skew , from_two_unit_vectors, boxminus

from commanded_state import CommandState

from IPython.core.debugger import set_trace

M_PI = 3.14159265359

class Controller:
    def __init__(self, params, id, dyn):
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

        self.P = np.eye(12)*0.01

        self.J = 0
        self.dyn = dyn

    def saturate(self, x, min, max):
        # if x > max:
        #     return max
        # if x < min:
        #     return min
        x = np.clip(x, min, max)
        return x

    def get_A_B(self, state):

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

        return A, B

    def get_commanded_state(self, state, t, vel_c):

        x_ = cp(state[0:3])
        v_ = cp(state[3:6])
        q_ = cp(state[6:10])
        omega_ = cp(state[10:13])

        ### Get Commanded States
        pos_c, v_c, q_c, omega_c, u_s, u_tau = \
                self.CommandState.get_commanded_state(cp(state), cp(t), cp(vel_c))

        u_c = np.block([[u_s],
                        [u_tau]])

        # pos_err = x_ - pos_c
        pos_err =  np.zeros((3,1))
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

        # set_trace()

        return x_err, x_c, u_c

    # def computeControl(self, state, t):
    def computeControl(self, state, dt, vel_c, t, psi_c = 0):
        if dt <= 1e-8:
            return


        A, B = self.get_A_B(cp(state))
        x_err, x_c, u_c = self.get_commanded_state(cp(state), cp(t), cp(vel_c))


        K = inv(self.R)@B.T@self.P
        # K = B.T@self.P
        u_err = -K@x_err
        # u2 = K@x_err

        # T = np.eye(12) - 2.0*x_err@x_err.T / (x_err.T@x_err)
        # A_h = -T@cp(A)@inv(T)
        # B_h = T@cp(B)
        # K = inv(self.R)@B_h.T@self.P
        # u_err = K@x_err

        u = u_err + u_c
        # print(u)

        _, self.P = self._rk4_step(cp(state), cp(self.P), cp(t), u)
        # u = np.zeros((4,1))
        # u[0,0] = self.gravity_ * self.mass_ / self.max_thrust_

        # print(u)
        nP = norm(self.P)
        # print(nP, t)
        # if nP > 1e6:
        #     self.P = self.P / nP
        self.J = x_err.T@self.Q@x_err + u_err.T@self.R@u_err

        # set_trace()

        return u #, x_c, self.J
        # return u, x_c, self.J



    def _rk4_step(self, state, P, t, u):
        # Integrate ODE using Runge-Kutta RK4 algorithm
        F1_x, F1_P = self._pdot(cp(state), cp(P), t, u)
        F2_x, F2_P = self._pdot(self.dyn.state_boxplus(cp(state), self.dt / 2.0 * F1_x),
                                cp(P) + self.dt / 2.0 * F1_P,
                                cp(t) + self.dt / 2.0,
                                u)
        F3_x, F3_P = self._pdot(self.dyn.state_boxplus(cp(state), self.dt / 2.0 * F2_x),
                                cp(P) + self.dt / 2.0 * F2_P,
                                cp(t) + self.dt / 2.0,
                                u)
        F4_x, F4_P = self._pdot(self.dyn.state_boxplus(cp(state), self.dt * F3_x),
                                cp(P) + self.dt * F3_P,
                                cp(t) + self.dt,
                                u)
        P = cp(P) + self.dt / 6 * (F1_P + 2 * F2_P + 2 * F3_P + F4_P)
        state = self.dyn.state_boxplus(cp(state), self.dt / 6 * (F1_x + 2 * F2_x + 2 * F3_x + F4_x))
        # print(np.max(np.abs(F1)))
        # set_trace()

        return state, P

    def _pdot(self, state, P, t, u):

        A, B = self.get_A_B(cp(state))
        # x_err, x_c, u_c = self.get_commanded_state(cp(state), cp(t))

        # K = -inv(self.R)@B.T@P
        # u = K@x
        Ac = A - B@inv(self.R)@B.T@P
        # iP = np.linalg.solve(P, np.eye(12))
        # # iP = inv(P)
        # S1 = ( iP@(Ac-Ac.T)+(Ac.T-Ac)@iP)
        # # E1 = 0.5*S1@P
        # # # E = Ac.T-Ac
        # S2 = ( (Ac-Ac.T)@P + P@(Ac.T-Ac) )
        # E2 = iP@(Ac-Ac.T)
        # # S2 = (Ac-Ac.T).T @ (Ac-Ac.T)
        # # E2 = -0.5 * S1 @ P
        # # E = (Ac-Ac.T)@P + P@(Ac.T-Ac)
        #
        E = np.zeros((12,12))
        # E = self.get_E(Ac, cp(x_err))
        Pdot = Ac.T@P + P@Ac + self.Q + P@B@inv(self.R)@B.T@P + E.T@P + P@E



        # T = np.eye(12) - 2.0*x_err@x_err.T / (x_err.T@x_err)
        # A_h = -T@A@inv(T)
        # B_h = T@B
        # # print(A_h, B_h)
        # # print(norm(A_h), norm(B_h))
        # # print(x_err)
        # # set_trace()
        # # K = inv(self.R)@B_h.T@P
        # # u = K@x
        # Pdot = -A_h.T@P - P@A_h + self.Q - P@B_h@inv(self.R)@B_h.T@P


        # # xdot = A@state + B@u
        # set_trace()
        xdot = self.dyn._fg(state, u)

        return xdot, Pdot


    def get_E(self, Ac, x_err):
        return np.zeros_like(Ac)

        n, _ = Ac.shape
        size_c = int((n**2-n)/2)
        size_d = n
        C = np.zeros((size_c, n**2))
        D = np.zeros((size_d, n**2))
        bc = np.zeros((size_c, 1))
        row = 0
        for i in range(0,n-1):
            for j in range(i+1,n):
                # print('i,j: ',i ,' ',j)
                C[row, i*n+j] = 1
                C[row, j*n+i] = -1
                bc[row, 0] = Ac[j,i] - Ac[i,j]
                row = row + 1
            D[i,i*n:i*n+n] = x_err.reshape((12,)) #np.ones(n)
        D[n-1,(n-1)*n:(n-1)*n+n] = x_err.reshape((12,)) #np.ones(n)



        A = np.block([[np.eye(n**2), C.T, D.T],
                      [C, np.zeros((size_c,size_c)), np.zeros((size_c,size_d))],
                      [D, np.zeros((size_d,size_c)), np.zeros((size_d,size_d))]])

        b = np.block([[np.zeros((n**2,1))],
                      [bc],
                      [np.zeros((size_d,1))]])

        x_E = inv(A)@b
        # Solve Ax=b with Tikhonov regularization
        # alpha = 1e-10
        # x_E, residuals, rank, s = lstsq(A.T.dot(A) + \
        #     alpha * np.eye(A.shape[1]), A.T.dot(b), rcond=None)

        E = x_E[0:n**2].reshape((n,n))
        # set_trace()

        return E
