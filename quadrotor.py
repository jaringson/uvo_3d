import numpy as np
from numpy.linalg import inv, norm
from utils import rota, rotp, boxplus

import copy

from IPython.core.debugger import set_trace


class QuadDynamics:
    def __init__(self, params, id, x0, wp1):
        self.id = id
        self._Ts = params.dt
        self.e3_ = np.array([[0.0], [0.0], [1.0]])
        self.gravity_ = params.gravity
        self.max_thrust_ = params.max_thrust
        self.mass_ = params.mass
        self.linear_drag_ = params.linear_drag
        self.angular_drag_ = params.angular_drag
        self.inertia_matrix_ = params.inertia_matrix
        self.inertia_inv_ = inv(self.inertia_matrix_)

        self.x_ = x0
        # waypoint1 = np.array([wp1[0:x0.shape[0]]])
        # dir = (waypoint1.T-x0) / norm(waypoint1.T-x0)
        # self.v_ = dir * np.random.uniform(0,1.0)
        self.v_ = params.v0
        self.q_ = params.q0
        self.omega_ = params.omega0

        self.state = np.block([[self.x_],
                                [self.v_],
                                [self.q_],
                                [self.omega_]])
    def update(self, u):
        # This is the external method that takes the input u at time
        # t and returns the output y at time t.
        self._rk4_step(u)  # propagate the state by one time sample
        return self.state

    def _rk1_step(self, u):
        # Integrate ODE using Runge-Kutta RK1 algorithm
        self.state += self._Ts * self._f(self.state, u)

    def _rk2_step(self, u):
        # Integrate ODE using Runge-Kutta RK2 algorithm
        F1 = self._f(self.state, u)
        F2 = self._f(self.state + self._Ts / 2 * F1, u)
        self.state += self._Ts / 2 * (F1 + F2)

    def _rk4_step(self, u):
        # Integrate ODE using Runge-Kutta RK4 algorithm
        F1 = self._fg(copy.copy(self.state), u)
        F2 = self._fg(self.state_boxplus(copy.copy(self.state), self._Ts / 2 * F1), u)
        F3 = self._fg(self.state_boxplus(copy.copy(self.state), self._Ts / 2 * F2), u)
        F4 = self._fg(self.state_boxplus(copy.copy(self.state), self._Ts * F3), u)
        self.state = self.state_boxplus(copy.copy(self.state), self._Ts / 6 * (F1 + 2 * F2 + 2 * F3 + F4))

        self.x_ = copy.copy(self.state[0:3])
        self.v_ = copy.copy(self.state[3:6])
        self.q_ = copy.copy(self.state[6:10])
        self.omega_ = copy.copy(self.state[10:13])
        # print('state6: ', self.state)

    def _fg(self, state, u):
        xdot = np.zeros((12,1))

        taus = u[0:3]
        thrust = u[3]

        x = copy.copy(state[0:3])
        v = copy.copy(state[3:6])
        q = copy.copy(state[6:10])
        omega = copy.copy(state[10:13])

        xdot[0:3] = rota(q, v)
        xdot[3:6] = -self.e3_ * thrust * self.max_thrust_ / self.mass_  - self.linear_drag_ @ v + \
                self.gravity_ * rotp(q, self.e3_) - np.cross(omega.T, v.T).T
        xdot[6:9] = omega
        xdot[9:12] = self.inertia_inv_ @ (taus - np.cross(omega.T, (self.inertia_matrix_ @ omega).T).T
            - self.angular_drag_ @ omega)

        # print('dyn taus: ', taus)
        # set_trace()
        return xdot

    def state_boxplus(self, state, state_delta):
        state[0:3] += state_delta[0:3]
        state[3:6] += state_delta[3:6]
        state[6:10] = boxplus(state[6:10], state_delta[6:9])
        state[10:13] += state_delta[9:12]


        # print(state - self.state)
        # print('state: ', state)
        # set_trace()

        return state
