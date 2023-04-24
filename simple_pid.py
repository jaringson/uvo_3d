import math
import numpy as np

class SimplePID:
    def __init__(self, kp, ki, kd, max, min, tau):
        self.kp_ = kp
        self.ki_ = ki
        self.kd_ = kd
        self.integrator_ = 0
        self.differentiator_ = 0
        self.last_error_ = 0
        self.last_state_ = 0
        self.tau_ = tau
        self.max_ = max
        self.min_ = min

    def saturate(self, u):
        ret_u = u
        if u>self.max_:
            ret_u = self.max_
        if u<self.min_:
            ret_u = self.min_
        return ret_u

    def computePID(self, desired, current, dt, x_dot=np.Inf):
        error = desired - current

        if dt < 1e-5 or abs(error) > 1e8:
            return 0.0

        if dt > 1.0:
            dt = 0.0
            self.differentiator_ = 0.0

        p_term = error*self.kp_
        i_term = 0.0
        d_term = 0.0

        ''' Derivative Term '''
        if self.kd_ > 0.0:
            if np.isfinite(x_dot):
                d_term = self.kd_ * x_dot
            else:
                self.differentiator_ = (2.0 * self.tau_ - dt) / (2.0 * self.tau_ + dt) * \
                                        self.differentiator_ + 2.0 / (2.0 * self.tau_ + dt) * \
                                        (current - self.last_state_)
                d_term = self.kd_ * self.differentiator_

        ''' Integrator Term '''
        if self.ki_ > 0.0:
            self.integrator_ += dt / 2.0 * (error + self.last_error_)
            i_term = self.ki_ * self.integrator_

        self.last_error_ = error
        self.last_state_ = current

        u = p_term + i_term - d_term

        ''' Integrator Windup '''
        u_sat = self.saturate(u)
        if u != u_sat and abs(i_term) > abs(u - p_term + d_term):
            self.integrator_ = (u_sat - p_term + d_term) / self.ki_

        return u_sat
