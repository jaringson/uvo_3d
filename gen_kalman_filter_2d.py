import numpy as np
from numpy.linalg import inv

from IPython.core.debugger import set_trace
import params

class GenKalmanFilter:
    def __init__(self, owner_id, id, x):

        self.sigmaQ_vel_ = params.sigmaQ_vel
        self.alphaQ_vel_ = params.alphaQ_vel
        self.sigmaQ_jrk_ = params.sigmaQ_jrk
        self.alphaQ_jrk_ = params.alphaQ_jrk

        self.sigmaR_pos_ = params.sigmaR_pos
        self.sigmaR_vel_ = params.sigmaR_vel
        # self.sigmaR_range_ = params.sigmaR_range
        # self.sigmaR_zenith_ = params.sigmaR_zenith

        self.id_ = id


        self.meas_dim_ = 2 #Hardcoded for now

        ''' constant jerk '''
        self.n_ = self.meas_dim_*4 # xhat = [pos vel acc jerk]^T

        ''' constant velocity '''
        # self.n_ = self.meas_dim_*2 # xhat = [pos vel]^T

        self.dt_ = params.dt

        ''' constant jerk '''
        # self.mmtype_ mmtype

        self.xhat_ = x[0:self.n_]

        self.A_ = self.build_A()
        self.Q_ = self.build_Q()

        self.C_pos_ = self.build_C(2, self.n_)
        self.C_vel_ = self.build_C(4, self.n_)
        self.R_pos_ = self.build_R(2)
        self.R_vel_ = self.build_R(4)
        self.P_ = np.eye(self.n_)*0.1
        self.P_[0,0] = 1
        self.P_[1,1] = 1
        self.P_[2,2] = 0.5
        self.P_[3,3] = 0.5

    def predict(self):
        self.xhat_ = self.A_ @ self.xhat_
        self.P_ = self.A_ @ self.P_ @ self.A_.T + self.Q_

    def update(self, measurement, hasVel=True):
        C = []
        R = []
        if hasVel:
            R = self.R_vel_
            C = self.C_vel_
        else:
            R = self.R_pos_
            C = self.C_pos_

        K = self.P_ @ C.T @ inv(C@self.P_@C.T + R)
        self.xhat_ = self.xhat_ + K@(measurement -C@self.xhat_)
        eye_n = np.eye(self.n_)

        self.P_ = (eye_n - K@C)@self.P_
        # set_trace()

    def update_radar(self):
        throw('Not Implemented')

    def build_A(self):
        A = np.zeros((self.n_, self.n_))
        for i in range(self.n_):
            A[i,i] = 1
            for j in range(1,int(self.n_/self.meas_dim_)):
                if i+(j*self.meas_dim_) < (self.n_):
                    A[i, i+(j*self.meas_dim_)] = self.dt_**j / j

        return A

    def build_Q(self):

        Q = np.array([
            [self.dt_**7/252, 0, self.dt_**6/72, 0, self.dt_**5/30, 0, self.dt_**4/24, 0],
            [0, self.dt_**7/252, 0, self.dt_**6/72, 0, self.dt_**5/30, 0, self.dt_**4/24],
            [self.dt_**6/72, 0, self.dt_**5/20, 0, self.dt_**4/8, 0, self.dt_**3/6, 0],
            [0, self.dt_**6/72, 0, self.dt_**5/20, 0, self.dt_**4/8, 0, self.dt_**3/6],
            [self.dt_**5/30, 0, self.dt_**4/8, 0, self.dt_**3/3, 0, self.dt_**2/2, 0],
            [0, self.dt_**5/30, 0, self.dt_**4/8, 0, self.dt_**3/3, 0, self.dt_**2/2],
            [self.dt_**4/24, 0, self.dt_**3/6, 0, self.dt_**2/2, 0, self.dt_**1/1, 0],
            [0, self.dt_**4/24, 0, self.dt_**3/6, 0, self.dt_**2/2, 0, self.dt_**1/1]
            ])

        Q = 2.0 * self.alphaQ_jrk_ * self.sigmaQ_jrk_**2 * Q;

        # Q = np.array([
        #             [(self.dt_**4)/4, 0, (self.dt_**3)/2, 0],
        #             [0, (self.dt_**4)/4, 0, (self.dt_**3)/2],
        #             [(self.dt_**3)/2, 0, (self.dt_**2)/1, 0],
        #             [0, (self.dt_**3)/2, 0, (self.dt_**2)/1]
        #             ])
        #
        # Q = 2.0 * self.alphaQ_vel_ * self.sigmaQ_vel_**2 * Q;

        return Q

    def build_C(self, m_request, n_request):
        ret_C = np.eye(m_request)
        if n_request > m_request:
            ret_C = np.eye(n_request)
        return ret_C[0 : m_request, 0 : n_request]

    def build_R(self, m_request):

        m_vel = self.meas_dim_ * 2

        Im_pos = np.eye(m_vel)
        Im_pos[2,2] = 0
        Im_pos[3,3] = 0

        Im_vel = np.eye(m_vel)
        Im_vel[0,0] = 0
        Im_vel[1,1] = 0

        R_whole = Im_pos * self.sigmaR_pos_**2 + Im_vel * self.sigmaR_vel_**2

        return R_whole[0:m_request,0:m_request]
