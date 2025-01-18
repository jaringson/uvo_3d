import numpy as np
from numpy.linalg import inv
import auto_diff
import copy

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
        self.sigmaR_range_ = params.sigmaR_range
        self.sigmaR_zenith_ = params.sigmaR_zenith
        self.sigmaR_azimuth_ = params.sigmaR_azimuth
        self.sigmaR_rangeDot_ = params.sigmaR_rangeDot
        self.sigmaR_zenithDot_ = params.sigmaR_zenithDot
        self.sigmaR_azimuthDot_ = params.sigmaR_azimuthDot

        self.radarPos = params.radarPos

        self.id_ = id


        self.meas_dim_ = 3 #Hardcoded for now

        ''' constant jerk '''
        self.n_ = self.meas_dim_*4 # xhat = [pos vel acc jerk]^T

        ''' constant velocity '''
        # self.n_ = self.meas_dim_*2 # xhat = [pos vel]^T

        self.dt_ = params.dt

        ''' constant jerk '''
        # self.mmtype_ mmtype

        self.xhat_ = x

        self.A_ = self.build_A()
        self.Q_ = self.build_Q()

        self.C_pos_ = self.build_C(3, self.n_)
        self.C_vel_ = self.build_C(6, self.n_)
        self.R_pos_ = self.build_R(3)
        self.R_vel_ = self.build_R(6)
        self.P_ = np.eye(self.n_)*0.1
        self.P_[0,0] = 1
        self.P_[1,1] = 1
        self.P_[2,2] = 1
        self.P_[3,3] = 1
        self.P_[4,4] = 1
        self.P_[5,5] = 1

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

        self.P_ = (eye_n - K@C)@self.P_@(eye_n - K@C).T + K@R@K.T

    def get_z(self, xIn, constant=1):


        x = xIn[0,0] - self.radarPos[0,0]
        y = xIn[1,0] - self.radarPos[1,0]
        z = xIn[2,0] - self.radarPos[2,0]
        xDot = xIn[3,0]
        yDot = xIn[4,0]
        zDot = xIn[5,0]

        range = np.sqrt(x**2 + y**2 + z**2)
        zenith = np.arctan(y/x)
        if x < 0 and y > 0:
            zenith = zenith + np.pi
        elif x < 0 and y < 0:
            zenith = zenith - np.pi

        flatRange = np.sqrt(x**2 + y**2)
        flatRangeDot = (x*xDot + y*yDot)/ flatRange
        azimuth = np.arctan(z/flatRange)

        rangeDot = (x*xDot + y*yDot + z*zDot) / range
        zenithDot = (x*yDot - y*xDot) / (x**2+y**2)
        azimuthDot = (flatRange*zDot - z*flatRangeDot) / (z**2+flatRange**2)

        return np.array([[range],[zenith],[azimuth],
                    [rangeDot],[zenithDot],[azimuthDot]])

        # return np.array([[range],[zenith],[azimuth]])

    def update_radar(self, measurement):

        R = self.build_polar_R()
        z, H = self.build_polar_H()
        C = self.C_vel_
        # C = self.C_pos_

        K = self.P_ @ H.T @ inv(H@self.P_@H.T + R)
        # self.xhat_ = self.xhat_ + K@(measurement -C@self.xhat_)
        self.xhat_ = self.xhat_ + K@(self.get_z(measurement) - z)
        eye_n = np.eye(self.n_)

        self.P_ = (eye_n - K@H)@self.P_ #@(eye_n - K@H).T + K@R@K.T

        # set_trace()


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
            [self.dt_**7/252, 0, 0, self.dt_**6/72, 0, 0, self.dt_**5/30, 0, 0, self.dt_**4/24, 0, 0],
            [0, self.dt_**7/252, 0, 0, self.dt_**6/72, 0, 0, self.dt_**5/30, 0, 0, self.dt_**4/24, 0],
            [0, 0, self.dt_**7/252, 0, 0, self.dt_**6/72, 0, 0, self.dt_**5/30, 0, 0, self.dt_**4/24],
            [self.dt_**6/72, 0, 0, self.dt_**5/20, 0, 0, self.dt_**4/8, 0, 0, self.dt_**3/6, 0, 0],
            [0, self.dt_**6/72, 0, 0, self.dt_**5/20, 0, 0, self.dt_**4/8, 0, 0, self.dt_**3/6, 0],
            [0, 0, self.dt_**6/72, 0, 0, self.dt_**5/20, 0, 0, self.dt_**4/8, 0, 0, self.dt_**3/6],
            [self.dt_**5/30, 0, 0, self.dt_**4/8, 0, 0, self.dt_**3/3, 0, 0, self.dt_**2/2, 0, 0],
            [0, self.dt_**5/30, 0, 0, self.dt_**4/8, 0, 0, self.dt_**3/3, 0, 0, self.dt_**2/2, 0],
            [0, 0, self.dt_**5/30, 0, 0, self.dt_**4/8, 0, 0, self.dt_**3/3, 0, 0, self.dt_**2/2],
            [self.dt_**4/24, 0, 0, self.dt_**3/6, 0, 0, self.dt_**2/2, 0, 0, self.dt_**1/1, 0, 0],
            [0, self.dt_**4/24, 0, 0, self.dt_**3/6, 0, 0, self.dt_**2/2, 0, 0, self.dt_**1/1, 0],
            [0, 0, self.dt_**4/24, 0, 0, self.dt_**3/6, 0, 0, self.dt_**2/2, 0, 0, self.dt_**1/1]
            ])

        Q = 2.0 * self.alphaQ_jrk_ * self.sigmaQ_jrk_**2 * Q

        return Q

    def build_C(self, m_request, n_request):
        ret_C = np.eye(m_request)
        if n_request > m_request:
            ret_C = np.eye(n_request)
        return ret_C[0 : m_request, 0 : n_request]

    def build_R(self, m_request):

        m_vel = self.meas_dim_ * 2

        Im_pos = np.eye(m_vel)
        Im_pos[3,3] = 0
        Im_pos[4,4] = 0
        Im_pos[5,5] = 0

        Im_vel = np.eye(m_vel)
        Im_vel[0,0] = 0
        Im_vel[1,1] = 0
        Im_vel[2,2] = 0

        R_whole = Im_pos * self.sigmaR_pos_**2 + Im_vel * self.sigmaR_vel_**2

        return R_whole[0:m_request,0:m_request]

    def build_polar_R(self):
        R = np.diag([self.sigmaR_range_**2,
                        self.sigmaR_zenith_**2,
                        self.sigmaR_azimuth_**2,
                        self.sigmaR_rangeDot_**2,
                        self.sigmaR_zenithDot_**2,
                        self.sigmaR_azimuthDot_**2])

        # R = np.diag([self.sigmaR_range_**2,
        #                 self.sigmaR_zenith_**2,
        #                 self.sigmaR_azimuth_**2])
        # set_trace()
        return R

    # def f_auto(x_arr):
    #     x = x_arr[0,0]
    #     y = x_arr[1,0]
    #     z = x_arr[2,0]
    #     xDot = x_arr[3,0]
    #     yDot = x_arr[4,0]
    #     zDot = x_arr[5,0]
    #
    #     flatRange = np.sqrt(x**2 + y**2)
    #     flatRangeDot = (x*xDot + y*yDot)/ flatRange
    #
    #     out = np.zeros((6,1))
    #     out[0,0] = np.sqrt(x**2+y**2)
    #     out[1,0] = np.arctan(y / x)
    #     out[2,0] = np.arctan(z,flatRange)
    #     out[3,0] = (x*xDot + y*yDot) / np.sqrt(x**2+y**2)
    #     out[4,0] = (x*yDot - y*xDot) / (x**2+y**2)
    #     out[5,0] = (flatRange*zDot - z*flatRangeDot) / (z**2+flatRange**2)
    #
    #     return out


    def build_polar_H(self):

        # x = self.xhat_[0,0]
        # y = self.xhat_[1,0]
        # z = self.xhat_[2,0]
        # xDot = self.xhat_[3,0]
        # yDot = self.xhat_[4,0]
        # zDot = self.xhat_[5,0]

        x_arr = copy.copy(self.xhat_) #np.array([[x],[y],[z],[xDot],[yDot],[zDot]])

        Hx = None
        zHat = None
        with auto_diff.AutoDiff(x_arr) as (x_arr):
            f_eval = self.get_z(x_arr)
            zHat, Hx = auto_diff.get_value_and_jacobian(f_eval)
            # print(Hx)
        # H = np.block([[Hx,np.zeros((Hx.shape[0],self.meas_dim_*2))]])
        # set_trace()

        return zHat, Hx
