import numpy as np
from numpy.linalg import norm
from numpy.random import normal as rndnorm

from gen_kalman_filter_2d import GenKalmanFilter
from cvo_gekko_2d import CVOGekko

from utils import rota

from IPython.core.debugger import set_trace

class CVOManager:
    def __init__(self, params, id, collision_range, max_vel):
        # self.params = params
        # self.dt = params.dt
        self.id = id
        self.drop_prob_ = params.drop_prob
        self.add_kalman_ = params.add_kalman
        self.addUncertainty_ = params.add_uncertianty
        self.allKalFilters = {}

        self.cvoGekko = CVOGekko(params, id, collision_range, max_vel)

        self.gps_pos_stdev = params.gps_pos_stdev
        self.gps_vel_stdev = params.gps_vel_stdev

        self.radar_range_stdev_ = params.radar_range_stdev
        self.radar_rangeDot_stdev_ = params.radar_rangeDot_stdev
        self.radar_zenith_stdev_ = params.radar_zenith_stdev
        self.radar_zenithDot_stdev_ = params.radar_zenithDot_stdev

        self.radarPos_ = params.radarPos

        self.radar_meas_ = params.radar_measurement

        self.con_vel_uncertain_ = params.con_vel_uncertain

    def get_kal_data(self, allKalStates, allKalCovariance):
        # print('get data ', self.id, " ", self.allKalFilters)
        # for quadKey in self.allKalFilters:
        # print(self.allKalFilters[1].P_)
        if 1 not in allKalStates:
            allKalStates[1] = []
            allKalCovariance[1] = []

        append_val = self.allKalFilters[1].xhat_
        allKalStates[1].append(append_val.flatten().tolist())

        allKalCovariance[1].append(np.diag(self.allKalFilters[1].P_).tolist())

    def get_radar_uncertainty(self, p, v):
        mav_x = p[0] - self.radarPos_[0]
        mav_y = p[1] - self.radarPos_[1]

        range = np.sqrt(mav_x**2 + mav_y**2)
        zenith = np.arctan2(mav_y, mav_x)
        rangeHat = range + rndnorm(0, self.radar_range_stdev_)
        zenithHat = zenith + rndnorm(0, self.radar_zenith_stdev_)

        p_hat = np.array([
                        [(rangeHat*np.cos(zenithHat))[0]],
                        [(rangeHat*np.sin(zenithHat))[0]],
                        [0.0]
                        ])

        # rangeDot = (mav_x * v[0] + mav_y*v[1]) / np.sqrt(mav_x**2+mav_y**2)
        # zenithDot = (mav_x * v[1] - mav_y*v[0]) / (mav_x**2+mav_y**2)
        rangeDot = (mav_x * v[0] + mav_y*v[1]) / np.sqrt(mav_x**2+mav_y**2)
        zenithDot = (mav_x * v[1] - mav_y*v[0]) / (mav_x**2+mav_y**2)
        rangeDotHat = rangeDot + rndnorm(0, self.radar_rangeDot_stdev_)
        zenithDotHat = zenithDot + rndnorm(0, self.radar_zenithDot_stdev_)

        v_hat = np.array([
                        [(rangeDotHat*np.cos(zenithHat)-rangeHat*zenithDotHat*np.sin(zenithHat))[0]],
                        [(rangeDotHat*np.sin(zenithHat)+rangeHat*zenithDotHat*np.cos(zenithHat))[0]],
                        [0]
                        ])

        # print(p, p_hat)
        # print(v, v_hat)
        # set_trace()
        return p_hat + self.radarPos_, v_hat

    def propagate(self):
        for quadKey in self.allKalFilters:
            self.allKalFilters[quadKey].predict()


    def get_best_vel(self, allQuads, t, vel_d):
        av1Pos = allQuads[self.id].x_
        av1Vel = rota(allQuads[self.id].q_, allQuads[self.id].v_)

        av1Pos = av1Pos[0:2]
        av1Vel = av1Vel[0:2]

        vel_d = vel_d[0:2]

        inRangePos = []
        inRangeVel = []
        uncertPos = []
        uncertVel = []

        ''' Loop over all other quadrotors to get position and velocity '''
        for quadKey in allQuads:

            ''' Don't include self '''
            if quadKey == self.id:
                continue

            ''' Add uncertainty to the data '''
            av2Pos = allQuads[quadKey].x_ + rndnorm(0, self.gps_pos_stdev, size=(3,1))
            av2Vel = rota(allQuads[quadKey].q_, allQuads[quadKey].v_) + rndnorm(0, self.gps_vel_stdev, size=(3,1))
            if self.radar_meas_:
                av2Pos, av2Vel = self.get_radar_uncertainty(allQuads[quadKey].x_, rota(allQuads[quadKey].q_, allQuads[quadKey].v_))
            # directionalV = np.block([[rndnorm(0, self.gps_vel_stdev, size=(1,1))],[0], [0]])
            # av2Vel = rota(allQuads[quadKey].q_, allQuads[quadKey].v_) + rota(allQuads[quadKey].q_, directionalV)


            av2Pos = av2Pos[0:2]
            av2Vel = av2Vel[0:2]

            if self.add_kalman_:

                ''' Checks if Kalman filter is started and
                    if other quad is in range.  '''
                if quadKey not in self.allKalFilters and \
                        norm(av2Pos - av1Pos) > self.cvoGekko.collisionRange:
                    continue

                if quadKey not in self.allKalFilters and \
                        norm(av2Pos - av1Pos) < self.cvoGekko.collisionRange:
                    ''' Initialize Kalman filter '''
                    inXHat = np.block([[av2Pos], [av2Vel], [np.zeros((4,1))]])
                    self.allKalFilters[quadKey] = GenKalmanFilter(self.id, quadKey, inXHat)

                elif quadKey in self.allKalFilters and \
                    norm(av2Pos - av1Pos) > self.cvoGekko.collisionRange:
                    ''' If the other vehicle is moved out of range delete Kalman
                        filter associated with it '''
                    del self.allKalFilters[quadKey]
                    continue

                elif np.random.random() < self.drop_prob_:
                    ''' Update Kalman filter if communication isn't dropped '''
                    inX = np.block([[av2Pos], [av2Vel]])
                    if self.radar_meas_:
                        self.allKalFilters[quadKey].update_radar(inX)
                    else:
                        self.allKalFilters[quadKey].update(inX)

                xhat = self.allKalFilters[quadKey].xhat_
                P_mat = self.allKalFilters[quadKey].P_
                # if quadKey == 1:
                #     print(xhat)
                #     print(allQuads[quadKey].x_)
                #     print(av2Pos)
                #     set_trace()

                inRangePos.append(xhat[0:2])
                inRangeVel.append(xhat[2:4])

                ''' If/Else for uncertainty from Kalman filter '''
                if self.addUncertainty_:
                    # print(P_mat[0,0], P_mat[1,1], P_mat[2,2], P_mat[3,3])
                    uncertPos.append( [3.0*P_mat[0,0]**0.5, 3.0*P_mat[1,1]**0.5] )
                    uncertVel.append( [3.0*P_mat[2,2]**0.5, 3.0*P_mat[3,3]**0.5] )

                else:
                    uncertPos.append( [0,0] )
                    uncertVel.append( [0,0] )

            else:

                if norm(av2Pos - av1Pos) < self.cvoGekko.collisionRange:
                    inRangePos.append(av2Pos)
                    inRangeVel.append(av2Vel)
                    uncertPos.append( [0,0] )
                    uncertVel.append( [0,0] )

        ''' Run optimzation '''
        vel = self.cvoGekko.get_best_vel(av1Pos, av1Vel, vel_d,
                        inRangePos, inRangeVel,
                        uncertPos, uncertVel)

        vel = np.block([[np.asarray([vel]).T],
                        [0.0]])

        return vel
