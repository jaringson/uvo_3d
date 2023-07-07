import numpy as np
from numpy.linalg import norm
from numpy.random import normal as rndnorm

from gen_kalman_filter import GenKalmanFilter
from cvo_gekko import CVOGekko

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
        self.radar_azimuth_stdev_ = params.radar_azimuth_stdev
        self.radar_azimuthDot_stdev_ = params.radar_azimuthDot_stdev

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
        x = p[0,0] - self.radarPos_[0,0]
        y = p[1,0] - self.radarPos_[1,0]
        z = p[2,0] - self.radarPos_[2,0]
        xDot = v[0,0]
        yDot = v[1,0]
        zDot = v[2,0]

        range = np.sqrt(x**2 + y**2 + z**2)
        rangeHat = range + rndnorm(0, self.radar_range_stdev_)
        zenith = np.arctan2(y, x)
        zenithHat = zenith + rndnorm(0, self.radar_zenith_stdev_)

        flatRange = np.sqrt(x**2 + y**2)
        flatRangeDot = (x*xDot + y*yDot)/ flatRange
        azimuth = np.arctan2(z,flatRange)
        azimuthHat = azimuth + rndnorm(0, self.radar_azimuth_stdev_)


        # range = np.sqrt(mav_x**2 + mav_y**2)
        # zenith = np.arctan2(mav_y, mav_x)

        p_hat = np.array([
                        [rangeHat*np.cos(zenithHat)*np.cos(azimuthHat)],
                        [rangeHat*np.sin(zenithHat)*np.cos(azimuthHat)],
                        [rangeHat*np.sin(azimuthHat)]
                        ])

        rangeDot = (x*xDot + y*yDot + z*zDot) / range
        rangeDotHat = rangeDot + rndnorm(0, self.radar_rangeDot_stdev_)
        zenithDot = (x*yDot - y*xDot) / (x**2+y**2)
        zenithDotHat = zenithDot + rndnorm(0, self.radar_zenithDot_stdev_)
        azimuthDot = (flatRange*zDot - z*flatRangeDot) / (z**2+flatRange**2)
        azimuthDotHat = azimuthDot + rndnorm(0, self.radar_azimuthDot_stdev_)

        v_hat = np.array([
                        [rangeDotHat*np.cos(zenithHat)*np.cos(azimuthHat)
                            -rangeHat*zenithDotHat*np.sin(zenithHat)*np.cos(azimuthHat)
                            -rangeHat*azimuthDotHat*np.cos(zenithHat)*np.sin(azimuthHat)],
                        [rangeDotHat*np.sin(zenithHat)*np.cos(azimuthHat)
                            +rangeHat*zenithDotHat*np.cos(zenithHat)*np.cos(azimuthHat)
                            -rangeHat*azimuthDotHat*np.sin(zenithHat)*np.sin(azimuthHat)],
                        [rangeDotHat*np.sin(azimuthHat)
                            +rangeHat*azimuthDotHat*np.cos(azimuthHat)]
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


            if self.add_kalman_:

                ''' Checks if Kalman filter is started and
                    if other quad is in range.  '''
                if quadKey not in self.allKalFilters and \
                        norm(av2Pos - av1Pos) > self.cvoGekko.collisionRange:
                    continue

                if quadKey not in self.allKalFilters and \
                        norm(av2Pos - av1Pos) < self.cvoGekko.collisionRange:
                    ''' Initialize Kalman filter '''
                    inXHat = np.block([[av2Pos], [av2Vel], [np.zeros((6,1))]])
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
                # xhat[0:3] = xhat[0:3] + self.radarPos_
                P_mat = self.allKalFilters[quadKey].P_
                # if quadKey == 1:
                #     print(xhat)
                #     print(allQuads[quadKey].x_)
                #     print(av2Pos)
                #     # print(rota(allQuads[quadKey].q_, allQuads[quadKey].v_))
                #     # print(av2Vel)
                #     set_trace()

                inRangePos.append(xhat[0:3])
                inRangeVel.append(xhat[3:6])

                ''' If/Else for uncertainty from Kalman filter '''
                if self.addUncertainty_:
                    uncertPos.append( [3.0*P_mat[0,0]**0.5, 3.0*P_mat[1,1]**0.5, 3.0*P_mat[2,2]**0.5] )
                    uncertVel.append( [3.0*P_mat[3,3]**0.5, 3.0*P_mat[4,4]**0.5, 3.0*P_mat[5,5]**0.5] )

                else:
                    uncertPos.append( [0,0,0] )
                    uncertVel.append( [0,0,0] )

            else:

                if norm(av2Pos - av1Pos) < self.cvoGekko.collisionRange:
                    inRangePos.append(av2Pos)
                    inRangeVel.append(av2Vel)
                    uncertPos.append( [0,0,0] )
                    uncertVel.append( [0,0,0] )

        ''' Run optimzation '''
        vel = self.cvoGekko.get_best_vel(av1Pos, av1Vel, vel_d,
                        inRangePos, inRangeVel,
                        uncertPos, uncertVel)

        return np.asarray([vel]).T
