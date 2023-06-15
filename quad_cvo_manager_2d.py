import numpy as np
from numpy.linalg import norm
from numpy.random import normal as rndnorm

from gen_kalman_filter_2d import GenKalmanFilter
from cvo_gekko_2d import CVOGekko

from utils import rota

from IPython.core.debugger import set_trace

class CVOManager:
    def __init__(self, params, id, collision_range):
        # self.params = params
        # self.dt = params.dt
        self.id = id
        self.drop_prob_ = params.drop_prob
        self.add_kalman_ = params.add_kalman
        self.addUncertainty_ = params.add_uncertianty
        self.allKalFilters = {}

        self.cvoGekko = CVOGekko(params, id, collision_range)

        self.gps_pos_stdev = params.gps_pos_stdev
        self.gps_vel_stdev = params.gps_vel_stdev

    def get_kal_data(self, allKalStates, allKalCovariance):
        # print('get data ', self.id, " ", self.allKalFilters)
        # for quadKey in self.allKalFilters:
        # print(self.allKalFilters[1].P_)
        if 1 not in allKalStates:
            allKalStates[1] = []
            allKalCovariance[1] = []
        allKalStates[1].append(self.allKalFilters[1].xhat_.flatten().tolist())

        allKalCovariance[1].append(np.diag(self.allKalFilters[1].P_).tolist())

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
                    self.allKalFilters[quadKey].update(inX)

                xhat = self.allKalFilters[quadKey].xhat_
                P_mat = self.allKalFilters[quadKey].P_

                inRangePos.append(xhat[0:2])
                inRangeVel.append(xhat[2:4])

                ''' If/Else for uncertainty from Kalman filter '''
                if self.addUncertainty_:
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
