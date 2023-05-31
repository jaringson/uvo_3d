import numpy as np
from numpy.linalg import norm
from numpy.random import normal as rndnorm

from gen_kalman_filter import GenKalmanFilter
from cvo_gekko import CVOGekko

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

    def get_kal_data(self, allKalStates):
        # print('get data ', self.id, " ", self.allKalFilters)
        # set_trace()
        for quadKey in self.allKalFilters:
            if quadKey not in allKalStates:
                allKalStates[quadKey] = []
            allKalStates[quadKey].append(self.allKalFilters[quadKey].xhat_.flatten().tolist())

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

        for quadKey in allQuads:
            if quadKey == self.id:
                continue



            av2Pos = allQuads[quadKey].x_ # + rndnorm(0, self.gps_pos_stdev, size=(3,1))
            av2Vel = rota(allQuads[quadKey].q_, allQuads[quadKey].v_) #+ rndnorm(0, self.gps_vel_stdev, size=(3,1))

            if self.add_kalman_:

                # print('cond1: ', quadKey not in self.allKalFilters)
                # print('cond2: ', norm(av2Pos - av1Pos) > self.cvoGekko.collisionRange)
                # print(self.cvoGekko.collisionRange)
                # print('together: ', quadKey not in self.allKalFilters and \
                # norm(av2Pos - av1Pos) > self.cvoGekko.collisionRange)

                if quadKey not in self.allKalFilters and \
                        norm(av2Pos - av1Pos) > self.cvoGekko.collisionRange:
                    # print('here: ', self.id)
                    continue


                if quadKey not in self.allKalFilters and \
                        norm(av2Pos - av1Pos) < self.cvoGekko.collisionRange:

                    inXHat = np.block([[av2Pos], [av2Vel], [np.zeros((6,1))]])

                    self.allKalFilters[quadKey] = GenKalmanFilter(self.id, quadKey, inXHat)


                elif quadKey in self.allKalFilters and \
                    norm(av2Pos - av1Pos) > self.cvoGekko.collisionRange:
                    del self.allKalFilters[quadKey]
                    continue


                elif np.random.random() < self.drop_prob_:
                    inX = np.block([[av2Pos], [av2Vel]])
                    # set_trace()
                    self.allKalFilters[quadKey].update(inX)

                xhat = self.allKalFilters[quadKey].xhat_
                P_mat = self.allKalFilters[quadKey].P_

                # print('x: ', xhat[0:3] - allQuads[quadKey].x_)

                inRangePos.append(xhat[0:3])
                inRangeVel.append(xhat[3:6])

                if self.addUncertainty_:
                    uncertPos.append( [2.0*P_mat[0,0]**0.5, 2.0*P_mat[1,1]**0.5, 2.0*P_mat[2,2]**0.5] )
                    uncertVel.append( [2.0*P_mat[3,3]**0.5, 2.0*P_mat[4,4]**0.5, 2.0*P_mat[5,5]**0.5] )

                else:
                    uncertPos.append( [0,0,0] )
                    uncertVel.append( [0,0,0] )


            else:

                if norm(av2Pos - av1Pos) < self.cvoGekko.collisionRange:
                    inRangePos.append(av2Pos)
                    inRangeVel.append(av2Vel)
                    uncertPos.append( [0,0,0] )
                    uncertVel.append( [0,0,0] )




        # print('get best ', self.id, " ", self.allKalFilters)
        # print(inRangeVel)
        # set_trace()
        vel = self.cvoGekko.get_best_vel(av1Pos, av1Vel, vel_d,
                        inRangePos, inRangeVel,
                        uncertPos, uncertVel)

        return np.asarray([vel]).T
