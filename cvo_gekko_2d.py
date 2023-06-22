#import yaml
import numpy as np
from numpy.linalg import norm
from gekko import gekko
import time
from random import random
from IPython.core.debugger import set_trace

from scipy.optimize import fsolve

from utils import angle_between

class CVOGekko:
    def __init__(self, params, id, collision_range):
        self.alpha = 1.0 #0.5
        self.id = id

        self.collisionRadius = params.collision_radius
        self.bufferPower = params.buffer_power
        self.bufferOn = params.buffer_on
        self.collisionRange = collision_range

        self.max_vel = params.max_vel



    def get_best_vel(self, av1Xo, av1Vo, av1VelDes,
            inRangePos, inRangeVel,
            uncertaintyPos, uncertaintyVel):
        numOther = len(inRangePos)

        # av1BuffVel = np.zeros((2,1))
        # trigger = False
        # if self.bufferOn:
        #     av1BuffVel, trigger = self.calculate_buffer(av1Xo, inRangePos, uncertaintyPos)
        #     # av1BuffVel = av1BuffVel / (0.5*norm(av1VelDes))
        #
        # if trigger:
        #     print('trigger')
        #     # set_trace()
        #     # return av1BuffVel[0,0], av1BuffVel[1,0]
        #
        # av1VelDes = (2.0*av1VelDes + av1BuffVel)/2.0;

        vdx = av1VelDes[0,0]
        vdy = av1VelDes[1,0]

        m = gekko(remote=False)

        sx = m.Var(av1VelDes[0,0], name='sx')
        sy = m.Var(av1VelDes[1,0], name='sy')

        y1 = m.Param(1e4)
        y2 = m.Param(1e-1)

        debug_val = 0


        # allHomogenousMatrices = []
        # allCollisionConeApex = []
        # allTangentLines = []
        # allTimeToCollisionWeights = []

        allContraints = []
        allDeltas = []

        for i in range(numOther):
            av2Xo = inRangePos[i]
            av2Vo = inRangeVel[i]

            from1XTo2X = av2Xo - av1Xo


            apexOfCollisionCone = (1.0-self.alpha)*av1Vo + self.alpha*av2Vo
            centerOfEllipsoid = from1XTo2X + apexOfCollisionCone


            ''' Position Uncertainty '''
            a = uncertaintyPos[i][0]+2.0*self.collisionRadius
            b = uncertaintyPos[i][1]+2.0*self.collisionRadius

            start_a = a
            start_b = b
            apexInEllipsoid = True
            failed = False

            while apexInEllipsoid:

                j_pos = from1XTo2X[0,0]
                k_pos = from1XTo2X[1,0]
                if j_pos**2/a**2 + k_pos**2/b**2 - 1.0 < 0:
                    failed = True

                    a = a - 0.001
                    b = b - 0.001

                    if a <= 0 or b <= 0:
                        a = 0.0001
                        b = 0.0001
                        apexInEllipsoid = False
                    continue
                apexInEllipsoid = False

            # if failed:
            #     print('id: ', self.id, ' other: ', i, ' start_a: ', start_a, ' a : ', a, ' norm: ', norm(from1XTo2X))

            ''' Velocity Uncertainty '''
            uVx = uncertaintyVel[i][0]
            uVy = uncertaintyVel[i][1]
            s_norm = m.Intermediate(m.sqrt((sx-apexOfCollisionCone[0,0])**2 + (sy-apexOfCollisionCone[1,0])**2))
            x_trans = m.Intermediate(uVx * (sx-apexOfCollisionCone[0,0]) / s_norm)
            y_trans = m.Intermediate(uVy * (sy-apexOfCollisionCone[1,0]) / s_norm)

            ''' Make sure the point is not inside velocity uncertainty '''
            m.Equation( (sx-apexOfCollisionCone[0,0])**2/uVx**2 +
                                (sy-apexOfCollisionCone[1,0])**2/uVy**2 - 1 >= 0 )

            apx = apexOfCollisionCone[0,0] + x_trans
            apy = apexOfCollisionCone[1,0] + y_trans

            j = centerOfEllipsoid[0,0] + x_trans
            k = centerOfEllipsoid[1,0] + y_trans


            new_s = np.array([
                    [sx-apx],
                    [sy-apy],
                    [0.0]])

            M = np.array([ [1.0/(a*a), 0.0, -j/(a*a)],
                [0.0, 1.0/(b*b), -k/(b*b)],
                [-j/(a*a), -k/(b*b), j*j/(a*a)+k*k/(b*b)-1.0] ])

            ap = np.array([[apx], [apy], [1.0]])

            lam1 = m.Intermediate((-new_s.T@M@ap)[0,0])
            lam2 = m.Intermediate((new_s.T@M@new_s)[0,0])
            # lam = m.Intermediate(m.sqrt( lam1*lam1/(lam2*lam2) ))
            lam = m.Intermediate(m.abs2(lam1/lam2))

            valx = apx+lam*(sx-apx)
            valy = apy+lam*(sy-apy)

            val = np.array([ [valx], [valy], [1.0] ])
            # set_trace()

            constraint = m.Intermediate((val.T@M@val)[0,0])
            # allContraints.append(constraint)

            #delta = m.Var()
            #allDeltas.append(delta)
            #
            #x1 = self.collisionRadius
            #x2 = self.collisionRange
            #m_line = (y2-y1)/(x2-x1)
            #b_line = y1-m_line*x1
            #weight = m.Intermediate(m_line*norm(from1XTo2X)+b_line)
            # weight = 1e6*self.collisionRange*1.0/(norm(from1XTo2X)+self.collisionRadius-0.1)

            #m.Equation( constraint + delta >= 0  )
            #m.Obj(delta * delta * weight)

            m.Equation( constraint >= 0  )



        # allDeltas = np.array(allDeltas)
        # allDistanceWeights = np.diag(allDistanceWeights)
        # deltas_cost = m.Intermediate(allDeltas.T @ allDistanceWeights @ allDeltas)

        normConstraint = m.Intermediate(m.sqrt(sx**2 + sy**2))
        allContraints.append(normConstraint)

        m.Equation( normConstraint <= self.max_vel )
        m.Minimize( (sx-vdx)**2 + (sy-vdy)**2 )
        m.options.SOLVER = 3 # IPOPT solver
        # m.options.MAX_ITER = 100
        # t1 = time.time()
        solve_success = False
        for i in range(10):
            # if i > 50:
            #     print(i, self.id)
            #     # set_trace()
            try:
                # print('try: ', self.id, " iter: ", i)
                debug_val = m.solve(disp=False, debug=True)
                solve_success = True
                break
            except:

                # set_trace()

                # print('Try again. id: ', self.id)
                n = self.max_vel #norm(av1VelDes)
                sx.value = [np.random.uniform(-1,1)*n]
                sy.value = [np.random.uniform(-1,1)*n]

                y1.value /= 1.0
                y2.value /= 1.0

                solve_success = False
                # print('Try again. id: ', self.id, " y1: ", y1.value, " y2: ", y2.value)


        if not solve_success:
            # print('Solve Success Error')
            sx.value = [0] #[av1Vo[0,0]] # [av1VelDes[0,0]]
            sy.value = [0] #[av1Vo[1,0]] # [av1VelDes[1,0]]
            # for c in allContraints:
            #     print("c value :", c.value)
            # print("s :", sx.value[0], sy.value[0], sz.value[0])
            print('Failed. id: ', self.id)
            # set_trace()


        # t2 = time.time()

        # print('sx: ' + str(sx.value))
        # print('sy: ' + str(sy.value))
        # print('sz: ' + str(sz.value))

        # for c in allContraints:
        #     print("const: ", c.value)

        # set_trace()


        return sx.value[0], sy.value[0]
