#import yaml
import numpy as np
from numpy.linalg import norm
from gekko import gekko
import time
from random import random
from IPython.core.debugger import set_trace


class CVOGekko:
    def __init__(self, params, id, collision_range):
        self.alpha = 0.5
        self.id = id

        self.collisionRadius = params.collision_radius
        self.bufferPower = params.buffer_power
        self.bufferOn = params.buffer_on
        self.collisionRange = collision_range

        self.max_vel = params.max_vel

        self.slackGamma = params.slack_gamme

    def calculate_buffer(self, host_pos, invader_pos):

        buffer_radius = 1.1*self.collisionRadius
        power = self.bufferPower

        num_invaders = len(invader_pos)
        if(num_invaders == 0):
            return np.zeros((3,1))

        sum_velocity = np.zeros((3,1))

        for i in range(num_invaders):


            host = host_pos
            invader = invader_pos[i]
            diff = invader - host
            euc_dist = norm(diff)

            dist_to_buffer = euc_dist - buffer_radius

            buffer_force = (buffer_radius/dist_to_buffer)**self.bufferPower
            buffer_velocity = abs(buffer_force) * (-diff)/norm(diff)
            sum_velocity += buffer_velocity

        avg_velocity = sum_velocity / (1.0*num_invaders)

        if norm(avg_velocity) > self.max_vel:
            avg_velocity = avg_velocity / norm(avg_velocity) * self.max_vel
        # set_trace()

        return avg_velocity

    def get_best_vel(self, av1Xo, av1Vo, av1VelDes,
            inRangePos, inRangeVel,
            uncertaintyPos, uncertaintyVel):
        numOther = len(inRangePos)

        av1BuffVel = np.zeros((3,1))
        if self.bufferOn:
            av1BuffVel = self.calculate_buffer(av1Xo, inRangePos)
            av1BuffVel = av1BuffVel / (0.5*norm(av1VelDes))

        av1VelDes = (2.0*av1VelDes + av1BuffVel)/2.0;

        vdx = av1VelDes[0,0]
        vdy = av1VelDes[1,0]
        vdz = av1VelDes[2,0]

        m = gekko(remote=False)

        sx = m.Var(2*(0.5-random()*norm(av1VelDes)), name='sx')
        sy = m.Var(2*(0.5-random()*norm(av1VelDes)), name='sy')
        sz = m.Var(2*(0.5-random()*norm(av1VelDes)), name='sz')

        s = np.array([[sx],[sy],[sz], [0.0]])

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

            ''' Velocity Uncertainty '''
            fromCenterToApex = apexOfCollisionCone - centerOfEllipsoid
            max_vel_uncertianty = np.max(uncertaintyVel)
            apexOfCollisionCone += max_vel_uncertianty*fromCenterToApex/norm(fromCenterToApex)

            # print('apex: ', apexOfCollisionCone)
            apexOfCollisionCone4D = np.block([[apexOfCollisionCone], [1.0]])


            ''' Position Uncertainty '''
            a = uncertaintyPos[i][0]+2.0*self.collisionRadius
            b = uncertaintyPos[i][1]+2.0*self.collisionRadius
            c = uncertaintyPos[i][2]+2.0*self.collisionRadius
            j = centerOfEllipsoid[0,0]
            k = centerOfEllipsoid[1,0]
            l = centerOfEllipsoid[2,0]

            M = np.array([ [1.0/(a*a), 0.0, 0.0, -2.0*j/(a*a)*0.5],
                  [0.0, 1.0/(b*b), 0.0, -2.0*k/(b*b)*0.5],
                  [0.0, 0.0, 1.0/(c*c), -2.0*l/(c*c)*0.5],
                  [-2.0*j/(a*a)*0.5, -2.0*k/(b*b)*0.5, -2.0*l/(c*c)*0.5, j*j/(a*a)+k*k/(b*b)+l*l/(c*c)-1.0] ])

            # allHomogenousMatrices.append(ellipsoidHomogenousMatrix)
            # allCollisionConeApex.append(apexOfCollisionCone4D)
            # allTangentLines.push_back(tangentLine)
            # allTimeToCollisionWeights.append(timeWeight)


            ap = apexOfCollisionCone4D
            # print(M)
            # print(ap)
            # print(s)
            # print(vdx,vdy,vdz)

            new_s = np.array([[sx-ap[0,0]],[sy-ap[1,0]],[sz-ap[2,0]], [0.0]])


            lam1 = m.Intermediate((-new_s.T@M@ap)[0,0])
            lam2 = m.Intermediate((new_s.T@M@new_s)[0,0])
            lam = m.Intermediate(m.sqrt( lam1*lam1/(lam2*lam2) ))
            # lam = self.m.Intermediate(self.m.abs2(lam1/lam2))

            valx = ap[0,0]+lam*(sx-ap[0,0])
            valy = ap[1,0]+lam*(sy-ap[1,0])
            valz = ap[2,0]+lam*(sz-ap[2,0])

            val = np.array([ [valx], [valy], [valz], [1.0] ])
            # set_trace()

            constraint = m.Intermediate((val.T@M@val)[0,0])
            allContraints.append(constraint)
            delta = m.Var()
            allDeltas.append(delta)

            y1 = m.Param(1e3)
            y2 = m.Param(1e2)
            x1 = self.collisionRadius
            x2 = self.collisionRange
            m_line = (y2-y1)/(x2-x1)
            b_line = y1-m_line*x1
            weight = m_line*norm(from1XTo2X)+b_line
            # weight = 1e6*self.collisionRange*1.0/(norm(from1XTo2X)+self.collisionRadius-0.1)

            m.Equation( constraint + delta >= 0  )
            m.Obj(delta * delta * weight)

        # allDeltas = np.array(allDeltas)
        # allDistanceWeights = np.diag(allDistanceWeights)
        # deltas_cost = m.Intermediate(allDeltas.T @ allDistanceWeights @ allDeltas)

        normConstraint = m.Intermediate(m.sqrt(sx**2 + sy**2 + sz**2))
        allContraints.append(normConstraint)

        m.Equation( normConstraint <= self.max_vel )
        m.Minimize( (sx-vdx)**2 + (sy-vdy)**2 + (sz-vdz)**2)
        m.options.SOLVER = 3 # IPOPT solver
        # m.options.MAX_ITER = 100
        # t1 = time.time()
        solve_success = False
        for i in range(100):
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
                n = norm(av1VelDes)
                sx.value = [2*(0.5-random())*n]
                sy.value = [2*(0.5-random())*n]
                sz.value = [2*(0.5-random())*n]

                y1.value /= 2.0
                y2.value /= 2.0

                solve_success = False


        if not solve_success:
            # print('Solve Success Error')
            sx.value = [av1Vo[0,0]] # [av1VelDes[0,0]]
            sy.value = [av1Vo[1,0]] # [av1VelDes[1,0]]
            sz.value = [av1Vo[2,0]] # [av1VelDes[2,0]]
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


        return sx.value[0], sy.value[0], sz.value[0]
