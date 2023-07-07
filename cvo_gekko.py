#import yaml
import numpy as np
from numpy.linalg import norm
from gekko import gekko
import time
from random import random
from IPython.core.debugger import set_trace

import scipy
from scipy.optimize import fsolve


from utils import angle_between

import warnings
warnings.filterwarnings("ignore")

class CVOGekko:
    def __init__(self, params, id, collision_range, max_vel):
        self.alpha = 1.0 #0.5
        self.id = id

        self.collisionRadius = params.collision_radius
        self.bufferPower = params.buffer_power
        self.bufferOn = params.buffer_on
        self.collisionRange = collision_range

        self.max_vel = max_vel

    def solve_equations(self, p, data):
        a1, b1, c1, j1, k1, l1, a2, b2, c2 = data
        # set_trace()
        x, y, z = p
        # x, y, z, _, _ = p
        # return (2*j1*k1*c1**2*(-y**2*c2**2-z**2*b2**2+b2**2*c2**2) - 2*x*y*c2**2*(-k1**2*c1**2-l1**2*b1**2+b1**2*c1**2),
        #          2*j1*l1*b1**2*(-x**2*c2**2-z**2*a2**2+a2**2*c2**2) - 2*x*z*b2**2*(-j1**2*c1**2-l1**2*a1**2+a1**2*c1**2),
        #           2*k1*l1*a1**2*(-x**2*b2**2-y**2*a2**2+a2**2*b2**2) - 2*y*z*a2**2*(-j1**2*b1**2-k1**2*a1**2+a1**2*b1**2),
        #            # j1*k1*c1**2 * x*z*b2**2 - x*y*c2**2 * j1*l1*b1**2,
        #             # j1*k1*c1**2 * y*z*a2**2 - x*y*c2**2 * k1*l1*a1**2
        #              )

        return (2*k1*l1*a1**2*(-x**2*b2**2-y**2*a2**2+a2**2*b2**2) - 2*y*z*a2**2*(-j1**2*b1**2-k1**2*a1**2+a1**2*b1**2),
                 2*j1*k1*c1**2*(-x**2*c2**2-z**2*a2**2+a2**2*c2**2) - 2*x*y*c2**2*(-j1**2*c1**2-l1**2*a1**2+a1**2*c1**2),
                  2*j1*l1*b1**2*(-y**2*c2**2-z**2*b2**2+b2**2*c2**2) - 2*x*z*b2**2*(-k1**2*c1**2-l1**2*b1**2+b1**2*c1**2),
                   # j1*k1*c1**2 * x*z*b2**2 - x*y*c2**2 * j1*l1*b1**2,
                    # j1*k1*c1**2 * y*z*a2**2 - x*y*c2**2 * k1*l1*a1**2
                     )
    def f(self, p, data):
        a1, b1, c1, j1, k1, l1, a2, b2, c2 = data
        x, y, z = p
        return (x-1000*j1)**2 + (y-1000*k1)**2 + (z-1000*l1)**2

    def g1(self, p, *data):
        a1, b1, c1, j1, k1, l1, a2, b2, c2 = data
        x, y, z = p
        return 2*k1*l1*a1**2*(-x**2*b2**2-y**2*a2**2+a2**2*b2**2) - 2*y*z*a2**2*(-j1**2*b1**2-k1**2*a1**2+a1**2*b1**2)

    def g2(self, p, *data):
        a1, b1, c1, j1, k1, l1, a2, b2, c2 = data
        x, y, z = p
        return 2*j1*k1*c1**2*(-x**2*c2**2-z**2*a2**2+a2**2*c2**2) - 2*x*y*c2**2*(-j1**2*c1**2-l1**2*a1**2+a1**2*c1**2)

    def g3(self, p, *data):
        a1, b1, c1, j1, k1, l1, a2, b2, c2 = data
        x, y, z = p
        return 2*j1*l1*b1**2*(-y**2*c2**2-z**2*b2**2+b2**2*c2**2) - 2*x*z*b2**2*(-k1**2*c1**2-l1**2*b1**2+b1**2*c1**2)

    def g(self, p, *data):
        a1, b1, c1, j1, k1, l1, a2, b2, c2 = data
        x, y, z = p

        M_pos = np.array([ [1.0/(a1*a1), 0.0, 0.0, -j1/(a1*a1)],
                        [0.0, 1.0/(b1*b1), 0.0, -k1/(b1*b1)],
                        [0.0, 0.0, 1.0/(c1*c1), -l1/(c1*c1)],
                        [-j1/(a1*a1), -k1/(b1*b1), -l1/(c1*c1),
                            j1*j1/(a1*a1)+k1*k1/(b1*b1)+l1*l1/(c1*c1)-1.0] ])

        ap_pos = np.block([[np.zeros((3,1))], [1.0]])
        s_pos = np.block([[x], [y], [z], [0.0]])
        lam_pos = np.abs((-s_pos.T@M_pos@ap_pos) / (s_pos.T@M_pos@s_pos))

        ret = (ap_pos+lam_pos*s_pos).T@M_pos@(ap_pos+lam_pos*s_pos)
        # set_trace()

        return ret[0,0]

    def get_best_vel(self, av1Xo, av1Vo, av1VelDes,
            inRangePos, inRangeVel,
            uncertaintyPos, uncertaintyVel):
        numOther = len(inRangePos)


        vdx = av1VelDes[0,0]
        vdy = av1VelDes[1,0]
        vdz = av1VelDes[2,0]

        m = gekko(remote=False)

        sx = m.Var(vdx, name='sx')
        sy = m.Var(vdy, name='sy')
        sz = m.Var(vdz, name='sz')


        y1 = m.Param(1e4)
        y2 = m.Param(1e1)

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
            c = uncertaintyPos[i][2]+2.0*self.collisionRadius
            j_pos = from1XTo2X[0,0]
            k_pos = from1XTo2X[1,0]
            l_pos = from1XTo2X[2,0]

            start_a = a
            start_b = b
            start_c = c
            apexInEllipsoid = True
            failed = False

            while apexInEllipsoid:

                if j_pos**2/a**2 + k_pos**2/b**2 + l_pos**2/c**2  - 1.0 < 0:
                    failed = True

                    a = a - 0.001
                    b = b - 0.001
                    c = c - 0.001

                    if a <= 0 or b <= 0 or c<=0:
                        a = 0.0001
                        b = 0.0001
                        c = 0.0001
                        apexInEllipsoid = False
                    continue
                apexInEllipsoid = False
            # if failed:
            #     print('id: ', self.id, ' start_a: ', start_a, ' a : ', a, ' norm: ', norm(from1XTo2X))


            ''' Velocity Uncertainty '''
            if norm(from1XTo2X) < 5.0*self.max_vel:
                data = [a,b,c,
                        from1XTo2X[0,0],from1XTo2X[1,0],from1XTo2X[2,0],
                        uncertaintyVel[i][0],uncertaintyVel[i][1],uncertaintyVel[i][2]]
                cons = [{'type':'ineq', 'fun':self.g, 'args':data},
                        {'type':'eq', 'fun':self.g1, 'args':data},
                        {'type':'eq', 'fun':self.g2, 'args':data},
                        {'type':'eq', 'fun':self.g3, 'args':data}]
                vel_translate_full = scipy.optimize.minimize(self.f,
                    (from1XTo2X[0,0],from1XTo2X[1,0],from1XTo2X[2,0]),
                    # method = 'SLSQP',
                    args=data,
                    constraints=cons )

                # vel_translate_full = fsolve(self.solve_equations,
                #     (from1XTo2X[0,0],from1XTo2X[1,0],from1XTo2X[2,0]),
                #     args=data)
                vel_translate = vel_translate_full.x
                vel_translate = vel_translate.reshape((3,1))

                ''' ---1 '''
                if norm(self.solve_equations(vel_translate_full.x, data)) < 1e-2:

                    M_pos = np.array([ [1.0/(a*a), 0.0, 0.0, -j_pos/(a*a)],
                                       [0.0, 1.0/(b*b), 0.0, -k_pos/(b*b)],
                                       [0.0, 0.0, 1.0/(c*c), -l_pos/(c*c)],
                                       [-j_pos/(a*a), -k_pos/(b*b), -l_pos/(c*c),
                                        j_pos*j_pos/(a*a)+k_pos*k_pos/(b*b)+l_pos*l_pos/(c*c)-1.0] ])

                    ap_pos = np.block([[np.zeros((3,1))], [1.0]])
                    s_pos = np.block([[vel_translate], [0.0]])
                    lam_pos = np.abs((-s_pos.T@M_pos@ap_pos) / (s_pos.T@M_pos@s_pos))

                    if (ap_pos+lam_pos*s_pos).T@M_pos@(ap_pos+lam_pos*s_pos) <= 0:
                        # print('passed')
                        apexOfCollisionCone = apexOfCollisionCone - vel_translate
                        centerOfEllipsoid = centerOfEllipsoid - vel_translate
                #     else:
                #         # print('switch. id: ', self.id)
                #         # print('   Ellipse fail. av1Xo: ', av1Xo, ' av2Xo: ', av2Xo)
                #         print('   Ellipse fail. val: ', (ap_pos+lam_pos*s_pos).T@M_pos@(ap_pos+lam_pos*s_pos))
                #         set_trace()
                # else:
                #     print('   Norm Fail. norm: ', norm(from1XTo2X) )
                #     # print('   Norm Fail.')
                #     set_trace()

                ''' 1---2 '''
                # if abs(angle_between(-vel_translate_check.T, from1XTo2X)) < abs(angle_between(vel_translate_check.T, from1XTo2X)):
                #     vel_translate_check = -vel_translate_check
                #     print('switch')
                # if norm(self.solve_equations(vel_translate, data)) < 1e-4:
                #         # print(angle_between(vel_translate.T, from1XTo2X))
                #         apexOfCollisionCone = apexOfCollisionCone - vel_translate_check
                #         centerOfEllipsoid = centerOfEllipsoid - vel_translate_check
                ''' 2--- '''



            apx = apexOfCollisionCone[0,0]
            apy = apexOfCollisionCone[1,0]
            apz = apexOfCollisionCone[2,0]

            j = centerOfEllipsoid[0,0]
            k = centerOfEllipsoid[1,0]
            l = centerOfEllipsoid[2,0]

            new_s = np.array([
                    [sx-apx],
                    [sy-apy],
                    [sz-apz],
                    [0.0]])

            M = np.array([ [1.0/(a*a), 0.0, 0.0, -j/(a*a)],
                            [0.0, 1.0/(b*b), 0.0, -k/(b*b)],
                            [0.0, 0.0, 1.0/(c*c), -l/(c*c)],
                            [-j/(a*a), -k/(b*b), -l/(c*c), j*j/(a*a)+k*k/(b*b)+l*l/(c*c)-1.0] ])

            ap = np.array([[apx], [apy], [apz], [1.0]])

            lam1 = m.Intermediate((-new_s.T@M@ap)[0,0])
            lam2 = m.Intermediate((new_s.T@M@new_s)[0,0])
            # lam = m.Intermediate(m.sqrt( lam1*lam1/(lam2*lam2) ))
            lam = m.Intermediate(m.abs2(lam1/lam2))

            valx = apx+lam*(sx-apx)
            valy = apy+lam*(sy-apy)
            valz = apz+lam*(sz-apz)

            val = np.array([ [valx], [valy], [valz], [1.0] ])
            # set_trace()

            constraint = m.Intermediate((val.T@M@val)[0,0])
            # allContraints.append(constraint)

            m.Equation( constraint >= 0  )


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
                sz.value = [np.random.uniform(-1,1)*n]

                solve_success = False
                # print('Try again. id: ', self.id, " y1: ", y1.value, " y2: ", y2.value)


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
