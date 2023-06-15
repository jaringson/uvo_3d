import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from numpy.linalg import norm
from utils import roll, pitch, yaw

import params as P
from quadrotor import QuadDynamics
if P.is_2d:
    from quad_cvo_manager_2d import CVOManager
    from waypoint_generator_2d import get_waypoints
else:
    from quad_cvo_manager import CVOManager
    from waypoint_generator import get_waypoints

from dataPlotter import dataPlotter
from quad_control import Controller
from quad_wp_manager import WPManager

import json
import time
from tqdm import tqdm
from IPython.core.debugger import set_trace

from multiprocessing import Pool

def init_pool(allWPManagers, allCVOManagers, allQuads, t):
    global allWP
    global allCVO
    global allQ
    global time_sim

    allWP = allWPManagers
    allCVO = allCVOManagers
    allQ = allQuads
    time_sim = t

def multi_cvo(id):

    rand_int = int((time.time()%100) * 1e6)
    np.random.seed(rand_int)

    vel_d = allWP[id].updateWaypointManager(allQ[id].x_)
    vel_c = allCVO[id].get_best_vel(allQ, time_sim, vel_d)

    return id, vel_c, allWP[id], allCVO[id], allQ[id]


def run_sim(num_quads, collision_range, max_vel, filename):


    num_quads = num_quads
    radius = P.start_radius
    seed = 1 #int(time.time())

    waypoints, allStartPositions = get_waypoints(radius, num_quads, P.collision_radius, seed=seed)
    print('waypoints: ', waypoints)
    print('Num of Vehicles: ', num_quads)


    allControllers = {}
    allVelCon = {}

    allStates = {}
    allKalStates = {}
    allKalCovariance = {}


    allWPManagers = {}
    allQuads = {}
    allCVOManagers = {}
    allEverything = []


    for i in range(num_quads):
        id = i
        quad = QuadDynamics(P, id, allStartPositions[id])
        controller = Controller(P, id)
        cvoManager = CVOManager(P, id, collision_range)
        wpManager = WPManager(P, id, waypoints[id], max_vel)

        allQuads[id] = quad
        allControllers[id] = controller
        allCVOManagers[id] = cvoManager
        allWPManagers[id] = wpManager
        allStates[id] = []

    # for i in range(num_quads):
    #     everything = [i, allQuads, allControllers, allCVOManagers, allWPManagers]
    #     allEverything.append(everything)

    # dataPlot = dataPlotter()
    # plt.waitforbuttonpress()

    pbar = tqdm(total = (P.sim_t - P.t_start)/P.dt)

    t = P.t_start  # time starts at t_start
    t_control = t
    u = 0
    fpreP = 0


    while t < P.sim_t:  # main simulation loop

        # Propagate dynamics in between plot samples
        t_next_cvo = t + P.cvo_dt

        p = Pool(initializer=init_pool,
                initargs=(allWPManagers, allCVOManagers, allQuads, t))
        poolRet = p.map(multi_cvo, range(num_quads))

        p.close()
        p.join()

        for item in poolRet:
            id = item[0]
            velCon = item[1]
            wpManager = item[2]
            cvoManager = item[3]
            quad = item[4]

            allVelCon[id] = velCon
            allWPManagers[id] = wpManager
            allCVOManagers[id] = cvoManager
            allQuads[id] = quad

        # for id in range(len(allQuads)):
        #     # print(t)
        #     vel_d = allWPManagers[id].updateWaypointManager(allQuads[id].x_)
        #     vel_c = allCVOManagers[id].get_best_vel(allQuads, t, vel_d)
        #     allVelCon[id] = vel_c #np.array([[-10],[0],[0]])



        # updates control and dynamics at faster simulation rate
        while t < t_next_cvo:

            for id in range(len(allQuads)):
                allCVOManagers[id].propagate()
                # set_trace()
                u = allControllers[id].computeControl(allQuads[id].state, P.dt, allVelCon[id])
                y = allQuads[id].update(u)  # propagate system
                allStates[id].append(allQuads[id].state.flatten().tolist())

            allCVOManagers[0].get_kal_data(allKalStates, allKalCovariance)

            pbar.update(1)
            t = t + P.dt  # advance time by dt
        # update data plots
        # dataPlot.update(t, allQuads[0].state, u)

        # the pause causes the figure to display during simulation
        plt.pause(0.0001)


    pbar.close()

    out_file = open(filename, "w")
    json.dump(allStates, out_file, indent=3)
    out_file.close()

    out_file2 = open('data/kaldata.json', "w")
    json.dump(allKalStates, out_file2, indent=3)
    out_file2.close()

    out_file3 = open('data/kalcov.json', "w")
    json.dump(allKalCovariance, out_file3, indent=3)
    out_file3.close()


if __name__ == "__main__":
    run_sim(P.num_quads, P.collision_range, P.max_vel, 'data/data.json')
