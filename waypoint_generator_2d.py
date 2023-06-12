import numpy as np
from numpy.linalg import norm
import time

PI = 3.14159265359

def get_random_position(startRadius):
    min = -PI
    max = PI
    randTheta = np.random.random() * (max-min) + min
    randPhi = np.pi #np.random.random() * (max-min) + min

    x = startRadius*np.cos(randPhi)*np.cos(randTheta)
    y = startRadius*np.cos(randPhi)*np.sin(randTheta)
    z = startRadius*np.sin(randPhi)

    return np.array([[x], [y], [z]]) # np.array([[-10.0], [10.0], [0.0]]) #

def get_waypoints(startRadius, numVehicles, vehicleRadius, seed=int(time.time())):

    np.random.seed(seed)

    allPositions = []
    allWaypoints = []

    point = get_random_position(startRadius)
    allPositions.append(point)
    point_w_psi = np.block([[-point], [0]])
    twoPoints = np.vstack((point_w_psi, -point_w_psi)).flatten().tolist()
    allWaypoints.append(twoPoints)

    for i in range(numVehicles-1):
        foundOther = True
        point = []
        while foundOther:

            point = get_random_position(startRadius)

            for pos in allPositions:
                if norm(pos-point) < vehicleRadius*2.1:
                    foundOther = True
                    break

                foundOther = False


        allPositions.append(point)
        point_w_psi = np.block([[-point], [0]])
        twoPoints = np.vstack((point_w_psi, -point_w_psi)).flatten().tolist()
        allWaypoints.append(twoPoints)

    return allWaypoints, allPositions
