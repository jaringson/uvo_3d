import numpy as np
from numpy.linalg import norm
import json
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from IPython.core.debugger import set_trace

import params

from multiprocessing import Pool

data = []

with open("dynData2/run1quads15cr8.36mv9.58.json", "r") as read_it:
    data = json.load(read_it)

numVehicles = len(data.keys())
numDataPoints = len(data['0'])

def check_collision(ij):
    global numDataPoints
    global data

    i = ij[0]
    j = ij[1]

    num_collisions = 0

    # print(i,j)
    av1Pos = np.array(data[str(i)])[:,0:3]
    av2Pos = np.array(data[str(j)])[:,0:3]
    allNorms = norm(av1Pos - av2Pos, axis=1)

    num_collisions = np.sum(allNorms < params.collision_radius)

    if num_collisions > 0:
        print(i,j)
        print(allNorms[allNorms < params.collision_radius])
        return 1
    return 0

num_collisions = 0

allIJ = []
for i in range(numVehicles):
    # print(i)
    for j in range(i+1,numVehicles):
        # print("  ", j)
        allIJ.append([i, j])

p = Pool()
collisions = p.map(check_collision, allIJ)
p.close()

# set_trace()

print('# Collisions: ', np.sum(collisions))
