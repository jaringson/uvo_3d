from os import walk
import re

import numpy as np
from numpy.linalg import norm
import json
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from IPython.core.debugger import set_trace

import params

from multiprocessing import Pool


data = []
numDataPoints = 0

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
        return 1
    return 0


mypath = 'superData/hardData1/'
# mypath = 'superData/2dData1/'
f = []
for (dirpath, dirnames, filenames) in walk(mypath):
    # print(dirpath, dirnames, filenames)
    f.extend(filenames)
    break


allCollisionsPerVel = []
allCollisionRange = []
allNumVehicles = []
allVelocities = []

for file in f:
    # global data
    # global numDataPoints
    # nums = re.findall("\d+\.\d+",file)
    values = re.findall("\d+",file)
    numVehicles = int(values[1])
    collisionRange = float(values[2]+'.'+values[3])
    velocity = float(values[4]+'.'+values[5])

    with open(mypath+file, "r") as read_it:
        data = json.load(read_it)

    numDataPoints = len(data['0'])

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

    # print('# Collisions/vehicle: ', np.sum(collisions)/numVehicles)
    if np.sum(collisions) > 0:
        print(mypath+file)
    allCollisionsPerVel.append(np.sum(collisions)/numVehicles)
    allCollisionRange.append(collisionRange)
    allNumVehicles.append(numVehicles)
    allVelocities.append(velocity)

    # if collisionRange > 40 and np.sum(collisions) > 0:
    #     print(mypath+file)

    # print(file, nums)
    # print(int(nums[1]))
    # print(float(nums[2]+'.'+nums[3]))


outData = {}
outData['numVehicles'] = allNumVehicles
outData['collisionsPerVel'] = allCollisionsPerVel
outData['collisionRange'] = allCollisionRange
outData['velocity'] = allVelocities
outFile = open('superData/dynDataProcessed.json', "w")
json.dump(outData, outFile, indent=3)
outFile.close()
