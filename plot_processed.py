import numpy as np
from numpy.linalg import norm
import json
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from IPython.core.debugger import set_trace

import matplotlib.pyplot as plt

file = 'superData/dynDataProcessed.json'

plt.rcParams.update({'font.size': 15})

with open(file, "r") as read_it:
    data = json.load(read_it)

# print(data)

collisionRange = np.array(data['collisionRange'])
collisionsPerVel = np.array(data['collisionsPerVel'])
numVehicles = np.array(data['numVehicles'])
velocity = np.array(data['velocity'])
zerosInd = collisionsPerVel==0
notZerosInd = collisionsPerVel!=0
# set_trace()

fig = plt.figure()
ax = fig.add_subplot(111)
ax.title.set_text(" ")
# ax.scatter(data['collisionRange'], data['numVehicles'])
ax.scatter( collisionRange[zerosInd], collisionsPerVel[zerosInd], color='red', s=20, marker="*", label='Zero Collisions' )
ax.scatter( collisionRange[notZerosInd], collisionsPerVel[notZerosInd], s=20, marker="o", label='Non-Zero Collisions' )
# ax.set_ylim([-0.5,2.5])

plt.xlabel('Collision Range (m)')
plt.ylabel('# Collisions / Vehicle')


fig = plt.figure()
ax = fig.add_subplot(111)
ax.title.set_text(" ")

ax.scatter( velocity[zerosInd], collisionRange[zerosInd], color='red', s=20, marker="*", label='Zero Collisions' )
ax.scatter( velocity[notZerosInd], collisionRange[notZerosInd], s=20, marker="o", label='Non-Zero Collisions' )

plt.show()
