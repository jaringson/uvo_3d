import numpy as np
import json
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from IPython.core.debugger import set_trace
import params as P

from animation import Animation

data = []
with open("data/data.json", "r") as read_it:
    data = json.load(read_it)

numVehicles = len(data.keys())

kalData = []
with open("data/kaldata.json", "r") as read_it2:
    kalData = json.load(read_it2)

kalCov = []
with open("data/kalcov.json", "r") as read_it3:
    kalCov = json.load(read_it3)

''' -------------------------- '''
# animation = Animation(numVehicles)
# jump = 60
#
# for i in range(len(data['0'])//jump):
#     # print(i*jump)
#
#     animation.drawAll(data, i*jump)
#     # print(quadStates[i*jump]['v'][0])
#     # print(quadCommandedStates[i*jump]['v'][0])
#
#     plt.pause(0.001)

''' -------------------------- '''
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# for key in data:
#     x = np.array(data[key])[:,0]
#     y = np.array(data[key])[:,1]
#     z = np.array(data[key])[:,2]
#     ax.plot(x,y,z)


''' -------------------------- '''
x = np.array(data['1'])[:,0]
y = np.array(data['1'])[:,1]
z = np.array(data['1'])[:,2]
kal_x = np.array(kalData['1'])[:,0]
kal_y = np.array(kalData['1'])[:,1]
kal_z = np.array(kalData['1'])[:,2]

kalcov_x = np.array(kalCov['1'])[:,0]
kalcov_y = np.array(kalCov['1'])[:,1]
kalcov_z = np.array(kalCov['1'])[:,2]
time = np.linspace(0,P.sim_t,x.shape[0])

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(x, y, z)
ax.scatter(kal_x, kal_y, kal_z)

fig, axs = plt.subplots(3, 1)
axs[0].plot(time,x)
axs[0].plot(time,kal_x)
axs[0].plot(time,kal_x+3*kalcov_x,color='green')
axs[0].plot(time,kal_x-3*kalcov_x,color='green')

axs[1].plot(time,y)
axs[1].plot(time,kal_y)
axs[1].plot(time,kal_y+3*kalcov_y,color='green')
axs[1].plot(time,kal_y-3*kalcov_y,color='green')

axs[2].plot(time,z)
axs[2].plot(time,kal_z)
axs[2].plot(time,kal_z+3*kalcov_z,color='green')
axs[2].plot(time,kal_z-3*kalcov_z,color='green')

print(kal_x)

plt.show()
