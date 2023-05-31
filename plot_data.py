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

# set_trace()

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
#     # set_trace()

# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# for key in data:
#     x = np.array(data[key])[:,0]
#     y = np.array(data[key])[:,1]
#     z = np.array(data[key])[:,2]
#     ax.plot(x,y,z)

# set_trace()

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
x = np.array(data['1'])[:,0]
y = np.array(data['1'])[:,1]
z = np.array(data['1'])[:,2]
kal_x = np.array(kalData['1'])[:,0]
kal_y = np.array(kalData['1'])[:,1]
kal_z = np.array(kalData['1'])[:,2]
# time = np.linspace(0,P.sim_t,x.shape[0])
ax.scatter(x, y, z)
ax.scatter(kal_x, kal_y, kal_z)

# ax.plot(time,x)
# ax.plot(time,kal_x)
plt.show()
