import numpy as np
import json
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from IPython.core.debugger import set_trace

from animation import Animation

data = []

with open("data/myfile.json", "r") as read_it:
    data = json.load(read_it)

# set_trace()
numVehicles = len(data.keys())
animation = Animation(numVehicles)

jump = 60

for i in range(len(data['0'])//jump):
    # print(i*jump)

    animation.drawAll(data, i*jump)
    # print(quadStates[i*jump]['v'][0])
    # print(quadCommandedStates[i*jump]['v'][0])

    plt.pause(0.001)
    # set_trace()

# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# for key in data:
#     # set_trace()
#     x = np.array(data[key])[:,0]
#     y = np.array(data[key])[:,1]
#     z = np.array(data[key])[:,2]
#     ax.plot(x,y,z)

plt.show()
