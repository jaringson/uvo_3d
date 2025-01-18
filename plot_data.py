import numpy as np
import json
import matplotlib.pyplot as plt
# plt.rcParams['text.usetex'] = True

from mpl_toolkits.mplot3d import Axes3D
from IPython.core.debugger import set_trace
import params as P

from animation import Animation

data = []
with open("data/data.json", "r") as read_it:
# with open("superData/3dData4/run3quads17cr96.57mv7.68.json", "r") as read_it:
# with open("superData/2dData1/run0quads18cr94.27mv9.21.json", "r") as read_it:
    data = json.load(read_it)

numVehicles = len(data.keys())

kalData = []
with open("data/kaldata.json", "r") as read_it2:
    kalData = json.load(read_it2)

kalCov = []
with open("data/kalcov.json", "r") as read_it3:
    kalCov = json.load(read_it3)

kalTime = []
with open("data/kaltime.json", "r") as read_it4:
    kalTime = json.load(read_it4)

''' -------------------------- '''
# animation = Animation(numVehicles)
# jump = 10

# for i in range(len(data['0'])//jump):
#     # print(i*jump)

#     animation.drawAll(data, i*jump)
#     # print(quadStates[i*jump]['v'][0])
#     # print(quadCommandedStates[i*jump]['v'][0])

#     plt.pause(0.001)

''' -------------------------- '''
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
# ax = fig.add_subplot(111)
for key in data:
    x = np.array(data[key])[:,0]
    y = np.array(data[key])[:,1]
    z = np.array(data[key])[:,2]
    ax.plot(x,y,z)
    # ax.plot(x,y)
# ax.set_zlim(-50,50)

# set_trace()

''' -------------------------- '''
x = np.array(data['1'])[:,0]
y = np.array(data['1'])[:,1]
z = np.array(data['1'])[:,2]
vx = np.array(data['1'])[:,3]
vy = np.array(data['1'])[:,4]
vz = np.array(data['1'])[:,5]
time = np.linspace(0,P.sim_t,x.shape[0])

back_cf = -35
front_cf = 42
kal_x = np.array(kalData['1'])[:,0][front_cf:back_cf]
kal_y = np.array(kalData['1'])[:,1][front_cf:back_cf]
kal_z = np.array(kalData['1'])[:,2][front_cf:back_cf]
kal_vx = np.array(kalData['1'])[:,3][front_cf:back_cf]
kal_vy = np.array(kalData['1'])[:,4][front_cf:back_cf]
kal_vz = np.array(kalData['1'])[:,5][front_cf:back_cf]

kalcov_x = np.array(kalCov['1'])[:,0][front_cf:back_cf]
kalcov_y = np.array(kalCov['1'])[:,1][front_cf:back_cf]
kalcov_z = np.array(kalCov['1'])[:,2][front_cf:back_cf]
kalcov_vx = np.array(kalCov['1'])[:,3][front_cf:back_cf]
kalcov_vy = np.array(kalCov['1'])[:,4][front_cf:back_cf]
kalcov_vz = np.array(kalCov['1'])[:,5][front_cf:back_cf]
kal_time = np.array(kalTime)[front_cf:back_cf]

# set_trace()

plt.rcParams.update({'font.size': 20})

fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
ax = fig.add_subplot(111)
ax.scatter(x, y)
ax.scatter(kal_x, kal_y)

fig, axs = plt.subplots(3, 1, sharex=True)
axs[0].plot(time,x,label='Truth')
axs[0].plot(kal_time,kal_x,label='Estimated')
axs[0].plot(kal_time,kal_x+3*kalcov_x**0.5,color='green',label='3$\sigma$')
axs[0].plot(kal_time,kal_x-3*kalcov_x**0.5,color='green')
axs[0].axvspan(kal_time[0], kal_time[-1], color='grey', alpha=0.5, label='Collision Range')
axs[0].set_ylabel('x (m)')
axs[0].legend(fontsize="15")

axs[1].plot(time,y)
axs[1].plot(kal_time,kal_y)
axs[1].plot(kal_time,kal_y+3*kalcov_y**0.5,color='green')
axs[1].plot(kal_time,kal_y-3*kalcov_y**0.5,color='green')
axs[1].axvspan(kal_time[0], kal_time[-1], color='grey', alpha=0.5, label='Collision Range')
axs[1].set_ylabel('y (m)')

axs[2].plot(time,z)
axs[2].plot(kal_time,kal_z)
axs[2].plot(kal_time,kal_z+3*kalcov_z**0.5,color='green')
axs[2].plot(kal_time,kal_z-3*kalcov_z**0.5,color='green')
axs[2].axvspan(kal_time[0], kal_time[-1], color='grey', alpha=0.5, label='Collision Range')
axs[2].set_ylabel('z (m)')
axs[2].set_xlabel('time (s)')

fig, axs = plt.subplots(3, 1, sharex=True)
axs[0].plot(time,vx,label='Truth')
axs[0].plot(kal_time,kal_vx,label='Estimated')
axs[0].plot(kal_time,kal_vx+3*kalcov_vx**0.5+P.con_vel_uncertain,color='green',label='3$\sigma$')
axs[0].plot(kal_time,kal_vx-3*kalcov_vx**0.5-P.con_vel_uncertain,color='green')
axs[0].axvspan(kal_time[0], kal_time[-1], color='grey', alpha=0.5, label='Collision Range')
axs[0].set_ylabel('$\dot{x}$ (m/s)')
axs[0].legend(fontsize="15")

axs[1].plot(time,vy)
axs[1].plot(kal_time,kal_vy)
axs[1].plot(kal_time,kal_vy+3*kalcov_vy**0.5+P.con_vel_uncertain,color='green')
axs[1].plot(kal_time,kal_vy-3*kalcov_vy**0.5-P.con_vel_uncertain,color='green')
axs[1].axvspan(kal_time[0], kal_time[-1], color='grey', alpha=0.5, label='Collision Range')
axs[1].set_ylabel('$\dot{y}$ (m/s)')

axs[2].plot(time,vz)
axs[2].plot(kal_time,kal_vz)
axs[2].plot(kal_time,kal_vz+3*kalcov_vz**0.5+P.con_vel_uncertain,color='green')
axs[2].plot(kal_time,kal_vz-3*kalcov_vz**0.5-P.con_vel_uncertain,color='green')
axs[2].axvspan(kal_time[0], kal_time[-1], color='grey', alpha=0.5, label='Collision Range')
axs[2].set_ylabel('$\dot{z}$ (m/s)')
axs[2].set_xlabel('time (s)')

# print(kal_x)

plt.show()
