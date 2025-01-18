import numpy as np
import json
import matplotlib.pyplot as plt
# plt.rcParams['text.usetex'] = True

from mpl_toolkits.mplot3d import Axes3D
from IPython.core.debugger import set_trace
import params as P

from animation import Animation

data_pi_6 = []
data_pi_2 = []


with open("data/change_vel_pi_6.json", "r") as read_it:
# with open("superData/2dData1/run0quads18cr94.27mv9.21.json", "r") as read_it:
    data_pi_6 = json.load(read_it)

with open("data/change_vel_pi_2.json", "r") as read_it:
    # with open("superData/2dData1/run0quads18cr94.27mv9.21.json", "r") as read_it:
    data_pi_2 = json.load(read_it)


vx_pi_6 = np.array(data_pi_6['0'])[:,3]
vx_pi_2 = np.array(data_pi_2['0'])[:,3]
time = np.linspace(0,P.sim_t,vx_pi_6.shape[0])

plt.rcParams.update({'font.size': 20})

fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_prop_cycle(color = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'],
                  ls    = ["-","--","-.",":",(0, (1, 10))])
ax.plot(time,vx_pi_6,label='$\pi/6$')
ax.plot(time,vx_pi_2,label='$\pi/2$')
ax.legend()

ax.set_xlabel('time (s)')
ax.set_ylabel('x-velocity (m/s)')

plt.show()
