import sim

import numpy as np

num_quads = [2,3]
collision_range = [100, 10]
max_vel = [2,10]

for i in range(2):
    sim.run_sim(num_quads[i], collision_range[i], max_vel[i], 'run'+str(i)+'.json')
