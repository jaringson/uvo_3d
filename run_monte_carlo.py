import sim

import numpy as np
from importlib import reload

from IPython.core.debugger import set_trace

nq = [3,2]

def trunc(values, decs=2):
    return np.trunc(values*10**decs)/(10**decs)

for i in range(1000):
    num_quads = nq[i] #np.random.randint(10,20)
    collision_range = np.random.random() * 100
    max_vel = 5.0
    outfile = 'data/run'+str(i)+'quads'+str(num_quads)+'cr'+str(trunc(collision_range))+'.json'
    # current_sim = Sim()
    sim.run_sim(num_quads, collision_range, max_vel, outfile)

    reload(sim)
    # set_trace()
