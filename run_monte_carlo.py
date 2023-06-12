import sim
import params

import numpy as np
from importlib import reload


from IPython.core.debugger import set_trace

nq = [3,2]
cr = [1,2]

def trunc(values, decs=2):
    return np.trunc(values*10**decs)/(10**decs)

#def run_monte_carlo(start, end=1000):

for i in range(0, 1000):
    num_quads = np.random.randint(10,40)
    if params.is_2d:
        num_quads = np.random.randint(10,20)
    collision_range = np.random.random() * 100
    max_vel = 5.0 + np.random.random() * 5.0
    outfile = 'hardData5/run'+str(i)+'quads'+str(num_quads)+'cr'+str(trunc(collision_range))+'mv'+str(trunc(max_vel))+'.json'
    # current_sim = Sim()
    print(outfile)
    sim.run_sim(num_quads, collision_range, max_vel, outfile)

    reload(sim)
    # set_trace()

#if __name__ == "__main__":
#	run_monte_carlo(arg[0])
