from multiprocessing import Pool

import time
from IPython.core.debugger import set_trace

# work = (["A", 5], ["B", 2], ["C", 1], ["D", 3])

# work = {"A": 5, "B": 2, "C": 1, "D": 3}
# print('items: ', work.items())
work = [["A", 5], ["B", 2], ["C", 1], ["D", 3]]

def work_log(work_data):
    print(" Process %s waiting %s seconds" % (work_data[0], work_data[1]))
    time.sleep(int(work_data[1]))
    print(" Process %s Finished." % work_data[0])
    return work_data[1]


def pool_handler():
    p = Pool(2)
    # result = p.map(work_log, work.items())
    result = p.map(work_log, work)
    print('result: ', result)

if __name__ == '__main__':
    pool_handler()
