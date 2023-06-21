import numpy as np
from numpy.linalg import inv, norm

from IPython.core.debugger import set_trace

M_PI = 3.14159265359

def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / norm(vector)

def angle_between(v1, v2):
    """ Returns the angle in radians between vectors 'v1' and 'v2'::

            >>> angle_between((1, 0, 0), (0, 1, 0))
            1.5707963267948966
            >>> angle_between((1, 0, 0), (1, 0, 0))
            0.0
            >>> angle_between((1, 0, 0), (-1, 0, 0))
            3.141592653589793
    """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))

def from_euler(roll, pitch, yaw):
    cp = np.cos(roll/2.0)
    ct = np.cos(pitch/2.0)
    cs = np.cos(yaw/2.0)
    sp = np.sin(roll/2.0)
    st = np.sin(pitch/2.0)
    ss = np.sin(yaw/2.0)

    q = np.array([[cp*ct*cs + sp*st*ss],
                  [sp*ct*cs - cp*st*ss],
                  [cp*st*cs + sp*ct*ss],
                  [cp*ct*ss - sp*st*cs]])
    return q


def roll(quat):
    w = quat[0,0]
    x = quat[1,0]
    y = quat[2,0]
    z = quat[3,0]
    ret_val = np.arctan2(2.0*(w*x + y*z), 1.0 - 2.0*(x*x + y*y))
    # if np.abs(ret_val) > 1e-9:
    #     set_trace()
    return ret_val

def pitch(quat):
    w = quat[0,0]
    x = quat[1,0]
    y = quat[2,0]
    z = quat[3,0]
    val = 2.0 * (w*y - x*z)

    # hold at 90 degrees if invalid
    if np.abs(val) > 1.0:
        return np.copysign(1.0, val) * M_PI / 2.0
    else:
        return np.arcsin(val)

def yaw(quat):
    w = quat[0,0]
    x = quat[1,0]
    y = quat[2,0]
    z = quat[3,0]
    return np.arctan2(2.0*(w*z + x*y), 1.0 - 2.0*(y*y + z*z))


def rotp(quat, v):
    w = quat[0]
    bar = np.array([ [quat[1,0]], [quat[2,0]], [quat[3,0]] ])
    t = 2.0 * np.cross(v.T, bar.T)
    ret = v + w*t.T + np.cross(t, bar.T).T
    # set_trace()
    return ret


def rota(quat, v):
    w = quat[0]
    bar = np.array([ [quat[1,0]], [quat[2,0]], [quat[3,0]] ])
    t = 2.0 * np.cross(v.T, bar.T)
    ret = v - w*t.T + np.cross(t, bar.T).T
    return ret

def exp(v):
    norm_v = norm(v)

    q = []
    if norm_v > 1e-4:
        v_scale = np.sin(norm_v/2.0)/norm_v
        q = np.block([[np.cos(norm_v/2.0)],
                    [v_scale*v]])
    else:
        q = np.block([[1.0],
                    [v/2.0]])
        q = q / norm(q)
    return q

def otimes(quat, q):
    w1 = quat[0,0]
    x1 = quat[1,0]
    y1 = quat[2,0]
    z1 = quat[3,0]

    w2 = q[0,0]
    x2 = q[1,0]
    y2 = q[2,0]
    z2 = q[3,0]

    qout = np.array([[w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2],
                     [w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2],
                     [w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2],
                     [w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2]])
    return qout

def boxplus(quat, delta):
    # set_trace()
    return otimes(quat, exp(delta))


def R_v2_to_b(phi):

    R_v22b = np.array([[1,         0,        0],
                [0,  np.cos(phi), np.sin(phi)],
                [0, -np.sin(phi), np.cos(phi)]])
    return R_v22b

def R_v1_to_v2(theta):
    R_v12v2 = np.array([[np.cos(theta), 0, -np.sin(theta)],
                      [0, 1,           0],
                      [np.sin(theta), 0,  np.cos(theta)]])
    return R_v12v2

def R_v_to_v1(psi):
    R_v2v1 = np.array([[np.cos(psi), np.sin(psi), 0],
                       [-np.sin(psi), np.cos(psi), 0],
                       [0,        0, 1]])
    return R_v2v1

def R_v_to_b(phi, theta, psi):
    return R_v2_to_b(phi) * R_v1_to_v2(theta) * R_v_to_v1(psi)
