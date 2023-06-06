#import yaml
import numpy as np
from numpy.linalg import norm

from IPython.core.debugger import set_trace

class WPManager:
    def __init__(self, params, id, loaded_wps, max_vel):
        self.id = id
        self.waypoints_ = np.asarray(loaded_wps)
        self.waypoints_ = np.reshape(self.waypoints_, (-1,4))
        self.current_waypoint_id_ = 0
        self.num_waypoints_ = self.waypoints_.shape[0]



        self.waypoint_threshold_ = params.waypoint_threshold
        self.waypoint_velocity_threshold_ = params.waypoint_velocity_threshold
        self.max_vel_ = max_vel

        self.K_p_ = params.Kp
        self.K_p_ = np.asarray(self.K_p_)
        self.K_p_ = np.reshape(self.K_p_, (3,1))



    def updateWaypointManager(self, position):
        current_waypoint = self.waypoints_[self.current_waypoint_id_]
        current_waypoint = np.reshape(current_waypoint, (4,1))
        error = current_waypoint[0:3] - position

        # if(norm(error) < self.waypoint_threshold_):
        #     # if self.current_waypoint_id_ < self.num_waypoints_
        #     self.current_waypoint_id_ = (self.current_waypoint_id_ + 1) % self.num_waypoints_
        #     # current_waypoint = self.waypoints_[current_waypoint_id_]
        #     # current_waypoint = np.reshape(current_waypoint, (4,1))

        desiredVelocity = np.multiply(self.K_p_, error)

        if(norm(desiredVelocity) > self.max_vel_):
            desiredVelocity = desiredVelocity / norm(desiredVelocity) * self.max_vel_
        # set_trace()

        return desiredVelocity
