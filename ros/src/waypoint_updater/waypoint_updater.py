#!/usr/bin/env python

import rospy
from geometry_msgs.msg import PoseStamped
from styx_msgs.msg import Lane, Waypoint

import math
import numpy as np

'''
This node will publish waypoints from the car's current position to some `x` distance ahead.

As mentioned in the doc, you should ideally first implement a version which does not care
about traffic lights or obstacles.

Once you have created dbw_node, you will update this node to use the status of traffic lights too.

Please note that our simulator also provides the exact location of traffic lights and their
current status in `/vehicle/traffic_lights` message. You can use this message to build this node
as well as to verify your TL classifier.

TODO (for Yousuf and Aaron): Stopline location for each traffic light.
'''
MPH_TO_MPS = 0.44704
MAX_SPEED = 10.0 * MPH_TO_MPS #: Vehicle speed limit
LOOKAHEAD_WPS = 30 # Number of waypoints we will publish. You can change this number

LOOKBACK_WPS = 10 # Number of waypoints to keep in the back for interpolation


class WaypointUpdater(object):
    def __init__(self):
        rospy.init_node('waypoint_updater')

        rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb, queue_size=1)
        rospy.Subscriber('/base_waypoints', Lane, self.waypoints_cb, queue_size=1)
        #rospy.Subscriber('/traffic_waypoint', Int32, self.traffic_cb)
        #rospy.Subscriber('/obstacle_waypoints', PoseStamped, self.obstacle_cb)

        # TODO: Add a subscriber for /traffic_waypoint and /obstacle_waypoint below
        self.final_waypoints_pub = rospy.Publisher('final_waypoints', Lane, queue_size=1)

        # TODO: Add other member variables you need below
        self.pose = None    # variable for current pose
        self.lane = Lane()
        self.base_waypoints = None
        self.frame_id = None
        self.previous_car_waypoint = 0;
        
        #self.car_index_pub = rospy.Publisher('car_index', Int32, queue_size=1)

        rospy.spin()
            
    def pose_cb(self, msg):
        # TODO: Implement - pose callback will be called to get the next waypoints
        self.pose = msg.pose # store location (x, y)
        
        if (self.base_waypoints is not None) and (self.pose is not None):
    
            # [Dmitry, 11.09.2017] - let's determine where we are and where are we heading
            #rospy.logwarn("get to this point")
                
            #waypoints = self.base_waypoints.waypoints
                    
            self.lane.header.stamp = rospy.Time.now()
            start_point = self.closest_waypoint(self.pose, self.base_waypoints)
            
            rospy.logwarn(start_point)
            #all_waypoints = waypoints + waypoints[:LOOKAHEAD_WPS]
            self.lane.waypoints = self.base_waypoints[start_point: start_point + LOOKAHEAD_WPS]
            
            #wp = np.array(self.lane.waypoints)
            #rospy.logwarn("waypoints shape "+str(wp.shape))
    
                    
            for waypoint in self.lane.waypoints:
                waypoint.twist.twist.linear.x = MAX_SPEED
                    
            self.final_waypoints_pub.publish(self.lane)
        
        else:
            rospy.loginfo("---- no data!! ----")            

           
    def closest_waypoint(self, Pose, waypoints):
        # [Dmitry, 11.09.2017]
        closest_waypoint = 0
        #rospy.logwarn(Pose)
        #rospy.logwarn(len(waypoints))
        
        min_dist = 100000 #self.distance(Pose.position, waypoints[best_waypoint].pose.pose.position)
        
        for i, point in enumerate(waypoints):
            dist = self.distance(Pose.position, point.pose.pose.position)
            if dist < min_dist:
                closest_waypoint = i
                min_dist = dist
        #rospy.logwarn("best waypoint "+ str(closest_waypoint))                
        return closest_waypoint
    
    def distance(self, p1, p2):
#        dist = 0
#        dl = lambda a, b: math.sqrt((a.x-b.x)**2 + (a.y-b.y)**2  + (a.z-b.z)**2)
#        for i in range(wp1, wp2+1):
#            dist += dl(waypoints[wp1].pose.pose.position, waypoints[i].pose.pose.position)
#            wp1 = i
#        return dist
        # [Dmitry, 11.09.2017]
        dx = p1.x - p2.x
        dy = p1.y - p2.y
        return math.sqrt(dx*dx + dy*dy)
    
    
    def waypoints_cb(self, Lane):
        # TODO: Implement
        # [Dmitry, 11.09.2017]
        self.base_waypoints = Lane.waypoints

    def traffic_cb(self, msg):
        # TODO: Callback for /traffic_waypoint message. Implement
        pass

    def obstacle_cb(self, msg):
        # TODO: Callback for /obstacle_waypoint message. We will implement it later
        pass

    def get_waypoint_velocity(self, waypoint):
        return waypoint.twist.twist.linear.x

    def set_waypoint_velocity(self, waypoints, waypoint, velocity):
        waypoints[waypoint].twist.twist.linear.x = velocity



if __name__ == '__main__':
    try:
        WaypointUpdater()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start waypoint updater node.')
