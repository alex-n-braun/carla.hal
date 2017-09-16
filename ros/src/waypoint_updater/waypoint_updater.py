#!/usr/bin/env python

import rospy
from geometry_msgs.msg import PoseStamped
from styx_msgs.msg import Lane, Waypoint

import math
from math import cos, sin, sqrt
import numpy as np
import tf

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

LOOKAHEAD_WPS = 200 # Number of waypoints we will publish. You can change this number
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
        self.pose = None            # car current position to estimate respective forward waypoints
        self.lane = Lane()          # lane object, which we use to publish forward waypoints
        self.base_waypoints = None  # waypoints from the map
        self.frame_id = None        
        
        rospy.logwarn("Waypoint Updater initiated")
        self.loop()                 # launch infinite cycle to publish forward waypoints, based on current position
        
    def loop(self):
        
        rate = rospy.Rate(50)       # setting refresh rate in Mhz

        while not rospy.is_shutdown():
            
            rate.sleep()
        
            if (self.base_waypoints is not None) and (self.pose is not None):
                # [Dmitry, 11.09.2017] - let's determine where we are and where are we heading                
                
                # identify the closest waypoint in front of the car
                start_point = self.closest_waypoint(self.pose, self.base_waypoints)
                
                # get the next 200 waypoints ahead and fill in lane object before publishing
                self.lane.waypoints = self.base_waypoints[start_point: start_point + LOOKAHEAD_WPS]
                self.lane.header.frame_id = self.frame_id
                self.lane.header.stamp = rospy.Time.now()
                                        
                # [Dmitry, 16.09.2017] setting the target speed for each forward waypoint
                # [Dmitry, 16.09.2017] now - just constant speed. Once traffic lights will be fixed, implement logic, 
                # based on distance to the next trafic light and its status
                for waypoint in self.lane.waypoints:
                    # UPDATE WITH TRAFFIC LIGHT LOGIC
                    waypoint.twist.twist.linear.x = MAX_SPEED
                        
                # [Dmitry, 11.09.2017] publish forward waypoints
                self.final_waypoints_pub.publish(self.lane)
            
            else:
                rospy.loginfo("---- no data!! ----")
            
    def pose_cb(self, msg):
        # TODO: Implement - pose callback will be called to get the next waypoints
        self.pose = msg.pose                    # store location (x, y)
        self.frame_id = msg.header.frame_id
           
    def closest_waypoint(self, Pose, waypoints):
        # [Dmitry, 11.09.2017]
        closest_waypoint = 0
        min_dist = float('inf')
        
        for i, point in enumerate(waypoints):
            dist = self.distance(Pose.position, point.pose.pose.position)
            if dist < min_dist:
                closest_waypoint = i
                min_dist = dist
        
        is_behind = self.is_waypoint_behind(Pose, waypoints[closest_waypoint])
        if is_behind:
            closest_waypoint += 1
        
        return closest_waypoint
    
    def distance(self, p1, p2):
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
                
 
    def is_waypoint_behind(self, pose, waypoint):
        yaw = self.get_Euler(pose)
        X = pose.position.x
        Y = pose.position.y
    
        shift_x = waypoint.pose.pose.position.x - X
        shift_y = waypoint.pose.pose.position.y - Y
    
        x = shift_x * cos(0 - yaw) - shift_y * sin(0 - yaw)
    
        if x > 0:
            return False
        return True

    def get_Euler(self, pose):
        # Returns the roll, pitch yaw angles from a Quaternion \
        _, _, yaw = tf.transformations.euler_from_quaternion([pose.orientation.x,
                                                         pose.orientation.y,
                                                         pose.orientation.z,
                                                         pose.orientation.w])
        return yaw


if __name__ == '__main__':
    try:
        WaypointUpdater()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start waypoint updater node.')
