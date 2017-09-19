#!/usr/bin/env python

import rospy
from geometry_msgs.msg import PoseStamped
from styx_msgs.msg import Lane, Waypoint, TrafficLight, TrafficLightArray
from std_msgs.msg import Int32

import math
from math import cos, sin, sqrt
import numpy as np
import tf
#from builtins import None

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
#LOOKBACK_WPS = 10 # Number of waypoints to keep in the back for interpolation


class WaypointUpdater(object):
    def __init__(self):
        rospy.init_node('waypoint_updater')

        rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb, queue_size=1)
        rospy.Subscriber('/base_waypoints', Lane, self.waypoints_cb, queue_size=1)
        #rospy.Subscriber('/traffic_waypoint', Int32, self.traffic_cb)
        rospy.Subscriber('/vehicle/traffic_lights', TrafficLightArray, self.traffic_cb)
        #rospy.Subscriber('/obstacle_waypoints', PoseStamped, self.obstacle_cb)

        # TODO: Add a subscriber for /traffic_waypoint and /obstacle_waypoint below
        self.final_waypoints_pub = rospy.Publisher('final_waypoints', Lane, queue_size=1)

        # TODO: Add other member variables you need below
        self.pose = None            # car current position to estimate respective forward waypoints
        self.lane = Lane()          # lane object, which we use to publish forward waypoints
        self.base_waypoints = None  # waypoints from the map
        self.frame_id = None
        self.next_traffic_light = None
        #self.next_traffic_light_wp = None
        self.next_wp = None
        self.last_timestamp = None
        
        rospy.logwarn("Waypoint Updater initiated")
        self.loop()                 # launch infinite cycle to publish forward waypoints, based on current position
        
    def loop(self):
        
        rate = rospy.Rate(30)       # setting refresh rate in Mhz
        
        while not rospy.is_shutdown():
                    
            if (self.base_waypoints is not None) and (self.pose is not None) and (self.next_wp is not None):
                # [Dmitry, 11.09.2017] - let's determine where we are and where are we heading                
                # identify the closest waypoint in front of the car
                # get the next 200 waypoints ahead and fill in lane object before publishing
                self.lane.waypoints = self.base_waypoints[self.next_wp: self.next_wp + LOOKAHEAD_WPS]
                self.lane.header.frame_id = self.frame_id
                self.lane.header.stamp = rospy.Time.now()
                                      
                # [Dmitry, 16.09.2017] setting the target speed for each forward waypoint
                # [Dmitry, 16.09.2017] now - just constant speed. Once traffic lights will be fixed, implement logic, 
                # based on distance to the next traffic light and its status

                target_speed = MAX_SPEED

                if(self.next_traffic_light is not None):
                    next_traffic_light_wp = int(self.closest_waypoint(self.next_traffic_light.pose.pose, self.lane.waypoints)) 
                    
                    #rospy.logwarn(next_traffic_light_wp)
                    
                    distance_to_tl = self.distance(self.lane.waypoints[next_traffic_light_wp].pose.pose.position, self.pose.position)
                    
                    if(distance_to_tl < 50) and ((self.next_traffic_light.state == 0) or
                                                (self.next_traffic_light.state == 1)):
                        target_speed = MAX_SPEED / 5.0 
                    if(distance_to_tl < 30) and ((self.next_traffic_light.state == 0) or
                                                (self.next_traffic_light.state == 1)):
                        target_speed = 0 
                    #rospy.logwarn("c_x = " + str(self.pose.position.x) + ", l_x = " + str(self.next_traffic_light.pose.pose.position.x) + ", "+ str(self.next_traffic_light.state))
                else:
                    distance_to_tl = 100               
                
                
                for waypoint in self.lane.waypoints:
                    # UPDATE WITH TRAFFIC LIGHT LOGIC
                    waypoint.twist.twist.linear.x = target_speed
                    #index = (index + 1)
                        
                # [Dmitry, 11.09.2017] publish forward waypoints
                self.final_waypoints_pub.publish(self.lane)
            
            else:
                rospy.loginfo("---- no data!! ----")
                
            rate.sleep()
            
    def pose_cb(self, msg):
        # TODO: Implement - pose callback will be called to get the next waypoints
        self.pose = msg.pose                    # store location (x, y)
        self.frame_id = msg.header.frame_id
        
        current_time = rospy.get_rostime()
        
        if (self.base_waypoints) and (self.pose):
            self.next_wp = int(self.closest_waypoint(self.pose, self.base_waypoints))
        self.last_timestamp = rospy.get_rostime()
           
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
        # [Dmitry, 17.09.2017]
        if (self.base_waypoints != Lane.waypoints):
            # if we got the new waypoints
            self.base_waypoints = Lane.waypoints
            # set the next waypoint tracket to none. It will be identified with closest waypoint call
            self.next_wp = None

    def traffic_cb(self, msg):
        # TODO: Callback for /traffic_waypoint message. Implement
        if (self.pose is None):
            return
        
        next_traffic_light_index = self.closest_waypoint(self.pose, msg.lights)
        
        self.next_traffic_light = msg.lights[next_traffic_light_index]
        
        #rospy.logwarn("next traffic line = "+str(self.next_traffic_light.pose.pose))
        
        #if (self.stop_line_index > -1):
            #self.set_stop_trajectory(self.next_wp, self.stop_line_index)
            #rospy.logerr("stop_trajectory: %s\n ", self.get_vels())
        #else:  
            #self.stop_trajectory = None       

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
    
        x = shift_x * cos(yaw) + shift_y * sin(yaw)
    
        if x >= 0:
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