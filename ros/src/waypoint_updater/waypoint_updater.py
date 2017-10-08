#!/usr/bin/env python

import rospy
from geometry_msgs.msg import PoseStamped, TwistStamped
from styx_msgs.msg import Lane, Waypoint, TrafficLight, TrafficLightArray
from std_msgs.msg import Bool
from std_msgs.msg import Int32

import math
from math import cos, sin, sqrt
import numpy as np
import tf
import os
import json
from keras.optimizers import Adadelta
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
#MAX_SPEED = 10.0 * MPH_TO_MPS #: Vehicle speed limit

BRAKE_DIST = 20.0 # meters; distance to brake from MAX_SPEED to 0
MIN_BRAKE_DIST = 2.0 # meters; min distance to brake from MAX_SPEED to 0; if closer, go ahead

LOOKAHEAD_WPS = 300 # Number of waypoints we will publish. You can change this number
#LOOKBACK_WPS = 10 # Number of waypoints to keep in the back for interpolation


class WaypointUpdater(object):
    def __init__(self):
        rospy.init_node('waypoint_updater')

        # TODO: Add a subscriber for /traffic_waypoint and /obstacle_waypoint below
        self.final_waypoints_pub = rospy.Publisher('final_waypoints', Lane, queue_size=1)

        # TODO: Add other member variables you need below
        self.dbw_enabled = False
        self.pose = None            # car current position to estimate respective forward waypoints
        self.lane = Lane()          # lane object, which we use to publish forward waypoints
        self.base_waypoints = None  # waypoints from the map
        self.frame_id = None
        self.next_wp = None
        self.next_traffic_light_index = -1
        self.curr_linear_velocity = 0
        
        self.brake_dist = BRAKE_DIST
        self.min_brake_dist = MIN_BRAKE_DIST
        
        self.stop={}
        
        self.MAX_SPEED = 0.0
        

        rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb, queue_size=1)
        rospy.Subscriber('/base_waypoints', Lane, self.waypoints_cb, queue_size=1)
        rospy.Subscriber('/traffic_waypoint', Int32, self.traffic_cb)
        rospy.Subscriber('/current_velocity', TwistStamped, self.current_velocity_cb, queue_size=1)
        rospy.Subscriber('/vehicle/dbw_enabled', Bool, self.dbw_enabled_cb, queue_size=1)
        
        #rospy.logwarn("Waypoint Updater initiated")
        self.loop()                 # launch infinite cycle to publish forward waypoints, based on current position
   
    def reload_params(self):
        fn=os.path.dirname(os.path.realpath(__file__))+"/waypointparams.json"
        with open(fn) as json_data:
            d = json.load(json_data)
            self.brake_dist = d['brake_dist']
            self.min_brake_dist = d['min_brake_dist']
        
   
    def loop(self):
        
        rate = rospy.Rate(35)       # setting refresh rate in Hz
        #self.reload_params()
        
        while not rospy.is_shutdown():
            
            MAX_SPEED = rospy.get_param('waypoint_loader/velocity')/3.6 # km/h to m/s
            if not MAX_SPEED == self.MAX_SPEED:
                self.MAX_SPEED = MAX_SPEED
                rospy.logwarn("set maximum velocity to "+str(MAX_SPEED))
            
            if (self.base_waypoints is not None) and (self.pose is not None) and (self.next_wp is not None):
                # [Dmitry, 11.09.2017] - let's determine where we are and where are we heading                
                # identify the closest waypoint in front of the car
                # get the next 200 waypoints ahead and fill in lane object before publishing                  
                self.lane.header.frame_id = self.frame_id
                self.lane.header.stamp = rospy.Time.now()
                                      
                # [Dmitry, 16.09.2017] setting the target speed for each forward waypoint
                # [Dmitry, 16.09.2017] now - just constant speed. Once traffic lights will be fixed, implement logic, 
                # based on distance to the next traffic light and its status
                
                start = int(self.next_wp)
                loop_length = len(self.base_waypoints)
                end_part = min(loop_length, start + LOOKAHEAD_WPS)
                start_part = (start + LOOKAHEAD_WPS) % loop_length

                if (self.next_traffic_light_index > -1) and ((start <= self.next_traffic_light_index <= end_part) or (self.next_traffic_light_index <= start_part)):
                    if not self.dbw_enabled:
                        self.stop={}
                        
                    if self.stop=={}:
                        self.stop['ego_dist_tl'] = self.distance(self.pose.position, 
                                                                 self.base_waypoints[self.next_traffic_light_index].pose.pose.position)
                        self.stop['speed_factor'] = self.curr_linear_velocity / self.MAX_SPEED
                        self.stop['clv'] = self.curr_linear_velocity
                        self.stop['stop'] = self.stop['ego_dist_tl'] >= self.min_brake_dist * self.stop['speed_factor']
                    
                    #ego_dist_tl = self.distance(self.pose.position, 
                    #                            self.base_waypoints[self.next_traffic_light_index].pose.pose.position)
                    ego_dist_tl = self.stop['ego_dist_tl']
                    
                    #rospy.logwarn("dist to tl: "+str(ego_dist_tl)+" "+str(self.min_brake_dist)+" "+str(self.brake_dist)+" "+str(self.curr_linear_velocity))
                    #speed_factor = self.curr_linear_velocity / MAX_SPEED
                    speed_factor = self.stop['speed_factor']
                    clv = self.stop['clv']
                    
                    if not self.stop['stop']: #ego_dist_tl < self.min_brake_dist*speed_factor: # too close to tl to stop
                        for i in range(LOOKAHEAD_WPS):
                            idx_wp = (start+i) % len(self.base_waypoints)
                            self.base_waypoints[idx_wp].twist.twist.linear.x = self.MAX_SPEED
                    elif ego_dist_tl < self.brake_dist*speed_factor:
                        for i in range(LOOKAHEAD_WPS):
                            idx_wp = (start+i) % len(self.base_waypoints)
                            dist_tl = self.distance(self.base_waypoints[idx_wp].pose.pose.position,
                                                    self.base_waypoints[self.next_traffic_light_index].pose.pose.position)
                            goal_speed = dist_tl / ego_dist_tl * clv
                            goal_speed = goal_speed if goal_speed>0.0 else 0.0
                            goal_speed = clv if goal_speed > clv else goal_speed
                            self.base_waypoints[idx_wp].twist.twist.linear.x = goal_speed
                        # check for increasing speed
                        for i in range(1, LOOKAHEAD_WPS):
                            idx_wp1 = (start+i) % len(self.base_waypoints)
                            idx_wp0 = (start+i-1) % len(self.base_waypoints)
                            vel1 = self.base_waypoints[idx_wp1].twist.twist.linear.x 
                            vel0 = self.base_waypoints[idx_wp0].twist.twist.linear.x 
                            if vel1>vel0:
                                self.base_waypoints[idx_wp1].twist.twist.linear.x = 0.0
                                self.base_waypoints[idx_wp0].twist.twist.linear.x = 0.0
                    else:
                        for i in range(LOOKAHEAD_WPS):
                            idx_wp = (start+i) % len(self.base_waypoints)
                            dist_tl = self.distance(self.base_waypoints[idx_wp].pose.pose.position,
                                                    self.base_waypoints[self.next_traffic_light_index].pose.pose.position)
                            if dist_tl > self.brake_dist*speed_factor:
                                goal_speed = self.MAX_SPEED
                                self.stop['speed_factor'] = self.curr_linear_velocity / self.MAX_SPEED
                                self.stop['clv'] = self.curr_linear_velocity
                            else:
                                goal_speed = dist_tl / self.brake_dist * clv
                                goal_speed = goal_speed if goal_speed > 0.0 else 0.0
                                goal_speed = clv if goal_speed > clv else goal_speed
                            self.base_waypoints[idx_wp].twist.twist.linear.x = goal_speed
                else:
                    self.stop={}
                    for i in range(LOOKAHEAD_WPS):
                        idx_wp = (start+i) % len(self.base_waypoints)
                        self.base_waypoints[idx_wp].twist.twist.linear.x = self.MAX_SPEED
                                                        
                # decrease speed to zero at the end of the track. 
                if start + LOOKAHEAD_WPS > loop_length:
                    #rospy.logwarn('I can see the end')
                    last_wp = self.base_waypoints[-1]
                    last_wp.twist.twist.linear.x = 0.
                    for idx_wp in range(start, loop_length):
                        dist = self.distance(self.base_waypoints[idx_wp].pose.pose.position, last_wp.pose.pose.position)
                        end_speed = (dist - 10.0) / self.brake_dist * self.MAX_SPEED
                        end_speed = max(0.0, min(self.base_waypoints[idx_wp].twist.twist.linear.x, end_speed))
                        self.base_waypoints[idx_wp].twist.twist.linear.x = end_speed
                                        
                # [Dmitry, 11.09.2017] publish forward waypoints
                # need to be careful at the end of the list of waypoints. Here, the list may end, and the lane will be empty.
                if (start + LOOKAHEAD_WPS > loop_length):
                    end_part = self.base_waypoints[start: loop_length]
                    start_part = self.base_waypoints[: LOOKAHEAD_WPS - (loop_length - start)]
                    self.lane.waypoints = end_part # + start_part
                else:
                    self.lane.waypoints = self.base_waypoints[start: start + LOOKAHEAD_WPS]
                
                self.final_waypoints_pub.publish(self.lane)
            
            #else:
                #rospy.logwarn("---- waypoint_updater: no data!! ----")
                
            rate.sleep()

    def dbw_enabled_cb(self, msg):

        if (msg.data == True):

            self.dbw_enabled = True
        else:
            self.dbw_enabled = False

            
    def pose_cb(self, msg):
        # TODO: Implement - pose callback will be called to get the next waypoints
        self.pose = msg.pose                    # store location (x, y)
        self.frame_id = msg.header.frame_id
        
        #current_time = rospy.get_rostime()
        
        if (self.base_waypoints) and (self.pose):
            next_wp = int(self.closest_waypoint(self.pose, self.base_waypoints))
            if next_wp!=self.next_wp:
                if (self.next_wp == None):
                    self.next_wp = next_wp
                if self.next_traffic_light_index>0:
                    disttl=self.distance(self.base_waypoints[next_wp].pose.pose.position, self.base_waypoints[self.next_traffic_light_index].pose.pose.position)
                else:
                    disttl=0.0
                rospy.logwarn("Next wp: "+str(next_wp)+", dist: "+str(self.distance(self.base_waypoints[next_wp].pose.pose.position, self.base_waypoints[self.next_wp].pose.pose.position))+" / "+str(disttl))
                self.next_wp = next_wp
        self.last_timestamp = rospy.get_rostime()
        
    def current_velocity_cb(self, message):
    #    """From the incoming message extract the velocity message """
        self.curr_linear_velocity = message.twist.linear.x
           
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
    
    
    def waypoints_cb(self, msg):
        # TODO: Implement
        # [Dmitry, 17.09.2017]
        if (self.base_waypoints != msg.waypoints):
            # if we got the new waypoints
            self.base_waypoints = msg.waypoints
            # set the next waypoint tracket to none. It will be identified with closest waypoint call
            self.next_wp = None

    def traffic_cb(self, msg):
        # TODO: Callback for /traffic_waypoint message. Implement
        if (self.pose is None):
            return
        
        next_traffic_light_index = int(msg.data)
        if next_traffic_light_index!=self.next_traffic_light_index:
            self.next_traffic_light_index = next_traffic_light_index
            rospy.logwarn("next red traffic light: "+str(self.next_traffic_light_index))
        

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