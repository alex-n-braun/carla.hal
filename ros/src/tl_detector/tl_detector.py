#!/usr/bin/env python
import rospy
from std_msgs.msg import Int32
from geometry_msgs.msg import PoseStamped, Pose
from styx_msgs.msg import TrafficLightArray, TrafficLight
from styx_msgs.msg import Lane
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from light_classification.tl_classifier import TLClassifier
import tf
from tf.transformations import euler_from_quaternion
import math
import cv2
import numpy as np
from traffic_light_config import config

STATE_COUNT_THRESHOLD = 3
LIGHTGAP = 5 # number of waypoints between the traffic light and the stop line
LOOKAHEAD_WPS = 70 # number of wp as tl in sight


class TLDetector(object):
    def __init__(self):
        rospy.init_node('tl_detector')

        self.pose = None
        self.base_waypoints = None
        self.camera_image = None
        self.lights = []

        rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb)
        rospy.Subscriber('/base_waypoints', Lane, self.waypoints_cb)


        '''
        /vehicle/traffic_lights provides you with the location of the traffic light in 3D map space and 
        helps you acquire an accurate ground truth data source for the traffic light
        classifier by sending the current color state of all traffic lights in the
        simulator. When testing on the vehicle, the color state will not be available. You'll need to
        rely on the position of the light and the camera image to predict it.
        '''

        rospy.Subscriber('/vehicle/traffic_lights', TrafficLightArray, self.traffic_cb)
        rospy.Subscriber('/camera/image_raw', Image, self.image_cb)

        self.upcoming_red_light_pub = rospy.Publisher('/traffic_waypoint', Int32, queue_size=1)

        self.bridge = CvBridge()
        self.light_classifier = TLClassifier()
        self.listener = tf.TransformListener()

        self.state = TrafficLight.UNKNOWN
        self.last_state = TrafficLight.UNKNOWN
        self.last_wp = -1
        self.state_count = 0

        rospy.spin()

    def pose_cb(self, msg):
        self.pose = msg.pose

    def waypoints_cb(self, data):
        self.base_waypoints = data.waypoints

    def traffic_cb(self, msg):
        self.lights = msg.lights
        #rospy.logwarn(self.lights[0].state)


    def image_cb(self, msg):
        """Identifies red lights in the incoming camera image and publishes the index
            of the waypoint closest to the red light to /traffic_waypoint

        Args:
            msg (Image): image from car-mounted camera

        """
        self.has_image = True
        self.camera_image = msg
        light_wp, state = self.process_traffic_lights()

        '''
        Publish upcoming red lights at camera frequency.
        Each predicted state has to occur `STATE_COUNT_THRESHOLD` number
        of times till we start using it. Otherwise the previous stable state is
        used.
        '''
        if self.state != state:
            self.state_count = 0
            self.state = state
        elif self.state_count >= STATE_COUNT_THRESHOLD:
            self.last_state = self.state
            light_wp = light_wp if state == TrafficLight.RED else -1
            self.last_wp = light_wp
            self.upcoming_red_light_pub.publish(Int32(light_wp))
            #rospy.logwarn(state)
            #rospy.logwarn(light_wp)
        else:
            self.upcoming_red_light_pub.publish(Int32(self.last_wp))
            #rospy.logwarn(self.last_wp)
        self.state_count += 1


    def project_to_image_plane(self, point_in_world):
        """Project point from 3D world coordinates to 2D camera image location

        Args:
            point_in_world (Point): 3D location of a point in the world

        Returns:
            x (int): x coordinate of target point in image
            y (int): y coordinate of target point in image

        """

        # get transform between pose of camera and world frame
        trans = None
        try:
            now = rospy.Time.now()
            self.listener.waitForTransform("/base_link",
                  "/world", now, rospy.Duration(1.0))
            (trans, rot) = self.listener.lookupTransform("/base_link",
                  "/world", now)

        except (tf.Exception, tf.LookupException, tf.ConnectivityException):
            rospy.logerr("Failed to find camera to map transform")

        #TODO Use tranform and rotation to calculate 2D position of light in image
        fx = config.camera_info.focal_length_x
        fy = config.camera_info.focal_length_y
        cx = config.camera_info.image_width / 2
        cy = config.camera_info.image_height / 2
        cameraMatrix = np.array([[fx, 0, cx],
                                 [0, fy, cy],
                                 [0,  0,  1]])
        x = 0
        y = 0
        #rvec = euler_from_quaternion(rot)
        #tvec = np.array(trans)
        #points3d = np.array([point_in_world.x, point_in_world.y, point_in_world.z])
        #rospy.logwarn(points3d)
        #(x, y), _ = cv2.projectPoints(points3d, rvec, tvec, cameraMatrix, None)
        
        return (x, y)

    def get_light_state(self, light):
        """Determines the current color of the traffic light

        Args:
            light (TrafficLight): light to classify

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        if(not self.has_image):
            self.prev_light_loc = None
            return False

        self.camera_image.encoding = "rgb8"
        cv_image = self.bridge.imgmsg_to_cv2(self.camera_image, "bgr8")

        #x, y = self.project_to_image_plane(light.pose.pose.position)
        #rospy.logwarn((x,y))
        #TODO use light location to zoom in on traffic light in image

        #Get classification
        return self.light_classifier.get_classification(cv_image)

    def process_traffic_lights(self):
        """Finds closest visible traffic light, if one exists, and determines its
            location and color

        Returns:
            int: index of waypoint closes to the upcoming traffic light (-1 if none exists)
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """

        # find the waypoint index that is closest to the car
        car_wp_idx = None
        if(self.pose and self.base_waypoints != None):
            car_wp_idx = self.get_closest_waypoint(self.pose)
            #rospy.logwarn(car_wp_idx)

        # find the closest visible traffic light (if one exists)
        light = None # TrafficLight object
        light_wp_idx = None
        tl_waypoint_indices = None
        light_positions = config.light_positions
        if self.base_waypoints != None and car_wp_idx != None:
            tl_waypoint_indices = self.get_traffic_light_wp_index(light_positions)
            for i, tl_wp_idx in enumerate(tl_waypoint_indices):
                idx_diff = tl_wp_idx - car_wp_idx
                # traffic light is ahead of the car within number of LOOKAHEAD_WPS
                if idx_diff >= 0 and idx_diff <= LOOKAHEAD_WPS:
                    # minus LIGHTGAP so that the car stops near the stop line
                    light_wp_idx = tl_wp_idx - LIGHTGAP
                    light = self.lights[i]
                    #rospy.logwarn(light_wp_idx)
                    #rospy.logwarn(tl_waypoint_indices)

        if light:
            #state = self.get_light_state(light)
            state = light.state
            return light_wp_idx, state
        self.pose = None
        self.base_waypoints = None
        return -1, TrafficLight.UNKNOWN

    def get_traffic_light_wp_index(self, light_positions):
        indices = []
        for pos in light_positions:
            pose = Pose()
            pose.position.x = pos[0]
            pose.position.y = pos[1]
            indices.append(self.get_closest_waypoint(pose))
        return indices

    def get_closest_waypoint(self, pose):
        """Identifies the closest path waypoint to the given position
            https://en.wikipedia.org/wiki/Closest_pair_of_points_problem
        Args:
            pose (Pose): position to match a waypoint to

        Returns:
            int: index of the closest waypoint in self.base_waypoints

        """
        dist = float('inf') # a very large number
        nearest_idx = None
        for idx,waypoint in enumerate(self.base_waypoints):
            temp_dist = self.distance(waypoint, pose)
            #rospy.logwarn(temp_dist)
            if dist > temp_dist:
                dist = temp_dist
                nearest_idx = idx
        return nearest_idx


    def distance(self, waypoint, pose):
        dl = lambda a, b: math.sqrt((a.x-b.x)**2 + (a.y-b.y)**2)
        return dl(waypoint.pose.pose.position, pose.position)


if __name__ == '__main__':
    try:
        TLDetector()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start traffic node.')
