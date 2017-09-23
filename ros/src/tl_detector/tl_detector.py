#!/usr/bin/env python
import rospy
from std_msgs.msg import Int32
from geometry_msgs.msg import PoseStamped, Pose
from styx_msgs.msg import TrafficLightArray, TrafficLight
from styx_msgs.msg import Lane
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import tf
import cv2
import yaml
import math
import numpy as np
from math import sin, cos
import keras
from keras.models import model_from_json
import os
import tensorflow

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
        with open('model.json', 'r') as jfile:
            self.model = model_from_json(jfile.read())
        self.model.load_weights('model.h5', by_name=True)
        self.model._make_predict_function()
        self.graph = tensorflow.get_default_graph()

        self.has_image = False

        sub1 = rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb)
        sub2 = rospy.Subscriber('/base_waypoints', Lane, self.waypoints_cb)

        '''
        /vehicle/traffic_lights provides you with the location of the traffic light in 3D map space and 
        helps you acquire an accurate ground truth data source for the traffic light
        classifier by sending the current color state of all traffic lights in the
        simulator. When testing on the vehicle, the color state will not be available. You'll need to
        rely on the position of the light and the camera image to predict it.
        '''
        sub3 = rospy.Subscriber('/vehicle/traffic_lights', TrafficLightArray, self.traffic_cb)
        sub6 = rospy.Subscriber('/image_color', Image, self.image_cb)

        config_string = rospy.get_param("/traffic_light_config")
        self.config = yaml.load(config_string)

        self.upcoming_red_light_pub = rospy.Publisher('/traffic_waypoint', Int32, queue_size=1)

        self.bridge = CvBridge()
        self.listener = tf.TransformListener()

        self.state = TrafficLight.UNKNOWN
        self.last_state = TrafficLight.UNKNOWN
        self.last_wp = -1
        self.state_count = 0

        rospy.spin()

    def pose_cb(self, msg):
        self.pose = msg.pose

    def waypoints_cb(self, data): # is only called once
        self.base_waypoints = data.waypoints

    def traffic_cb(self, msg):  # call-back for cheating, as long as we dont have a classifier
        self.lights = msg.lights

    def image_cb(self, msg):
        """Identifies red lights in the incoming camera image and publishes the index
            of the waypoint closest to the red light's stop line to /traffic_waypoint

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
        else:
            self.upcoming_red_light_pub.publish(Int32(self.last_wp))
        self.state_count += 1

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
            is_behind = self.is_waypoint_behind(pose, self.base_waypoints[nearest_idx])
            if is_behind:
                nearest_idx += 1
        return nearest_idx

    def distance(self, waypoint, pose):
        dl = lambda a, b: math.sqrt((a.x-b.x)**2 + (a.y-b.y)**2)
        return dl(waypoint.pose.pose.position, pose.position)

    def project_to_image_plane(self, point_in_world):
        """Project point from 3D world coordinates to 2D camera image location

        Args:
            point_in_world (Point): 3D location of a point in the world

        Returns:
            x (int): x coordinate of target point in image
            y (int): y coordinate of target point in image

        """

        fx = self.config['camera_info']['focal_length_x']
        fy = self.config['camera_info']['focal_length_y']
        image_width = self.config['camera_info']['image_width']
        image_height = self.config['camera_info']['image_height']

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

        x = 0
        y = 0

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

        cv_image = self.bridge.imgmsg_to_cv2(self.camera_image, "bgr8")

        #Get classification
        network_label = self.get_classification(cv_image)
        rospy.logwarn('%s', network_label)
        if network_label >= 0.95:
            return TrafficLight.GREEN
        else:
            return TrafficLight.RED

    def process_traffic_lights(self):
        """Finds closest visible traffic light, if one exists, and determines its
            location and color

        Returns:
            int: index of waypoint closes to the upcoming stop line for a traffic light (-1 if none exists)
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        light = None

        # List of positions that correspond to the line to stop in front of for a given intersection
        stop_line_positions = self.config['stop_line_positions']
        if self.pose and self.base_waypoints:
            car_position = self.pose
            closest = self.closest_node(stop_line_positions, np.array(
                (car_position.position.x, car_position.position.y)))

            rospy.logwarn('{}'.format(abs(self.lights[closest].pose.pose.position.x - car_position.position.x) +
                    abs(self.lights[closest].pose.pose.position.y - car_position.position.y)))
            if 5 < (abs(self.lights[closest].pose.pose.position.x - car_position.position.x) +
                    abs(self.lights[closest].pose.pose.position.y - car_position.position.y)) < 80:

                light_wp = self.get_closest_waypoint(self.lights[closest].pose.pose)

                if self.base_waypoints and len(self.base_waypoints) > closest:
                    light_wp = self.base_waypoints[closest]
                    light = self.lights[closest]

        if light:
            state = self.get_light_state(light)
            if state is False:
                self.base_waypoints = None
                return -1, TrafficLight.UNKNOWN
            return light_wp, state
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

    def closest_node(self, node, nodes):
        """paralelize finding a single closest node using numpy.
        :param node -- numpy array with dimension rank 1 lower than nodes
        :param nodes -- numpy convertible array of nodes"""
        nodes = np.asarray(nodes)
        dist_2 = np.sum((nodes - node) ** 2, axis=1)
        return np.argmin(dist_2)

    def is_waypoint_behind(self, pose, waypoint):
        """convert to car central reference frame using yaw from tf.transformations euler coordinate transformation

        :param pose --  car position PoseStamped.pose object
        :param waypoint -- waypoint Lane object

        :return bool -- True if the waypoint is behind the car False if in front

        Reference:    https://answers.ros.org/question/69754/quaternion-transformations-in-python/
        """
        roll, pitch, yaw = tf.transformations.euler_from_quaternion([pose.orientation.x, pose.orientation.y,
                                                  pose.orientation.z, pose.orientation.w])

        shift_x = waypoint.pose.pose.position.x - pose.position.x
        shift_y = waypoint.pose.pose.position.y - pose.position.y

        x = shift_x * cos(0 - yaw) - shift_y * sin(0 - yaw)

        if x > 0:
            return False
        return True

    def get_classification(self, image):
        """Determines the color of the traffic light in the image
        Args:
            image (cv::Mat): image containing the traffic light
        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)
        """

        resize_image = cv2.resize(image, (160, 80))
        hls_image = cv2.cvtColor(resize_image, cv2.COLOR_BGR2HLS)
        processed_image = (hls_image.astype(np.float32) / 255.0) + 0.01
        final_4d = np.expand_dims(processed_image, axis=0)

        network_label = self.model.predict_classes(final_4d)[0][0]
        return network_label


if __name__ == '__main__':
    try:
        TLDetector()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start traffic node.')
