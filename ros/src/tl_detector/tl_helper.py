from math import cos, sin

import numpy as np
import tf


def closest_node(node, nodes):
    """paralelize finding a single closest node using numpy.
    :param node -- numpy array with dimension rank 1 lower than nodes
    :param nodes -- numpy convertible array of nodes"""
    nodes = np.asarray(nodes)
    dist_2 = np.sum((nodes - node)**2, axis=1)
    return np.argmin(dist_2)


def is_waypoint_behind(pose, waypoint):
    """convert to car central reference frame using yaw from tf.transformations euler coordinate transformation

    :param pose --  car position PoseStamped.pose object
    :param waypoint -- waypoint Lane object

    :return bool -- True if the waypoint is behind the car False if in front

    Reference:    https://answers.ros.org/question/69754/quaternion-transformations-in-python/
    """
    roll, pitch, yaw = tf.transformations.euler_from_quaternion([pose.orientation.x,
                           pose.orientation.y, pose.orientation.z, pose.orientation.w])

    shift_x = waypoint.pose.pose.position.x - pose.position.x
    shift_y = waypoint.pose.pose.position.y - pose.position.y

    x = shift_x * cos(0 - yaw) - shift_y * sin(0 - yaw)

    if x > 0:
        return False
    return True
