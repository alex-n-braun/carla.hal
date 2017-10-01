from styx_msgs.msg import TrafficLight
import cv2
import numpy as np
import rospy
import os
import tensorflow


class TLClassifier(object):
    def __init__(self, model):
        self.model = model
        self.model._make_predict_function()

    def get_classification(self, image):
        """Determines the color of the traffic light in the image
        Args:
            image (cv::Mat): image containing the traffic light
        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)
        """

        resize_image = np.array(cv2.resize(image, (75, 100)))
        resize_image = np.expand_dims(resize_image / 255., axis=0)
        network_label = np.argmax(self.model.predict(resize_image, batch_size=1)[0])
        #rospy.logwarn(self.model.predict(resize_image))
        if network_label == 0:
            return TrafficLight.RED
        else:
            return TrafficLight.GREEN

