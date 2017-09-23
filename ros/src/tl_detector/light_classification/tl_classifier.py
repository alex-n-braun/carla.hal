from styx_msgs.msg import TrafficLight
import cv2
import numpy as np
from keras.models import model_from_json
import rospy
import os
import tensorflow


class TLClassifier(object):
    def __init__(self):
        with open('model.json', 'r') as jfile:
            self.model = model_from_json(jfile.read())
        self.model.load_weights('model.h5', by_name=True)
        self.model._make_predict_function()
        self.graph = tensorflow.get_default_graph()

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

        # rospy.logwarn(network_label)
        if network_label == 1:
            return TrafficLight.GREEN
        else:
            return TrafficLight.RED
