from styx_msgs.msg import TrafficLight
import cv2
import numpy as np
from keras.models import model_from_json
import rospy
import os


class TLClassifier(object):
    def __init__(self):
        self.c = 0
        rospy.logwarn(os.getcwd())
        rospy.logwarn(os.path.dirname(os.path.realpath(__file__)))
        with open('light_classification/model.json', 'r') as jfile:
            self.model = model_from_json(jfile.read())
        self.model.load_weights('light_classification/model.h5', by_name=True)

    def get_classification(self, image):
        """Determines the color of the traffic light in the image
        Args:
            image (cv::Mat): image containing the traffic light
        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)
        """

        self.c += 1

        resize_image = cv2.resize(image, (160, 80))
        hls_image = cv2.cvtColor(resize_image, cv2.COLOR_BGR2HLS)
        processed_image = (hls_image.astype(np.float32) / 255.0) + 0.01
        final_4d = np.expand_dims(processed_image, axis=0)

        network_label = self.model.predict_classes(final_4d)[0][0]

        # rospy.logwarn(a)
        # if network_label == 1:
        #     return 1
        # else:
        #     return 0

        # rospy.logwarn(a)
        if network_label == 1:
            return TrafficLight.GREEN
        else:
            return TrafficLight.RED


# light_classifier = TLClassifier()
#
# greenimage = os.path.abspath("D:\ANN\CarND-SS\classifier_images\image-308.jpg")
# redimage = os.path.abspath("D:\ANN\CarND-SS\classifier_images\image-1.jpg")
#
# greenimage = cv2.imread(greenimage)
# redimage = cv2.imread(redimage)
#
# classification_green = light_classifier.get_classification(greenimage)
# classification_red = light_classifier.get_classification(redimage)
#
# something = 1
