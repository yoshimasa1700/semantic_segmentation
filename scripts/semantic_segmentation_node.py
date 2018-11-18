#!/usr/bin/env python

import time

import rospy
import numpy as np
import cv2

from deeplab_cityscapes.deeplab_model import DeeplabModel
from deeplab_cityscapes.deeplab_model import label_to_color_image

from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError


class SemanticSegmentationNode():
    def __init__(self, weight_path, publish_debug_image=False):
        self.model = DeeplabModel(weight_path)
        self.bridge = CvBridge()

        queue_size = 1

        self.subscriber = rospy.Subscriber(
            "image", Image, self.image_callback, queue_size=queue_size)

        self.pub_label = rospy.Publisher(
            "label", Image, queue_size=queue_size)

        self.publish_debug_image = publish_debug_image
        if self.publish_debug_image:
            self.pub_debug_image = rospy.Publisher(
                "debug_image", Image, queue_size=queue_size)

    def image_callback(self, image_msg):
        try:
            image = self.bridge.imgmsg_to_cv2(image_msg, "bgr8")

            start = time.time()
            _, segmap = self.model.inference(image)

            elapsed_time = time.time() - start
            print ("elapsed_time:{0}".format(elapsed_time) + "[sec]")

            segmap_int8 = segmap.astype(np.uint8)
            segmap_msg = self.bridge.cv2_to_imgmsg(segmap_int8, "mono8")
            self.pub_label.publish(segmap_msg)

            if self.publish_debug_image:
                debug_image = label_to_color_image(segmap).astype(np.uint8)
                debug_image_msg = self.bridge.cv2_to_imgmsg(
                    debug_image, "bgr8")
                self.pub_debug_image.publish(debug_image_msg)
        except CvBridgeError as e:
            print(e)


def main():
    rospy.init_node('semantic_segmentation', anonymous=True)

    weight_path = rospy.get_param("~weight_path")
    publish_debug_image = rospy.get_param("~publish_debug_image", True)

    SemanticSegmentationNode(weight_path, publish_debug_image)

    rospy.spin()


if __name__ == '__main__':
    main()
