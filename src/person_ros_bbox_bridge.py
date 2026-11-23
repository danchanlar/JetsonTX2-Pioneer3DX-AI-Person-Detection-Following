#!/usr/bin/env python
from __future__ import print_function

import json
import socket
import time
import math

import rospy

from geometry_msgs.msg import PointStamped
from std_msgs.msg import Bool, Float32MultiArray


UDP_IP = "127.0.0.1"
UDP_PORT = 5005

MAX_AGE = 0.5


def finite(value):
    return not (math.isnan(value) or math.isinf(value))


def main():

    rospy.init_node("person_ros_bbox_bridge")

    center_pub = rospy.Publisher(
        "/person/center",
        PointStamped,
        queue_size=1
    )

    visible_pub = rospy.Publisher(
        "/person/visible",
        Bool,
        queue_size=1
    )

    bbox_pub = rospy.Publisher(
        "/person/bbox",
        Float32MultiArray,
        queue_size=1
    )

    sock = socket.socket(
        socket.AF_INET,
        socket.SOCK_DGRAM
    )

    sock.bind((UDP_IP, UDP_PORT))
    sock.settimeout(0.2)

    rospy.loginfo(
        "Bounding Box ROS Bridge listening on UDP 5005"
    )

    try:
        while not rospy.is_shutdown():

            try:

                data, address = sock.recvfrom(4096)

                msg = json.loads(
                    data.decode("utf-8")
                )

                age = time.time() - float(
                    msg["timestamp"]
                )

                valid = (
                    -0.2 <= age <= MAX_AGE
                )

                detected = (
                    valid and
                    msg["detected"] and
                    msg["bbox"] is not None and
                    int(msg["width"]) == 1280 and
                    int(msg["height"]) == 720
                )

                bbox = None

                if detected:

                    bbox = [
                        float(v) for v in msg["bbox"]
                    ]

                    detected = (
                        len(bbox) == 9 and
                        all(finite(v) for v in bbox)
                    )

                if detected:

                    left, top, right, bottom, cx, cy, bw, bh, conf = bbox

                    detected = (
                        0 <= left < right <= 1280 and
                        0 <= top < bottom <= 720 and
                        0 <= cx <= 1280 and
                        0 <= cy <= 720 and
                        bw > 0 and
                        bh > 0 and
                        0.60 <= conf <= 1.0
                    )

                visible_pub.publish(
                    Bool(data=bool(detected))
                )

                if detected:

                    bbox_msg = Float32MultiArray()
                    bbox_msg.data = bbox

                    bbox_pub.publish(bbox_msg)

                    point = PointStamped()

                    point.header.stamp = rospy.Time.now()
                    point.header.frame_id = "camera_pixels"

                    point.point.x = cx
                    point.point.y = cy
                    point.point.z = conf

                    center_pub.publish(point)

            except socket.timeout:

                visible_pub.publish(
                    Bool(data=False)
                )

            except (
                ValueError,
                KeyError,
                TypeError,
                OverflowError
            ) as e:

                rospy.logwarn(
                    "Invalid packet: %s",
                    str(e)
                )

                visible_pub.publish(
                    Bool(data=False)
                )

    finally:
        sock.close()


if __name__ == "__main__":
    main()

