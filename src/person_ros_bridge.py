#!/usr/bin/env python
from __future__ import print_function

import json
import socket
import time
import rospy

from geometry_msgs.msg import PointStamped
from std_msgs.msg import Bool


UDP_IP = "127.0.0.1"
UDP_PORT = 5005

MAX_AGE = 0.6


def main():

    rospy.init_node("person_ros_bridge")

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

    sock = socket.socket(
        socket.AF_INET,
        socket.SOCK_DGRAM
    )

    sock.bind((UDP_IP, UDP_PORT))
    sock.settimeout(0.2)

    rospy.loginfo(
        "ROS bridge listening on 127.0.0.1:5005"
    )

    try:
        while not rospy.is_shutdown():

            try:
                data, address = sock.recvfrom(4096)

                msg = json.loads(data.decode("utf-8"))

                age = time.time() - float(
                    msg["timestamp"]
                )

                valid = (
                    -0.2 <= age <= MAX_AGE
                )

                detected = (
                    valid and
                    msg["detected"] and
                    msg["cx"] is not None and
                    msg["cy"] is not None and
                    float(msg["confidence"]) >= 0.60 and
                    int(msg["width"]) == 1280 and
                    int(msg["height"]) == 720
                )

                visible_pub.publish(
                    Bool(data=bool(detected))
                )

                if detected:

                    point = PointStamped()

                    point.header.stamp = rospy.Time.now()
                    point.header.frame_id = "camera_pixels"

                    point.point.x = float(msg["cx"])
                    point.point.y = float(msg["cy"])
                    point.point.z = float(msg["confidence"])

                    center_pub.publish(point)

            except socket.timeout:

                visible_pub.publish(
                    Bool(data=False)
                )

            except (ValueError, KeyError, TypeError) as e:

                rospy.logwarn(
                    "Invalid detection packet: %s",
                    str(e)
                )

                visible_pub.publish(
                    Bool(data=False)
                )

    finally:
        sock.close()


if __name__ == "__main__":
    main()

