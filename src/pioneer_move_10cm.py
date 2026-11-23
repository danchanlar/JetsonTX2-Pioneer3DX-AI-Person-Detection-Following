# compile with python -m py_compile ~/pioneer_move_10cm.py
# run with python ~/pioneer_move_10cm.py

#!/usr/bin/env python
from __future__ import print_function

import math
import time
import rospy

from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from std_msgs.msg import Bool
from std_srvs.srv import Empty

TARGET_DISTANCE = 0.10
STOP_DISTANCE = 0.08
SPEED = 0.05
MAX_DURATION = 4.0
ODOM_TIMEOUT = 0.5


class PioneerMove(object):

    def __init__(self):
        rospy.init_node("pioneer_move_10cm")

        self.position = None
        self.last_odom = None
        self.motors = None
        self.initial_position = None
        self.motors_enabled = False

        self.pub = rospy.Publisher(
            "/RosAria/cmd_vel",
            Twist,
            queue_size=1
        )

        rospy.Subscriber(
            "/RosAria/pose",
            Odometry,
            self.odom_callback
        )

        rospy.Subscriber(
            "/RosAria/motors_state",
            Bool,
            self.motors_callback
        )

        rospy.wait_for_service(
            "/RosAria/enable_motors", timeout=5
        )

        rospy.wait_for_service(
            "/RosAria/disable_motors", timeout=5
        )

        self.enable = rospy.ServiceProxy(
            "/RosAria/enable_motors", Empty
        )

        self.disable = rospy.ServiceProxy(
            "/RosAria/disable_motors", Empty
        )

        rospy.on_shutdown(self.stop)

    def odom_callback(self, msg):
        self.position = (
            msg.pose.pose.position.x,
            msg.pose.pose.position.y
        )
        self.last_odom = time.time()

    def motors_callback(self, msg):
        self.motors = msg.data

    def stop(self):
        zero = Twist()

        for i in range(5):
            try:
                self.pub.publish(zero)
            except Exception:
                pass
            time.sleep(0.05)

    def distance(self):
        x0, y0 = self.initial_position
        x, y = self.position

        return math.hypot(x - x0, y - y0)

    def run(self):

        rospy.loginfo("Waiting for fresh odometry...")

        deadline = time.time() + 5.0

        while not rospy.is_shutdown():
            if (self.position is not None and
                self.last_odom is not None and
                time.time() - self.last_odom < 0.5 and
                self.motors is not None):
                break

            if time.time() > deadline:
                raise RuntimeError(
                    "Odometry or motor state unavailable"
                )

            time.sleep(0.1)

        if rospy.is_shutdown():
            raise RuntimeError("ROS shutdown")

        if self.motors is not False:
            raise RuntimeError(
                "Motors must initially be disabled"
            )

        self.stop()

        self.initial_position = self.position

        rospy.loginfo(
            "Initial position: x=%.3f y=%.3f",
            self.position[0],
            self.position[1]
        )

        if time.time() - self.last_odom > ODOM_TIMEOUT:
            raise RuntimeError("Odometry became stale")

        rospy.loginfo("Enabling motors...")
        self.enable()
        self.motors_enabled = True

        deadline = time.time() + 3.0

        while self.motors is not True:
            if rospy.is_shutdown():
                raise RuntimeError("ROS shutdown")

            if time.time() > deadline:
                raise RuntimeError(
                    "Motor activation not confirmed"
                )

            time.sleep(0.05)

        rospy.loginfo("Motors enabled")

        cmd = Twist()
        cmd.linear.x = SPEED
        cmd.angular.z = 0.0

        rate = rospy.Rate(10)
        start = time.time()

        rospy.loginfo("Starting forward movement")

        while not rospy.is_shutdown():

            now = time.time()

            if now - start >= MAX_DURATION:
                rospy.logwarn("Maximum duration reached")
                break

            if (self.last_odom is None or
                now - self.last_odom > ODOM_TIMEOUT):
                rospy.logwarn("Odometry timeout")
                break

            if self.motors is not True:
                rospy.logwarn("Motors disabled unexpectedly")
                break

            distance = self.distance()

            rospy.loginfo_throttle(
                0.5,
                "Distance: %.3f m" % distance
            )

            if distance >= STOP_DISTANCE:
                rospy.loginfo(
                    "Stopping threshold reached: %.3f m",
                    distance
                )
                break

            self.pub.publish(cmd)
            rate.sleep()

    def finish(self):

        rospy.loginfo("Stopping robot...")
        self.stop()

        if self.motors_enabled:
            try:
                self.disable()
                rospy.loginfo(
                    "Motor disable service called"
                )
            except Exception as e:
                rospy.logerr(
                    "Motor disable failed: %s", str(e)
                )

        if (self.position is not None and
            self.initial_position is not None):
            rospy.loginfo(
                "Final odometry distance: %.3f m",
                self.distance()
            )


if __name__ == "__main__":

    robot = None

    try:
        robot = PioneerMove()
        robot.run()

    except Exception as e:
        rospy.logerr("Experiment aborted: %s", str(e))

    finally:
        if robot is not None:
            robot.finish()

