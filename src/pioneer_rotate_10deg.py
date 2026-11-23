#python -m py_compile ~/pioneer_rotate_10deg.py
#python ~/pioneer_rotate_10deg.py

#!/usr/bin/env python
from __future__ import print_function

import math
import time
import rospy

from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from std_msgs.msg import Bool
from std_srvs.srv import Empty


TARGET_ANGLE_DEG = 10.0
STOP_ANGLE_DEG = 8.0

ANGULAR_SPEED = 0.12
MAX_DURATION = 4.0
ODOM_TIMEOUT = 0.5


def normalize_angle(angle):
    return math.atan2(math.sin(angle), math.cos(angle))


def quaternion_to_yaw(q):
    siny = 2.0 * (q.w * q.z + q.x * q.y)
    cosy = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny, cosy)


class PioneerRotate(object):

    def __init__(self):
        rospy.init_node("pioneer_rotate_10deg")

        self.yaw = None
        self.initial_yaw = None
        self.last_odom = None
        self.motors = None
        self.enable_attempted = False

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
        q = msg.pose.pose.orientation
        self.yaw = quaternion_to_yaw(q)
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

    def angle_rotated(self):
        return normalize_angle(
            self.yaw - self.initial_yaw
        )

    def run(self):

        rospy.loginfo("Waiting for fresh odometry...")

        deadline = time.time() + 5.0

        while not rospy.is_shutdown():

            fresh = (
                self.yaw is not None and
                self.last_odom is not None and
                time.time() - self.last_odom < ODOM_TIMEOUT
            )

            if fresh and self.motors is not None:
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

        rospy.loginfo("Waiting for cmd_vel subscriber...")

        deadline = time.time() + 5.0

        while self.pub.get_num_connections() == 0:

            if rospy.is_shutdown():
                raise RuntimeError("ROS shutdown")

            if time.time() > deadline:
                raise RuntimeError(
                    "No subscriber connected to cmd_vel"
                )

            time.sleep(0.1)

        rospy.loginfo("cmd_vel subscriber connected")

        self.stop()

        if time.time() - self.last_odom > ODOM_TIMEOUT:
            raise RuntimeError("Stale odometry")

        self.initial_yaw = self.yaw

        rospy.loginfo(
            "Initial yaw: %.3f deg",
            math.degrees(self.initial_yaw)
        )

        rospy.loginfo("Enabling motors...")

        self.enable_attempted = True
        self.enable()

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
        cmd.linear.x = 0.0
        cmd.angular.z = ANGULAR_SPEED

        rate = rospy.Rate(10)
        start = time.time()

        rospy.loginfo(
            "Starting CCW rotation at %.3f rad/s",
            ANGULAR_SPEED
        )

        while not rospy.is_shutdown():

            now = time.time()

            if now - start >= MAX_DURATION:
                rospy.logwarn(
                    "Maximum rotation time reached"
                )
                break

            if (
                self.last_odom is None or
                now - self.last_odom > ODOM_TIMEOUT
            ):
                rospy.logwarn("Odometry timeout")
                break

            if self.motors is not True:
                rospy.logwarn(
                    "Motors unexpectedly disabled"
                )
                break

            angle_deg = math.degrees(
                self.angle_rotated()
            )

            rospy.loginfo_throttle(
                0.5,
                "Rotation: %.2f deg" % angle_deg
            )

            if angle_deg < -2.0 or abs(angle_deg) > 15.0:
                rospy.logerr(
                    "Unexpected rotation: %.2f deg",
                    angle_deg
                )
                break

            if angle_deg >= STOP_ANGLE_DEG:
                rospy.loginfo(
                    "Stopping threshold reached: %.2f deg",
                    angle_deg
                )
                break

            self.pub.publish(cmd)
            rate.sleep()

    def finish(self):

        rospy.loginfo("Stopping robot...")
        self.stop()

        if self.enable_attempted:
            try:
                self.disable()
                rospy.loginfo(
                    "Motor disable service called"
                )
            except Exception as e:
                rospy.logerr(
                    "Motor disable failed: %s",
                    str(e)
                )

        if (
            self.yaw is not None and
            self.initial_yaw is not None
        ):
            rospy.loginfo(
                "Final odometry rotation: %.2f deg",
                math.degrees(self.angle_rotated())
            )


if __name__ == "__main__":

    robot = None

    try:
        robot = PioneerRotate()
        robot.run()

    except Exception as e:
        rospy.logerr(
            "Experiment aborted: %s",
            str(e)
        )

    finally:
        if robot is not None:
            robot.finish()
