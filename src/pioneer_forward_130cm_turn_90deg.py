# python -m py_compile ~/pioneer_forward_130cm_turn_90deg.py
# python ~/pioneer_forward_130cm_turn_90deg.py

#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Pioneer 3-DX: forward 1.30 m, then CCW turn 90 degrees.
ROS Melodic / Python 2.7, ROSARIA. Requires supervised clear test area.
"""
from __future__ import print_function

import math
import time
import threading
import rospy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from std_msgs.msg import Bool
from std_srvs.srv import Empty

# Targets are odometry-based, not independently measured physical motion.
LINEAR_TARGET_M = 1.30
LINEAR_STOP_M = 1.28        # conservative initial braking margin
LINEAR_SPEED_MPS = 0.05
LINEAR_MAX_S = 40.0

TURN_TARGET_DEG = 90.0
TURN_STOP_DEG = 87.0        # conservative initial braking margin
TURN_SPEED_RADPS = 0.12   # positive => counterclockwise
TURN_MAX_S = 20.0

ODOM_TIMEOUT_S = 0.5
CMD_HZ = 10
SETTLE_MAX_S = 3.0
STILL_REQUIRED_S = 0.5
STILL_LINEAR_MPS = 0.012
STILL_ANGULAR_RADPS = 0.03
MAX_LATERAL_M = 0.15


def angle_diff(a, b):
    return math.atan2(math.sin(a - b), math.cos(a - b))


def yaw_from_quaternion(q):
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                      1.0 - 2.0 * (q.y*q.y + q.z*q.z))


class Controller(object):
    def __init__(self):
        rospy.init_node('pioneer_forward_130cm_turn_90deg')
        self.lock = threading.Lock()
        self.pose = None      # (x, y, yaw, linear velocity, angular velocity)
        self.received = None
        self.motors = None
        self.enable_attempted = False
        self.pub = rospy.Publisher('/RosAria/cmd_vel', Twist, queue_size=1)
        self.odom_sub = rospy.Subscriber('/RosAria/pose', Odometry, self.odom_cb, queue_size=1)
        self.motor_sub = rospy.Subscriber('/RosAria/motors_state', Bool, self.motor_cb, queue_size=1)
        rospy.wait_for_service('/RosAria/enable_motors', timeout=5)
        rospy.wait_for_service('/RosAria/disable_motors', timeout=5)
        self.enable = rospy.ServiceProxy('/RosAria/enable_motors', Empty)
        self.disable = rospy.ServiceProxy('/RosAria/disable_motors', Empty)
        rospy.on_shutdown(self.stop)

    def odom_cb(self, msg):
        p = msg.pose.pose.position
        q = msg.pose.pose.orientation
        v = msg.twist.twist
        with self.lock:
            self.pose = (p.x, p.y, yaw_from_quaternion(q),
                         v.linear.x, v.angular.z)
            self.received = time.time()

    def motor_cb(self, msg):
        with self.lock:
            self.motors = msg.data

    def snapshot(self):
        with self.lock:
            return self.pose, self.received, self.motors

    def fresh_pose(self):
        pose, received, motors = self.snapshot()
        if pose is None or received is None or time.time() - received > ODOM_TIMEOUT_S:
            raise RuntimeError('Odometry missing or stale; aborting')
        return pose, motors

    def publish(self, linear=0.0, angular=0.0):
        cmd = Twist()
        cmd.linear.x = linear
        cmd.angular.z = angular
        self.pub.publish(cmd)

    def stop(self):
        for _ in range(10):
            try:
                self.publish()
            except Exception:
                pass
            time.sleep(0.05)

    def wait_still(self):
        """Wait for fresh odometry indicating the robot has settled."""
        deadline = time.time() + SETTLE_MAX_S
        still_since = None
        while not rospy.is_shutdown() and time.time() < deadline:
            pose, motors = self.fresh_pose()
            if motors is not True:
                raise RuntimeError('Motor state changed unexpectedly')
            if (abs(pose[3]) <= STILL_LINEAR_MPS and
                    abs(pose[4]) <= STILL_ANGULAR_RADPS):
                if still_since is None:
                    still_since = time.time()
                if time.time() - still_since >= STILL_REQUIRED_S:
                    return
            else:
                still_since = None
            self.publish()  # keep sending explicit zero command
            time.sleep(0.05)
        raise RuntimeError('Robot did not confirm a stable stop')

    def wait_ready(self):
        deadline = time.time() + 5.0
        while not rospy.is_shutdown() and time.time() < deadline:
            pose, received, motors = self.snapshot()
            if (pose is not None and received is not None and
                    time.time() - received < ODOM_TIMEOUT_S and
                    motors is not None and self.pub.get_num_connections() > 0):
                if motors is not False:
                    raise RuntimeError('Motors must be disabled before start')
                return
            time.sleep(0.1)
        raise RuntimeError('ROSARIA topics or cmd_vel subscriber unavailable')

    def enable_motors(self):
        self.enable_attempted = True
        self.enable()
        deadline = time.time() + 3.0
        while not rospy.is_shutdown() and time.time() < deadline:
            _, _, motors = self.snapshot()
            if motors is True:
                return
            time.sleep(0.05)
        raise RuntimeError('Motor activation not confirmed')

    def forward(self):
        (x0, y0, heading, _, _), _ = self.fresh_pose()
        rospy.loginfo('Stage 1: forward %.2f m at %.2f m/s',
                      LINEAR_TARGET_M, LINEAR_SPEED_MPS)
        started = time.time()
        rate = rospy.Rate(CMD_HZ)
        reached = False
        while not rospy.is_shutdown():
            if time.time() - started >= LINEAR_MAX_S:
                raise RuntimeError('Forward stage time limit reached')
            (x, y, yaw, _, _), motors = self.fresh_pose()
            if motors is not True:
                raise RuntimeError('Motors unexpectedly disabled')
            dx, dy = x-x0, y-y0
            distance = dx*math.cos(heading) + dy*math.sin(heading)
            lateral = -dx*math.sin(heading) + dy*math.cos(heading)
            rospy.loginfo_throttle(1.0, 'Forward odom: %.3f m, lateral: %.3f m' %
                                   (distance, lateral))
            if abs(lateral) > MAX_LATERAL_M:
                raise RuntimeError('Excessive lateral drift')
            if distance < -0.03:
                raise RuntimeError('Unexpected reverse motion')
            if distance >= LINEAR_STOP_M:
                rospy.loginfo('Forward stop threshold reached: %.3f m', distance)
                reached = True
                break
            self.publish(linear=LINEAR_SPEED_MPS)
            rate.sleep()
        self.stop()
        self.wait_still()
        (x, y, _, _, _), _ = self.fresh_pose()
        distance = (x-x0)*math.cos(heading) + (y-y0)*math.sin(heading)
        rospy.loginfo('Final forward odometry: %.3f m', distance)
        if not reached or not (1.20 <= distance <= 1.38):
            raise RuntimeError('Forward result outside allowed test interval; no turn')

    def turn(self):
        start_yaw = self.fresh_pose()[0][2]
        total_rad = 0.0
        prev_yaw = start_yaw
        started = time.time()
        reached = False
        rospy.loginfo('Stage 2: +%.1f degree CCW turn', TURN_TARGET_DEG)
        rate = rospy.Rate(CMD_HZ)
        while not rospy.is_shutdown():
            if time.time() - started >= TURN_MAX_S:
                raise RuntimeError('Turn stage time limit reached')
            (x, y, yaw, _, _), motors = self.fresh_pose()
            if motors is not True:
                raise RuntimeError('Motors unexpectedly disabled')
            total_rad += angle_diff(yaw, prev_yaw)
            prev_yaw = yaw
            degrees = math.degrees(total_rad)
            rospy.loginfo_throttle(1.0, 'Rotation: %.2f deg' % degrees)
            if degrees < -3.0 or degrees > 105.0:
                raise RuntimeError('Unexpected turn direction/overshoot')
            if degrees >= TURN_STOP_DEG:
                rospy.loginfo('Turn stop threshold reached: %.2f deg', degrees)
                reached = True
                break
            self.publish(angular=TURN_SPEED_RADPS)
            rate.sleep()
        self.stop()
        self.wait_still()
        final_yaw = self.fresh_pose()[0][2]
        # Target is only 90 deg, so normalized difference is sufficient here.
        final_deg = math.degrees(angle_diff(final_yaw, start_yaw))
        rospy.loginfo('Final odometry turn: %.2f deg', final_deg)
        if not reached or not (80.0 <= final_deg <= 100.0):
            raise RuntimeError('Turn result outside allowed test interval')

    def run(self):
        rospy.loginfo('Waiting for fresh ROSARIA odometry and cmd_vel connection...')
        self.wait_ready()
        self.stop()
        self.fresh_pose()
        self.enable_motors()
        rospy.loginfo('Motors enabled')
        self.forward()
        rospy.loginfo('Forward completed and stopped; proceeding to turn')
        self.turn()
        rospy.loginfo('Both stages finished based on odometry')

    def cleanup(self):
        rospy.loginfo('Sending zero velocity...')
        self.stop()
        if self.enable_attempted:
            try:
                self.disable()
                rospy.loginfo('Motor disable service called')
            except Exception as exc:
                rospy.logerr('MOTOR DISABLE FAILED: %s', exc)
        _, _, motors = self.snapshot()
        rospy.loginfo('Last received motor state: %s; verify using rostopic echo', motors)


if __name__ == '__main__':
    controller = None
    try:
        controller = Controller()
        controller.run()
    except (Exception, KeyboardInterrupt) as exc:
        rospy.logerr('Experiment aborted: %s', exc)
    finally:
        if controller is not None:
            controller.cleanup()
