#!/usr/bin/env python
from __future__ import print_function

import math
import time
import argparse
import rospy

from geometry_msgs.msg import PointStamped
from std_msgs.msg import Bool
from sensor_msgs.msg import JointState


WIDTH = 1280.0
HEIGHT = 720.0

DEAD_ZONE = 0.08
MAX_STEP = math.radians(0.5)
MAX_OFFSET = math.radians(5.0)
COMMAND_VELOCITY = 0.08

DATA_TIMEOUT = 0.6
STATE_TIMEOUT = 1.0


def clamp(value, low, high):
    return max(low, min(high, value))


class PTUTracker(object):

    def __init__(self, move=False):

        rospy.init_node("ptu_person_tracker")

        self.move = move

        self.person = None
        self.person_time = 0.0
        self.visible = False
        self.visible_time = 0.0

        self.pan = None
        self.tilt = None
        self.state_time = 0.0

        self.initial_pan = None
        self.initial_tilt = None
        self.names = None

        self.pub = rospy.Publisher(
            "/ptu/cmd",
            JointState,
            queue_size=1
        )

        rospy.Subscriber(
            "/person/center",
            PointStamped,
            self.person_callback
        )

        rospy.Subscriber(
            "/person/visible",
            Bool,
            self.visible_callback
        )

        rospy.Subscriber(
            "/ptu/state",
            JointState,
            self.state_callback
        )

    def person_callback(self, msg):

        if msg.header.frame_id != "camera_pixels":
            return

        if not (
            0.0 <= msg.point.x <= WIDTH and
            0.0 <= msg.point.y <= HEIGHT and
            0.0 <= msg.point.z <= 1.0
        ):
            return

        self.person = msg
        self.person_time = time.time()

    def visible_callback(self, msg):
        self.visible = bool(msg.data)
        self.visible_time = time.time()

    def state_callback(self, msg):

        if len(msg.position) < 2:
            return

        if len(msg.name) < 2:
            return

        pan_index = next(
            (i for i, n in enumerate(msg.name)
             if "pan" in n.lower()), None
        )

        tilt_index = next(
            (i for i, n in enumerate(msg.name)
             if "tilt" in n.lower()), None
        )

        if pan_index is None or tilt_index is None:
            return

        if max(pan_index, tilt_index) >= len(msg.position):
            return

        self.pan = msg.position[pan_index]
        self.tilt = msg.position[tilt_index]
        self.names = [
            msg.name[pan_index],
            msg.name[tilt_index]
        ]
        self.state_time = time.time()

        if self.initial_pan is None:
            self.initial_pan = self.pan
            self.initial_tilt = self.tilt

            rospy.loginfo(
                "Initial PTU: pan=%.2f tilt=%.2f deg",
                math.degrees(self.pan),
                math.degrees(self.tilt)
            )

    def run(self):

        rate = rospy.Rate(2)

        while not rospy.is_shutdown():

            now = time.time()

            if self.pan is None or self.tilt is None:
                rospy.logwarn_throttle(
                    3.0, "Waiting for PTU feedback"
                )
                rate.sleep()
                continue

            if now - self.state_time > STATE_TIMEOUT:
                rospy.logwarn_throttle(
                    3.0, "PTU feedback stale"
                )
                rate.sleep()
                continue

            if (
                self.person is None or
                not self.visible or
                now - self.person_time > DATA_TIMEOUT or
                now - self.visible_time > DATA_TIMEOUT
            ):
                rospy.loginfo_throttle(
                    3.0, "No fresh person detection"
                )
                rate.sleep()
                continue

            cx = self.person.point.x
            cy = self.person.point.y

            ex = (cx - WIDTH / 2.0) / (WIDTH / 2.0)
            ey = (cy - HEIGHT / 2.0) / (HEIGHT / 2.0)

            if abs(ex) < DEAD_ZONE:
                ex = 0.0

            if abs(ey) < DEAD_ZONE:
                ey = 0.0

            # Direction signs are not calibrated yet.
            pan_step = clamp(
                -ex * MAX_STEP,
                -MAX_STEP,
                MAX_STEP
            )

            tilt_step = clamp(
                -ey * MAX_STEP,
                -MAX_STEP,
                MAX_STEP
            )

            target_pan = clamp(
                self.pan + pan_step,
                self.initial_pan - MAX_OFFSET,
                self.initial_pan + MAX_OFFSET
            )

            target_tilt = clamp(
                self.tilt + tilt_step,
                self.initial_tilt - MAX_OFFSET,
                self.initial_tilt + MAX_OFFSET
            )

            rospy.loginfo(
                "Person (%.0f, %.0f) | "
                "error (%.2f, %.2f) | "
                "target pan=%.2f tilt=%.2f",
                cx, cy, ex, ey,
                math.degrees(target_pan),
                math.degrees(target_tilt)
            )

            if self.move and (ex != 0.0 or ey != 0.0):

                cmd = JointState()
                cmd.header.stamp = rospy.Time.now()
                cmd.name = list(self.names)

                cmd.position = [
                    target_pan,
                    target_tilt
                ]

                cmd.velocity = [
                    COMMAND_VELOCITY,
                    COMMAND_VELOCITY
                ]

                self.pub.publish(cmd)

            rate.sleep()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--move",
        action="store_true",
        help="Enable real PTU movement"
    )

    args = parser.parse_args()

    tracker = PTUTracker(move=args.move)

    if args.move:
        rospy.logwarn(
            "REAL PTU MOVEMENT ENABLED"
        )
    else:
        rospy.loginfo(
            "DRY RUN: PTU movement disabled"
        )

    tracker.run()

