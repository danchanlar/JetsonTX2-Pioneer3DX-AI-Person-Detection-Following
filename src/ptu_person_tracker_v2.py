#!/usr/bin/env python
from __future__ import print_function

import math
import time
import argparse

import rospy

from geometry_msgs.msg import PointStamped
from sensor_msgs.msg import JointState
from std_msgs.msg import Bool


# Camera
WIDTH = 1280.0
HEIGHT = 720.0

# Tracking parameters
CONTROL_HZ = 2.0
DEAD_ZONE = 0.08
ALPHA = 0.35
MIN_CONFIDENCE = 0.70

# PTU movement parameters
MAX_STEP = math.radians(0.3)
MAX_OFFSET = math.radians(2.0)
COMMAND_VELOCITY = 0.08

# Safety parameters
DETECTION_TIMEOUT = 0.5
FEEDBACK_TIMEOUT = 0.5
GOAL_TOLERANCE = math.radians(0.18)
GOAL_TIMEOUT = 2.5


def clamp(value, minimum, maximum):
    return max(minimum, min(maximum, value))


class PTUTrackerV2(object):

    def __init__(self, move=False):

        rospy.init_node("ptu_person_tracker_v2")

        self.move = move

        # Person detection
        self.person = None
        self.person_time = 0.0

        self.visible = False
        self.visible_time = 0.0

        # Smoothed coordinates
        self.filtered_x = None
        self.filtered_y = None

        # PTU feedback
        self.pan = None
        self.tilt = None
        self.feedback_time = 0.0

        # Initial PTU position
        self.initial_pan = None
        self.initial_tilt = None

        # Last commanded target
        self.pending_goal = None
        self.goal_time = 0.0
        self.fault = False

        self.publisher = rospy.Publisher(
            "/ptu/cmd",
            JointState,
            queue_size=1
        )

        rospy.Subscriber(
            "/person/center",
            PointStamped,
            self.person_callback,
            queue_size=1
        )

        rospy.Subscriber(
            "/person/visible",
            Bool,
            self.visible_callback,
            queue_size=1
        )

        rospy.Subscriber(
            "/joint_states",
            JointState,
            self.feedback_callback,
            queue_size=1
        )

    def person_callback(self, msg):

        if msg.header.frame_id != "camera_pixels":
            return

        x = msg.point.x
        y = msg.point.y
        confidence = msg.point.z

        if not all(math.isfinite(float(v))
                   for v in (x, y, confidence)):
            return

        if not (
            0 <= x <= WIDTH and
            0 <= y <= HEIGHT and
            MIN_CONFIDENCE <= confidence <= 1.0
        ):
            return

        self.person = (x, y, confidence)
        self.person_time = time.time()

    def visible_callback(self, msg):

        self.visible = bool(msg.data)
        self.visible_time = time.time()

        if not self.visible:
            self.filtered_x = None
            self.filtered_y = None

    def feedback_callback(self, msg):

        try:
            pan_index = list(msg.name).index("ptu_pan")
            tilt_index = list(msg.name).index("ptu_tilt")
            pan = float(msg.position[pan_index])
            tilt = float(msg.position[tilt_index])
        except (ValueError, IndexError, TypeError):
            return

        if not (
            math.isfinite(pan) and math.isfinite(tilt)
        ):
            return

        self.pan = pan
        self.tilt = tilt
        self.feedback_time = time.time()

        if self.initial_pan is None:

            self.initial_pan = pan
            self.initial_tilt = tilt

            rospy.loginfo(
                "Initial PTU: Pan=%.3f deg, Tilt=%.3f deg",
                math.degrees(pan),
                math.degrees(tilt)
            )

    def detection_is_valid(self, now):

        return (
            self.person is not None and
            self.visible and
            now - self.person_time < DETECTION_TIMEOUT and
            now - self.visible_time < DETECTION_TIMEOUT
        )

    def feedback_is_valid(self, now):

        return (
            self.pan is not None and
            self.tilt is not None and
            now - self.feedback_time < FEEDBACK_TIMEOUT
        )

    def goal_is_reached(self):

        if self.pending_goal is None:
            return True

        target_pan, target_tilt = self.pending_goal

        return (
            abs(self.pan - target_pan) <= GOAL_TOLERANCE and
            abs(self.tilt - target_tilt) <= GOAL_TOLERANCE
        )

    def publish_target(self, pan, tilt):

        cmd = JointState()
        cmd.header.stamp = rospy.Time.now()

        cmd.name = ["ptu_pan", "ptu_tilt"]

        cmd.position = [pan, tilt]

        cmd.velocity = [
            COMMAND_VELOCITY,
            COMMAND_VELOCITY
        ]

        self.publisher.publish(cmd)

        self.pending_goal = (pan, tilt)
        self.goal_time = time.time()

    def run(self):

        rate = rospy.Rate(CONTROL_HZ)

        rospy.loginfo(
            "PTU Tracker V2 | MOVE=%s",
            str(self.move)
        )

        while not rospy.is_shutdown():

            now = time.time()

            if self.fault:
                rospy.logerr_throttle(
                    3.0,
                    "TRACKER FAULT: no further commands; "
                    "stop and inspect PTU"
                )
                rate.sleep()
                continue

            if not self.feedback_is_valid(now):

                rospy.logwarn_throttle(
                    3.0,
                    "Waiting for fresh PTU feedback"
                )

                rate.sleep()
                continue

            # Do not command another movement until
            # the previous target has been reached.
            if self.move and self.pending_goal is not None:

                if self.goal_is_reached():
                    self.pending_goal = None

                elif now - self.goal_time > GOAL_TIMEOUT:

                    self.fault = True

                    rospy.logerr(
                        "PTU failed to reach target. "
                        "Automatic commands disabled."
                    )

                    rate.sleep()
                    continue

                else:
                    rate.sleep()
                    continue

            if not self.detection_is_valid(now):

                self.filtered_x = None
                self.filtered_y = None

                rospy.loginfo_throttle(
                    3.0,
                    "No fresh person detection"
                )

                rate.sleep()
                continue

            x, y, confidence = self.person

            # Exponential moving average
            if self.filtered_x is None:

                self.filtered_x = x
                self.filtered_y = y

            else:

                self.filtered_x = (
                    ALPHA * x +
                    (1.0 - ALPHA) * self.filtered_x
                )

                self.filtered_y = (
                    ALPHA * y +
                    (1.0 - ALPHA) * self.filtered_y
                )

            # Normalized image error
            ex = (
                self.filtered_x - WIDTH / 2.0
            ) / (WIDTH / 2.0)

            ey = (
                self.filtered_y - HEIGHT / 2.0
            ) / (HEIGHT / 2.0)

            # Dead zone
            if abs(ex) < DEAD_ZONE:
                ex = 0.0

            if abs(ey) < DEAD_ZONE:
                ey = 0.0

            # Calibrated directions:
            # Positive Pan = left
            # Positive Tilt = up
            pan_step = clamp(
                ex * MAX_STEP,
                -MAX_STEP,
                MAX_STEP
            )

            tilt_step = clamp(
                -ey * MAX_STEP,
                -MAX_STEP,
                MAX_STEP
            )

            # Position limits around initial position
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
                "PERSON (%.0f, %.0f) conf=%.2f | "
                "error=(%.2f, %.2f) | "
                "PTU=(%.2f, %.2f) | "
                "TARGET=(%.2f, %.2f)",
                self.filtered_x,
                self.filtered_y,
                confidence,
                ex,
                ey,
                math.degrees(self.pan),
                math.degrees(self.tilt),
                math.degrees(target_pan),
                math.degrees(target_tilt)
            )

            if self.move:

                if (
                    abs(target_pan - self.pan) > math.radians(0.02)
                    or
                    abs(target_tilt - self.tilt) > math.radians(0.02)
                ):

                    # Check the command destination is connected
                    if self.publisher.get_num_connections() != 1:

                        self.fault = True

                        rospy.logerr(
                            "Expected exactly one PTU command subscriber"
                        )

                    else:

                        # Avoid moving from an invalid initial state
                        if (
                            abs(self.pan - self.initial_pan)
                            > MAX_OFFSET + GOAL_TOLERANCE
                            or
                            abs(self.tilt - self.initial_tilt)
                            > MAX_OFFSET + GOAL_TOLERANCE
                        ):
                            self.fault = True
                            rospy.logerr(
                                "PTU outside tracking envelope"
                            )
                        else:
                            self.publish_target(
                                target_pan,
                                target_tilt
                            )

            rate.sleep()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--move",
        action="store_true",
        help="Enable real PTU movement"
    )

    parser.add_argument(
        "--confirm-calibrated",
        action="store_true",
        help="Confirm Pan/Tilt directions and clearance"
    )

    args = parser.parse_args()

    if args.move and not args.confirm_calibrated:
        parser.error(
            "--move requires --confirm-calibrated"
        )

    tracker = PTUTrackerV2(move=args.move)

    try:
        tracker.run()
    except rospy.ROSInterruptException:
        pass

