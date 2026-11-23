#!/usr/bin/env python
from __future__ import print_function

import math
import time
import argparse
import rospy

from sensor_msgs.msg import JointState
from geometry_msgs.msg import PointStamped
from std_msgs.msg import Bool


# Camera
WIDTH = 1280.0
HEIGHT = 720.0

# Controller
CONTROL_HZ = 5.0
ALPHA = 0.40
DEAD_ZONE = 0.06
KP = 0.9

# Motion limits
MAX_STEP = math.radians(0.25)
MAX_OFFSET = math.radians(2.0)
COMMAND_VELOCITY = 0.12

# Data validation
MIN_CONFIDENCE = 0.70
DETECTION_TIMEOUT = 0.5
FEEDBACK_TIMEOUT = 0.5

# Calibrated directions from working V2
PAN_SIGN = 1.0
TILT_SIGN = -1.0


def clamp(value, low, high):
    return max(low, min(high, value))


def finite(value):
    return not (math.isnan(value) or math.isinf(value))


class PTUTrackerV3(object):

    def __init__(self, move=False):

        rospy.init_node("ptu_person_tracker_v3")

        self.move = move

        self.person = None
        self.person_time = 0.0

        self.visible = False
        self.visible_time = 0.0

        self.fx = None
        self.fy = None

        self.pan = None
        self.tilt = None
        self.feedback_time = 0.0

        self.initial_pan = None
        self.initial_tilt = None

        self.tracking = False
        self.last_command = None
        self.hold_sent = False

        self.cmd_pub = rospy.Publisher(
            "/ptu/cmd",
            JointState,
            queue_size=1
        )

        rospy.Subscriber(
            "/person/center",
            PointStamped,
            self.person_cb,
            queue_size=1
        )

        rospy.Subscriber(
            "/person/visible",
            Bool,
            self.visible_cb,
            queue_size=1
        )

        rospy.Subscriber(
            "/joint_states",
            JointState,
            self.feedback_cb,
            queue_size=1
        )

    def person_cb(self, msg):

        if msg.header.frame_id != "camera_pixels":
            return

        # Reject old ROS detection messages.
        stamp_age = (
            rospy.Time.now() - msg.header.stamp
        ).to_sec()

        if stamp_age < -0.2 or stamp_age > DETECTION_TIMEOUT:
            return

        x = float(msg.point.x)
        y = float(msg.point.y)
        confidence = float(msg.point.z)

        if not all(finite(v) for v in (x, y, confidence)):
            return

        if not (
            0.0 <= x <= WIDTH and
            0.0 <= y <= HEIGHT and
            MIN_CONFIDENCE <= confidence <= 1.0
        ):
            return

        self.person = (x, y, confidence)
        self.person_time = time.time()

    def visible_cb(self, msg):

        self.visible = bool(msg.data)
        self.visible_time = time.time()

        if not self.visible:
            self.fx = None
            self.fy = None

    def feedback_cb(self, msg):

        try:
            names = list(msg.name)

            pi = names.index("ptu_pan")
            ti = names.index("ptu_tilt")

            pan = float(msg.position[pi])
            tilt = float(msg.position[ti])

        except (ValueError, IndexError, TypeError):
            return

        if not (finite(pan) and finite(tilt)):
            return

        self.pan = pan
        self.tilt = tilt
        self.feedback_time = time.time()

        if self.initial_pan is None:

            self.initial_pan = pan
            self.initial_tilt = tilt

            rospy.loginfo(
                "Initial PTU: Pan=%.2f Tilt=%.2f deg",
                math.degrees(pan),
                math.degrees(tilt)
            )

    def feedback_ok(self, now):

        return (
            self.pan is not None and
            self.tilt is not None and
            now - self.feedback_time < FEEDBACK_TIMEOUT
        )

    def detection_ok(self, now):

        return (
            self.person is not None and
            self.visible and
            now - self.person_time < DETECTION_TIMEOUT and
            now - self.visible_time < DETECTION_TIMEOUT
        )

    def publish(self, pan, tilt):

        msg = JointState()

        msg.header.stamp = rospy.Time.now()
        msg.name = ["ptu_pan", "ptu_tilt"]

        msg.position = [pan, tilt]
        msg.velocity = [
            COMMAND_VELOCITY,
            COMMAND_VELOCITY
        ]

        self.cmd_pub.publish(msg)

        self.last_command = (pan, tilt)

    def stop_tracking(self):

        self.fx = None
        self.fy = None

        if self.tracking:
            rospy.logwarn("Tracking paused")

        self.tracking = False

        # One position-hold command when detection
        # is lost, only if feedback is fresh.
        if (
            self.move and
            not self.hold_sent and
            self.last_command is not None and
            self.feedback_ok(time.time()) and
            self.cmd_pub.get_num_connections() == 1
        ):
            self.publish(self.pan, self.tilt)
            self.hold_sent = True

            rospy.logwarn(
                "Hold position command sent"
            )

    def run(self):

        rate = rospy.Rate(CONTROL_HZ)

        rospy.loginfo(
            "PTU Tracker V3 started | MOVE=%s",
            self.move
        )

        while not rospy.is_shutdown():

            now = time.time()

            if not self.feedback_ok(now):

                self.stop_tracking()

                rospy.logwarn_throttle(
                    3.0,
                    "Waiting for fresh PTU feedback"
                )

                rate.sleep()
                continue

            # Check that the PTU has not moved
            # beyond the permitted tracking envelope.
            if (
                abs(self.pan - self.initial_pan)
                    > MAX_OFFSET + math.radians(0.2)
                or
                abs(self.tilt - self.initial_tilt)
                    > MAX_OFFSET + math.radians(0.2)
            ):
                self.stop_tracking()

                rospy.logerr(
                    "PTU outside tracking envelope"
                )
                break

            if not self.detection_ok(now):

                self.stop_tracking()

                rospy.loginfo_throttle(
                    3.0,
                    "No fresh person detection"
                )

                rate.sleep()
                continue

            x, y, confidence = self.person

            if self.fx is None:
                self.fx = x
                self.fy = y
            else:
                self.fx = (
                    ALPHA * x +
                    (1.0 - ALPHA) * self.fx
                )

                self.fy = (
                    ALPHA * y +
                    (1.0 - ALPHA) * self.fy
                )

            ex = (
                self.fx - WIDTH / 2.0
            ) / (WIDTH / 2.0)

            ey = (
                self.fy - HEIGHT / 2.0
            ) / (HEIGHT / 2.0)

            if abs(ex) < DEAD_ZONE:
                ex = 0.0

            if abs(ey) < DEAD_ZONE:
                ey = 0.0

            pan_step = clamp(
                PAN_SIGN * KP * ex * MAX_STEP,
                -MAX_STEP,
                MAX_STEP
            )

            tilt_step = clamp(
                TILT_SIGN * KP * ey * MAX_STEP,
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
                "PERSON=(%.0f,%.0f) conf=%.2f | "
                "ERROR=(%.2f,%.2f) | "
                "PTU=(%.2f,%.2f) | "
                "TARGET=(%.2f,%.2f)",
                self.fx,
                self.fy,
                confidence,
                ex,
                ey,
                math.degrees(self.pan),
                math.degrees(self.tilt),
                math.degrees(target_pan),
                math.degrees(target_tilt)
            )

            self.tracking = True

            if self.move:

                if self.cmd_pub.get_num_connections() != 1:
                    rospy.logerr(
                        "PTU command connection invalid"
                    )
                    break

                if (
                    abs(target_pan - self.pan)
                        > math.radians(0.02)
                    or
                    abs(target_tilt - self.tilt)
                        > math.radians(0.02)
                ):
                    self.publish(
                        target_pan,
                        target_tilt
                    )
                    self.hold_sent = False

            rate.sleep()

        self.stop_tracking()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--move",
        action="store_true"
    )

    parser.add_argument(
        "--confirm-calibrated",
        action="store_true"
    )

    args = parser.parse_args()

    if args.move and not args.confirm_calibrated:
        parser.error(
            "--move requires --confirm-calibrated"
        )

    tracker = PTUTrackerV3(move=args.move)

    try:
        tracker.run()

    except rospy.ROSInterruptException:
        pass

