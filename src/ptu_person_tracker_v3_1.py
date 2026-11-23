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

# Tracking
CONTROL_HZ = 5.0
ALPHA = 0.40
DEAD_ZONE = 0.06
MIN_CONFIDENCE = 0.60

# Calibrated directions
PAN_SIGN = 1.0
TILT_SIGN = -1.0

# Proportional controller
KP = 0.9

# Motion limits
MAX_STEP = math.radians(0.25)
MAX_OFFSET = math.radians(2.0)

MIN_SPEED = 0.03
MAX_SPEED = 0.10
MAX_ACCEL = 0.12
SPEED_GAIN = 0.18

# Timeouts
DETECTION_TIMEOUT = 0.50
LOSS_GRACE = 0.40
FEEDBACK_TIMEOUT = 0.50

# Safety
POSITION_MARGIN = math.radians(0.2)


def clamp(value, minimum, maximum):
    return max(minimum, min(maximum, value))


def finite(value):
    return not (math.isnan(value) or math.isinf(value))


class PTUTrackerV31(object):

    def __init__(self, move=False):

        rospy.init_node("ptu_person_tracker_v3_1")

        self.move = move

        # Detection
        self.person = None
        self.person_time = 0.0

        self.visible = False
        self.visible_time = 0.0

        # Filtered position
        self.fx = None
        self.fy = None

        # PTU feedback
        self.pan = None
        self.tilt = None
        self.feedback_time = 0.0

        self.initial_pan = None
        self.initial_tilt = None

        # Velocity command state
        self.pan_speed = 0.0
        self.tilt_speed = 0.0

        # Target loss
        self.loss_start = None
        self.hold_sent = False
        self.tracking_active = False

        # Fault state
        self.fault = False
        self.command_sent = False

        self.last_loop = time.time()

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

        age = (
            rospy.Time.now() - msg.header.stamp
        ).to_sec()

        if age < -0.2 or age > DETECTION_TIMEOUT:
            return

        x = float(msg.point.x)
        y = float(msg.point.y)
        conf = float(msg.point.z)

        if not all(finite(v) for v in (x, y, conf)):
            return

        if not (
            0 <= x <= WIDTH and
            0 <= y <= HEIGHT and
            MIN_CONFIDENCE <= conf <= 1.0
        ):
            return

        self.person = (x, y, conf)
        self.person_time = time.time()

    def visible_cb(self, msg):

        self.visible = bool(msg.data)
        self.visible_time = time.time()

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
                "Initial PTU: Pan=%.3f Tilt=%.3f deg",
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

    def ramp_speed(self, current, desired, dt):

        maximum_change = MAX_ACCEL * dt

        return clamp(
            desired,
            current - maximum_change,
            current + maximum_change
        )

    def publish_command(self, pan, tilt, vpan, vtilt):

        if not self.move:
            return True

        if self.cmd_pub.get_num_connections() != 1:
            rospy.logerr(
                "Invalid PTU command connection"
            )
            self.fault = True
            return False

        cmd = JointState()

        cmd.header.stamp = rospy.Time.now()

        cmd.name = ["ptu_pan", "ptu_tilt"]

        cmd.position = [pan, tilt]

        cmd.velocity = [
            max(MIN_SPEED, vpan),
            max(MIN_SPEED, vtilt)
        ]

        self.cmd_pub.publish(cmd)
        self.command_sent = True

        return True

    def hold_position(self):

        if self.hold_sent:
            return

        if not self.feedback_ok(time.time()):
            rospy.logerr(
                "Cannot issue hold: PTU feedback stale"
            )
            self.fault = True
            return

        if self.move and self.command_sent:

            if not self.publish_command(
                self.pan,
                self.tilt,
                MIN_SPEED,
                MIN_SPEED
            ):
                return

            rospy.logwarn(
                "HOLD command sent at actual PTU position"
            )

        self.pan_speed = 0.0
        self.tilt_speed = 0.0

        self.hold_sent = True

    def handle_loss(self, now):

        if self.loss_start is None:

            self.loss_start = now

            rospy.logwarn(
                "Detection lost: grace period started"
            )

        elapsed = now - self.loss_start

        if elapsed < LOSS_GRACE:

            rospy.loginfo_throttle(
                1.0,
                "Temporary detection loss"
            )

            return

        if self.tracking_active:

            rospy.logwarn(
                "Detection loss confirmed"
            )

        self.tracking_active = False

        self.fx = None
        self.fy = None

        self.hold_position()

    def run(self):

        rate = rospy.Rate(CONTROL_HZ)

        rospy.loginfo(
            "PTU Tracker V3.1 | MOVE=%s",
            self.move
        )

        while not rospy.is_shutdown():

            now = time.time()

            dt = clamp(
                now - self.last_loop,
                0.001,
                1.0 / CONTROL_HZ
            )

            self.last_loop = now

            if self.fault:

                rospy.logerr_throttle(
                    3.0,
                    "TRACKER FAULT - commands disabled"
                )

                rate.sleep()
                continue

            if not self.feedback_ok(now):

                if self.command_sent:
                    rospy.logerr(
                        "PTU feedback lost after motion command"
                    )
                    self.fault = True

                else:
                    rospy.logwarn_throttle(
                        3.0,
                        "Waiting for PTU feedback"
                    )

                rate.sleep()
                continue

            # Check tracking envelope
            if (
                abs(self.pan - self.initial_pan)
                > MAX_OFFSET + POSITION_MARGIN
                or
                abs(self.tilt - self.initial_tilt)
                > MAX_OFFSET + POSITION_MARGIN
            ):

                rospy.logerr(
                    "PTU outside permitted tracking envelope"
                )

                self.fault = True
                rate.sleep()
                continue

            if not self.detection_ok(now):

                self.handle_loss(now)

                rate.sleep()
                continue

            # Detection recovered
            if self.loss_start is not None:

                rospy.loginfo(
                    "Detection recovered"
                )

                self.loss_start = None

            self.hold_sent = False
            self.tracking_active = True

            x, y, conf = self.person

            # EMA smoothing
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

            # Normalized image errors
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

            # Desired speed magnitude
            desired_pan_speed = min(
                MAX_SPEED,
                SPEED_GAIN * abs(ex)
            )

            desired_tilt_speed = min(
                MAX_SPEED,
                SPEED_GAIN * abs(ey)
            )

            # Acceleration-limited speed
            self.pan_speed = self.ramp_speed(
                self.pan_speed,
                desired_pan_speed,
                dt
            )

            self.tilt_speed = self.ramp_speed(
                self.tilt_speed,
                desired_tilt_speed,
                dt
            )

            # Position step with velocity constraint
            pan_step = min(
                MAX_STEP * KP * abs(ex),
                self.pan_speed * dt
            )

            tilt_step = min(
                MAX_STEP * KP * abs(ey),
                self.tilt_speed * dt
            )

            pan_step *= PAN_SIGN * (
                1.0 if ex >= 0 else -1.0
            )

            tilt_step *= TILT_SIGN * (
                1.0 if ey >= 0 else -1.0
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
                "ERR=(%.2f,%.2f) | "
                "PTU=(%.2f,%.2f) | "
                "TARGET=(%.2f,%.2f) | "
                "SPEED=(%.3f,%.3f)",
                self.fx,
                self.fy,
                conf,
                ex,
                ey,
                math.degrees(self.pan),
                math.degrees(self.tilt),
                math.degrees(target_pan),
                math.degrees(target_tilt),
                self.pan_speed,
                self.tilt_speed
            )

            if self.move:

                if (
                    abs(target_pan - self.pan)
                    > math.radians(0.02)
                    or
                    abs(target_tilt - self.tilt)
                    > math.radians(0.02)
                ):

                    self.publish_command(
                        target_pan,
                        target_tilt,
                        self.pan_speed,
                        self.tilt_speed
                    )

            rate.sleep()

        rospy.loginfo("PTU Tracker V3.1 stopped")


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

    tracker = PTUTrackerV31(move=args.move)

    try:
        tracker.run()

    except rospy.ROSInterruptException:
        pass

