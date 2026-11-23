
#!/usr/bin/env python

from __future__ import print_function

import sys
import math
import time
import rospy

from sensor_msgs.msg import JointState


STEP = math.radians(0.5)
MAX_ANGLE = math.radians(5.0)
SPEED = 0.10


def read_position():

    msg = rospy.wait_for_message(
        "/joint_states",
        JointState,
        timeout=3.0
    )

    names = list(msg.name)

    pan = msg.position[names.index("ptu_pan")]
    tilt = msg.position[names.index("ptu_tilt")]

    return pan, tilt


def main():

    rospy.init_node("ptu_calibration")

    if len(sys.argv) != 2:
        print("Usage: python ptu_calibration.py pan+|tilt+|zero")
        return

    mode = sys.argv[1]

    if mode not in ("pan+", "tilt+", "zero"):
        print("Invalid mode")
        return

    pub = rospy.Publisher(
        "/ptu/cmd",
        JointState,
        queue_size=1
    )

    pan, tilt = read_position()

    print("Current position:")
    print("Pan:  {:.3f} deg".format(math.degrees(pan)))
    print("Tilt: {:.3f} deg".format(math.degrees(tilt)))

    if abs(pan) > MAX_ANGLE or abs(tilt) > MAX_ANGLE:
        raise RuntimeError(
            "PTU is outside the calibration range (+/-5 deg)"
        )

    target_pan = pan
    target_tilt = tilt

    if mode == "pan+":
        target_pan = pan + STEP

    elif mode == "tilt+":
        target_tilt = tilt + STEP

    elif mode == "zero":
        target_pan = 0.0
        target_tilt = 0.0

    if (
        abs(target_pan) > MAX_ANGLE or
        abs(target_tilt) > MAX_ANGLE
    ):
        raise RuntimeError("Target outside calibration range")

    print("")
    print("Target position:")
    print("Pan:  {:.3f} deg".format(math.degrees(target_pan)))
    print("Tilt: {:.3f} deg".format(math.degrees(target_tilt)))

    start = time.time()

    while pub.get_num_connections() == 0:

        if time.time() - start > 3.0:
            raise RuntimeError("PTU driver not connected")

        rospy.sleep(0.1)

    cmd = JointState()

    cmd.header.stamp = rospy.Time.now()

    cmd.name = ["ptu_pan", "ptu_tilt"]

    cmd.position = [
        target_pan,
        target_tilt
    ]

    cmd.velocity = [
        SPEED,
        SPEED
    ]

    print("")
    print("Sending PTU command...")

    pub.publish(cmd)

    rospy.sleep(3.0)

    pan_after, tilt_after = read_position()

    print("")
    print("Measured position:")
    print("Pan:  {:.3f} deg".format(math.degrees(pan_after)))
    print("Tilt: {:.3f} deg".format(math.degrees(tilt_after)))

    print("")
    print("Measured displacement:")
    print("Delta Pan:  {:.3f} deg".format(
        math.degrees(pan_after - pan)
    ))
    print("Delta Tilt: {:.3f} deg".format(
        math.degrees(tilt_after - tilt)
    ))


if __name__ == "__main__":
    main()

