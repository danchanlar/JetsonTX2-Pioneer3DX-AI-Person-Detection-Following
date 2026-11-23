#!/usr/bin/env python
from __future__ import print_function

import os
import csv
import time
import math
import argparse
import threading
import statistics

import rospy
from std_msgs.msg import Float32MultiArray


# Camera resolution
WIDTH = 1280.0
HEIGHT = 720.0

# Validation parameters
MIN_CONFIDENCE = 0.70
EDGE_MARGIN = 20.0
MIN_SAMPLE_INTERVAL = 0.20

# Maximum recording duration
RECORDING_TIMEOUT = 60.0

OUTPUT_FILE = os.path.expanduser(
    "~/person_detection/distance_estimation/calibration_data.csv"
)


def finite(value):
    return not (math.isnan(value) or math.isinf(value))


class BBoxCalibration(object):

    def __init__(self, distance, samples):

        self.distance = distance
        self.samples = samples

        self.data = []

        self.rejected = 0
        self.rejection_reasons = {}

        self.last_sample_time = 0.0

        self.lock = threading.Lock()
        self.enabled = False

        rospy.init_node(
            "bbox_calibration",
            anonymous=True
        )

        rospy.Subscriber(
            "/person/bbox",
            Float32MultiArray,
            self.bbox_callback,
            queue_size=1
        )

    def reject(self, reason):

        self.rejected += 1

        self.rejection_reasons[reason] = (
            self.rejection_reasons.get(reason, 0) + 1
        )

    def bbox_callback(self, msg):

        with self.lock:

            if not self.enabled:
                return

            if len(self.data) >= self.samples:
                return

            now = time.time()

            if now - self.last_sample_time < MIN_SAMPLE_INTERVAL:
                return

            if len(msg.data) != 9:
                self.reject("INVALID_LENGTH")
                return

            values = [float(v) for v in msg.data]

            if not all(finite(v) for v in values):
                self.reject("NON_FINITE_VALUE")
                return

            left, top, right, bottom, cx, cy, bw, bh, conf = values

            if conf < MIN_CONFIDENCE or conf > 1.0:
                self.reject("LOW_CONFIDENCE")
                return

            if (
                left <= EDGE_MARGIN or
                top <= EDGE_MARGIN or
                right >= WIDTH - EDGE_MARGIN or
                bottom >= HEIGHT - EDGE_MARGIN
            ):
                self.reject("CLIPPED_BBOX")
                return

            if (
                not (0 <= left < right <= WIDTH) or
                not (0 <= top < bottom <= HEIGHT) or
                bw <= 0 or
                bh <= 0 or
                abs(bw - (right - left)) > 2.0 or
                abs(bh - (bottom - top)) > 2.0 or
                abs(cx - (left + right) / 2.0) > 2.0 or
                abs(cy - (top + bottom) / 2.0) > 2.0
            ):
                self.reject("INVALID_GEOMETRY")
                return

            self.data.append([
                now,
                self.distance,
                left,
                top,
                right,
                bottom,
                cx,
                cy,
                bw,
                bh,
                conf
            ])

            self.last_sample_time = now

            count = len(self.data)

            if count % 10 == 0:

                rospy.loginfo(
                    "Accepted: %d/%d | "
                    "Height: %.1f px | "
                    "Confidence: %.2f",
                    count,
                    self.samples,
                    bh,
                    conf
                )

    def run(self):

        print("")
        print("===================================")
        print("   BOUNDING BOX CALIBRATION A3")
        print("===================================")
        print("")
        print("Distance: {:.2f} m".format(self.distance))
        print("Required samples: {}".format(self.samples))
        print("")
        print("Stand upright at the measured distance.")
        print("Keep the full body visible.")
        print("Keep the camera and PTU stationary.")
        print("")

        try:
            raw_input("Press ENTER to start recording...")
        except EOFError:
            print("An interactive terminal is required.")
            return

        start = time.time()

        with self.lock:
            self.enabled = True

        rate = rospy.Rate(10)

        while not rospy.is_shutdown():

            with self.lock:
                count = len(self.data)

            if count >= self.samples:
                break

            if time.time() - start > RECORDING_TIMEOUT:
                print("Recording timeout.")
                break

            rate.sleep()

        with self.lock:
            self.enabled = False
            rows = list(self.data)
            rejected = self.rejected
            reasons = dict(self.rejection_reasons)

        if len(rows) < self.samples:

            print("")
            print("Not enough valid samples.")
            print("Accepted: {}/{}".format(
                len(rows), self.samples
            ))

            print("Rejected:", rejected)
            print("Reasons:", reasons)

            print("No partial measurements saved.")
            return

        heights = [row[9] for row in rows]
        widths = [row[8] for row in rows]
        confidences = [row[10] for row in rows]

        mean_height = statistics.mean(heights)
        median_height = statistics.median(heights)
        std_height = statistics.pstdev(heights)

        mean_width = statistics.mean(widths)
        mean_conf = statistics.mean(confidences)

        exists = os.path.exists(OUTPUT_FILE)

        with open(OUTPUT_FILE, "ab") as csvfile:

            writer = csv.writer(csvfile)

            if not exists or os.path.getsize(OUTPUT_FILE) == 0:

                writer.writerow([
                    "timestamp",
                    "distance_m",
                    "left",
                    "top",
                    "right",
                    "bottom",
                    "center_x",
                    "center_y",
                    "bbox_width_px",
                    "bbox_height_px",
                    "confidence"
                ])

            writer.writerows(rows)

        print("")
        print("===================================")
        print("       CALIBRATION RESULTS")
        print("===================================")

        print("Distance: {:.2f} m".format(self.distance))
        print("Accepted samples:", len(rows))
        print("Rejected samples:", rejected)

        print(
            "Mean height: {:.2f} px".format(
                mean_height
            )
        )

        print(
            "Median height: {:.2f} px".format(
                median_height
            )
        )

        print(
            "Height std: {:.2f} px".format(
                std_height
            )
        )

        print(
            "Mean width: {:.2f} px".format(
                mean_width
            )
        )

        print(
            "Mean confidence: {:.3f}".format(
                mean_conf
            )
        )

        print("Rejection reasons:", reasons)
        print("CSV saved:", OUTPUT_FILE)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--distance",
        type=float,
        required=True
    )

    parser.add_argument(
        "--samples",
        type=int,
        default=40
    )

    args = parser.parse_args()

    if args.distance <= 0:
        parser.error("Distance must be positive")

    if args.samples < 1:
        parser.error("Samples must be positive")

    calibration = BBoxCalibration(
        args.distance,
        args.samples
    )

    calibration.run()

# We use Python 2.7 of ROS Melodic. However, Python 2.7 does not have module statistics of Python 3. So, before execute the program, make this small change in order to avoid extra library.

python3 - <<'PY'
path = '/home/perception/person_detection/distance_estimation/bbox_calibration.py'

with open(path, 'r') as f:
    code = f.read()

code = code.replace(
    'import statistics',
    'import math'
)

code = code.replace(
    'mean_height = statistics.mean(heights)',
    'mean_height = sum(heights) / float(len(heights))'
)

code = code.replace(
    'median_height = statistics.median(heights)',
    '''sorted_heights = sorted(heights)
        n = len(sorted_heights)
        if n % 2:
            median_height = sorted_heights[n // 2]
        else:
            median_height = (
                sorted_heights[n // 2 - 1] +
                sorted_heights[n // 2]
            ) / 2.0'''
)

code = code.replace(
    'std_height = statistics.pstdev(heights)',
    '''std_height = math.sqrt(
            sum((h - mean_height) ** 2 for h in heights)
            / float(len(heights))
        )'''
)

code = code.replace(
    'mean_width = statistics.mean(widths)',
    'mean_width = sum(widths) / float(len(widths))'
)

code = code.replace(
    'mean_conf = statistics.mean(confidences)',
    'mean_conf = sum(confidences) / float(len(confidences))'
)

with open(path, 'w') as f:
    f.write(code)

print("Python 2.7 compatibility applied")
PY
