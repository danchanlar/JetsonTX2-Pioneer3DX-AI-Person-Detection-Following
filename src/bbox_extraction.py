#!/usr/bin/env python3

import os
import csv
import time
import jetson

jetson.__path__.append(
    os.path.expanduser(
        "~/person_detection/jetson-inference/utils/python/python/jetson"
    )
)

import jetson.inference
import jetson.utils


MODEL = "ssd-mobilenet-v2"
THRESHOLD = 0.60

CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 720

MAX_FRAMES = 120

OUTPUT_FILE = os.path.expanduser(
    "~/person_detection/distance_estimation/bbox_data.csv"
)


def main():

    print("Loading SSD-MobileNet V2...")

    net = jetson.inference.detectNet(
        MODEL,
        [],
        THRESHOLD
    )

    camera = jetson.utils.gstCamera(
        CAMERA_WIDTH,
        CAMERA_HEIGHT,
        "0"
    )

    print("Camera initialized")
    print("Starting Bounding Box Extraction")

    frame_count = 0
    detection_count = 0

    start_time = time.time()

    with open(OUTPUT_FILE, "w") as csvfile:

        writer = csv.writer(csvfile)

        writer.writerow([
            "frame",
            "timestamp",
            "left",
            "top",
            "right",
            "bottom",
            "center_x",
            "center_y",
            "bbox_width",
            "bbox_height",
            "confidence"
        ])

        try:

            while frame_count < MAX_FRAMES:

                img, width, height = camera.CaptureRGBA()

                if img is None:
                    print("Camera capture failed")
                    break

                detections = net.Detect(
                    img,
                    width,
                    height
                )

                persons = []

                for d in detections:

                    label = net.GetClassDesc(d.ClassID)

                    if label.lower() == "person":
                        persons.append(d)

                # Same initial selection method as V2
                target = max(
                    persons,
                    key=lambda d: d.Confidence
                ) if persons else None

                frame_count += 1

                if target is not None:

                    left = float(target.Left)
                    top = float(target.Top)
                    right = float(target.Right)
                    bottom = float(target.Bottom)

                    bbox_width = right - left
                    bbox_height = bottom - top

                    center_x = (left + right) / 2.0
                    center_y = (top + bottom) / 2.0

                    confidence = float(
                        target.Confidence
                    )

                    if bbox_width <= 0 or bbox_height <= 0:
                        continue

                    detection_count += 1

                    writer.writerow([
                        frame_count,
                        time.time(),
                        left,
                        top,
                        right,
                        bottom,
                        center_x,
                        center_y,
                        bbox_width,
                        bbox_height,
                        confidence
                    ])

                    if frame_count % 10 == 0:

                        print(
                            "Frame: {} | "
                            "Center: ({:.1f}, {:.1f}) | "
                            "BBox: {:.1f} x {:.1f} px | "
                            "Confidence: {:.2f}".format(
                                frame_count,
                                center_x,
                                center_y,
                                bbox_width,
                                bbox_height,
                                confidence
                            )
                        )

                elif frame_count % 10 == 0:

                    print(
                        "Frame: {} | No person detected".format(
                            frame_count
                        )
                    )

        except KeyboardInterrupt:
            print("Experiment interrupted")

    elapsed = time.time() - start_time

    fps = (
        frame_count / elapsed
        if elapsed > 0 else 0
    )

    print("")
    print("Bounding Box Extraction Completed")
    print("Frames:", frame_count)
    print("Frames with detection:", detection_count)
    print("Average FPS: {:.2f}".format(fps))
    print("CSV saved:", OUTPUT_FILE)


if __name__ == "__main__":
    main()

