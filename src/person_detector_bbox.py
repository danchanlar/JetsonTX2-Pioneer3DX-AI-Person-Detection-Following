#!/usr/bin/env python3

import os
import json
import time
import socket
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

WIDTH = 1280
HEIGHT = 720

UDP_IP = "127.0.0.1"
UDP_PORT = 5005


def main():

    net = jetson.inference.detectNet(
        MODEL, [], THRESHOLD
    )

    camera = jetson.utils.gstCamera(
        WIDTH, HEIGHT, "0"
    )

    sock = socket.socket(
        socket.AF_INET,
        socket.SOCK_DGRAM
    )

    print("Bounding Box Detector started")
    print("Sending detections to UDP port 5005")

    frame_count = 0

    try:
        while True:

            img, width, height = camera.CaptureRGBA()

            if img is None:
                raise RuntimeError("Camera capture failed")

            detections = net.Detect(
                img, width, height
            )

            persons = []

            for d in detections:

                if net.GetClassDesc(d.ClassID).lower() == "person":
                    persons.append(d)

            # Same target selection as original detector
            target = max(
                persons,
                key=lambda d: d.Confidence
            ) if persons else None

            message = {
                "timestamp": time.time(),
                "width": width,
                "height": height,
                "detected": False,
                "bbox": None
            }

            if target is not None:

                left = float(target.Left)
                top = float(target.Top)
                right = float(target.Right)
                bottom = float(target.Bottom)

                bbox_width = right - left
                bbox_height = bottom - top

                if bbox_width > 0 and bbox_height > 0:

                    cx = (left + right) / 2.0
                    cy = (top + bottom) / 2.0

                    confidence = float(
                        target.Confidence
                    )

                    # Flag potentially incomplete boxes
                    clipped = (
                        left <= 5 or
                        top <= 5 or
                        right >= width - 5 or
                        bottom >= height - 5
                    )

                    message["detected"] = True

                    message["bbox"] = [
                        left,
                        top,
                        right,
                        bottom,
                        cx,
                        cy,
                        bbox_width,
                        bbox_height,
                        confidence
                    ]

                    frame_count += 1

                    if frame_count % 15 == 0:

                        print(
                            "PERSON | "
                            "Center=({:.1f},{:.1f}) | "
                            "BBox={:.1f}x{:.1f} | "
                            "Conf={:.2f} | "
                            "Clipped={}".format(
                                cx,
                                cy,
                                bbox_width,
                                bbox_height,
                                confidence,
                                clipped
                            )
                        )

            sock.sendto(
                json.dumps(message).encode("utf-8"),
                (UDP_IP, UDP_PORT)
            )

    except KeyboardInterrupt:
        print("Detector stopped")

    finally:
        sock.close()


if __name__ == "__main__":
    main()

