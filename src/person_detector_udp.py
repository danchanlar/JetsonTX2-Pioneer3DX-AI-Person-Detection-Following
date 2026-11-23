#!/usr/bin/env python3

import os
import sys
import time
import json
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

    sock = socket.socket(
        socket.AF_INET,
        socket.SOCK_DGRAM
    )

    print("Loading SSD-MobileNet V2...")

    net = jetson.inference.detectNet(
        MODEL, [], THRESHOLD
    )

    camera = jetson.utils.gstCamera(
        WIDTH, HEIGHT, "0"
    )

    print("Camera initialized")
    print("Sending detections to UDP port 5005")

    count = 0
    start = time.time()

    try:
        while True:

            img, width, height = camera.CaptureRGBA()

            if img is None:
                raise RuntimeError("Camera capture failed")

            detections = net.Detect(img, width, height)

            persons = []

            for d in detections:

                label = net.GetClassDesc(d.ClassID)

                if label.lower() == "person":
                    persons.append(d)

            # Temporary target-selection strategy:
            # choose the person with highest confidence.
            target = max(
                persons,
                key=lambda d: d.Confidence
            ) if persons else None

            message = {
                "timestamp": time.time(),
                "width": width,
                "height": height,
                "detected": target is not None,
                "cx": None,
                "cy": None,
                "confidence": 0.0
            }

            if target is not None:

                message["cx"] = (
                    target.Left + target.Right
                ) / 2.0

                message["cy"] = (
                    target.Top + target.Bottom
                ) / 2.0

                message["confidence"] = float(
                    target.Confidence
                )

            sock.sendto(
                json.dumps(message).encode("utf-8"),
                (UDP_IP, UDP_PORT)
            )

            count += 1

            if count % 15 == 0:

                elapsed = time.time() - start

                fps = count / elapsed if elapsed > 0 else 0

                if target is not None:
                    print(
                        "PERSON | x={:.1f} y={:.1f} "
                        "confidence={:.2f} | FPS={:.2f}".format(
                            message["cx"],
                            message["cy"],
                            message["confidence"],
                            fps
                        )
                    )
                else:
                    print(
                        "NO PERSON | FPS={:.2f}".format(fps)
                    )

    except KeyboardInterrupt:
        print("Detector stopped")

    finally:
        sock.close()


if __name__ == "__main__":
    main()

