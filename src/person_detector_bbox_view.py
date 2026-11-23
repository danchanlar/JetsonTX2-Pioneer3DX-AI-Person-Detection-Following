#!/usr/bin/env python3

import os
import sys
import json
import time
import socket
import threading

from http.server import BaseHTTPRequestHandler, HTTPServer
from socketserver import ThreadingMixIn

import cv2
import numpy as np
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

WEB_IP = "127.0.0.1"
WEB_PORT = 8080

# Lower resolution to reduce CPU/network load
VIEW_WIDTH = 640
VIEW_HEIGHT = 360

# Create a new JPEG every 3 camera frames
VIEW_EVERY_N_FRAMES = 3

JPEG_QUALITY = 65
EDGE_MARGIN = 20

latest_jpeg = None
jpeg_lock = threading.Lock()


class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    daemon_threads = True
    allow_reuse_address = True


class StreamHandler(BaseHTTPRequestHandler):

    def do_GET(self):

        if self.path == "/":

            page = """
            <!DOCTYPE html>
            <html>
            <head>
              <title>Jetson TX2 Person Detection</title>
              <meta charset="UTF-8">
              <style>
                body {
                  background: #151515;
                  color: white;
                  font-family: Arial;
                  text-align: center;
                }
                img {
                  width: 95%;
                  max-width: 960px;
                  border: 2px solid #555;
                }
              </style>
            </head>
            <body>
              <h2>Jetson TX2 - Live Person Detection</h2>
              <p>SSD-MobileNet V2 | Bounding Box Viewer</p>
              <img src="/stream">
            </body>
            </html>
            """

            self.send_response(200)
            self.send_header(
                "Content-Type",
                "text/html; charset=utf-8"
            )
            self.end_headers()

            self.wfile.write(
                page.encode("utf-8")
            )

        elif self.path == "/stream":

            self.send_response(200)

            self.send_header(
                "Content-Type",
                "multipart/x-mixed-replace; boundary=frame"
            )

            self.send_header(
                "Cache-Control",
                "no-cache"
            )

            self.end_headers()

            try:

                while True:

                    with jpeg_lock:
                        frame = latest_jpeg

                    if frame is not None:

                        self.wfile.write(b"--frame\r\n")
                        self.wfile.write(
                            b"Content-Type: image/jpeg\r\n"
                        )

                        self.wfile.write(
                            ("Content-Length: %d\r\n\r\n" %
                             len(frame)).encode("ascii")
                        )

                        self.wfile.write(frame)
                        self.wfile.write(b"\r\n")
                        self.wfile.flush()

                    time.sleep(0.10)

            except (BrokenPipeError, ConnectionResetError):
                pass

        else:

            self.send_error(404)

    def log_message(self, format, *args):
        pass


def start_web_server():

    server = ThreadedHTTPServer(
        (WEB_IP, WEB_PORT),
        StreamHandler
    )

    thread = threading.Thread(
        target=server.serve_forever
    )

    thread.daemon = True
    thread.start()

    print(
        "Web viewer listening on {}:{}".format(
            WEB_IP, WEB_PORT
        )
    )

    return server


def make_preview(img, width, height, target):

    global latest_jpeg

    # Historical jetson.utils API: CUDA RGBA float4 -> NumPy
    rgba = jetson.utils.cudaToNumpy(
        img, width, height, 4
    )

    rgba8 = np.clip(
        rgba, 0, 255
    ).astype(np.uint8)

    bgr = cv2.cvtColor(
        rgba8, cv2.COLOR_RGBA2BGR
    )

    preview = cv2.resize(
        bgr,
        (VIEW_WIDTH, VIEW_HEIGHT)
    )

    sx = float(VIEW_WIDTH) / width
    sy = float(VIEW_HEIGHT) / height

    if target is not None:

        left = float(target.Left)
        top = float(target.Top)
        right = float(target.Right)
        bottom = float(target.Bottom)

        bw = right - left
        bh = bottom - top

        cx = (left + right) / 2.0
        cy = (top + bottom) / 2.0

        conf = float(target.Confidence)

        clipped = (
            left <= EDGE_MARGIN or
            top <= EDGE_MARGIN or
            right >= width - EDGE_MARGIN or
            bottom >= height - EDGE_MARGIN
        )

        p1 = (
            int(left * sx),
            int(top * sy)
        )

        p2 = (
            int(right * sx),
            int(bottom * sy)
        )

        color = (
            (0, 0, 255) if clipped
            else (0, 255, 0)
        )

        cv2.rectangle(
            preview,
            p1,
            p2,
            color,
            2
        )

        info = "PERSON conf={:.2f}".format(conf)

        cv2.putText(
            preview,
            info,
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 0),
            2
        )

        cv2.putText(
            preview,
            "Center=({:.0f},{:.0f})".format(cx, cy),
            (10, 52),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2
        )

        cv2.putText(
            preview,
            "BBox={:.0f} x {:.0f} px".format(bw, bh),
            (10, 79),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2
        )

        status = (
            "CLIPPED - INVALID FOR CALIBRATION"
            if clipped
            else "BBOX INSIDE IMAGE"
        )

        cv2.putText(
            preview,
            status,
            (10, VIEW_HEIGHT - 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            color,
            2
        )

    else:

        cv2.putText(
            preview,
            "NO PERSON DETECTED",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2
        )

    ok, encoded = cv2.imencode(
        ".jpg",
        preview,
        [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY]
    )

    if ok:

        with jpeg_lock:
            latest_jpeg = encoded.tobytes()


def main():

    print("Loading SSD-MobileNet V2...")

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

    server = start_web_server()

    print("Camera + Bounding Box Viewer started")

    frame_count = 0

    try:

        while True:

            img, width, height = camera.CaptureRGBA(
                zeroCopy=1
            )

            if img is None:
                continue

            detections = net.Detect(
                img, width, height
            )

            persons = []

            for d in detections:

                if net.GetClassDesc(d.ClassID).lower() == "person":
                    persons.append(d)

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

                bw = right - left
                bh = bottom - top

                if bw > 0 and bh > 0:

                    cx = (left + right) / 2.0
                    cy = (top + bottom) / 2.0
                    conf = float(target.Confidence)

                    message["detected"] = True

                    message["bbox"] = [
                        left, top, right, bottom,
                        cx, cy, bw, bh, conf
                    ]

            # Keep the same UDP format as A2
            sock.sendto(
                json.dumps(message).encode("utf-8"),
                (UDP_IP, UDP_PORT)
            )

            frame_count += 1

            # Visualization is less frequent than detection
            if frame_count % VIEW_EVERY_N_FRAMES == 0:

                make_preview(
                    img,
                    width,
                    height,
                    target
                )

                if frame_count % 30 == 0:

                    print(
                        "Processed {} frames".format(
                            frame_count
                        )
                    )

    except KeyboardInterrupt:

        print("Detector stopped")

    finally:

        server.shutdown()
        server.server_close()
        sock.close()


if __name__ == "__main__":
    main()

