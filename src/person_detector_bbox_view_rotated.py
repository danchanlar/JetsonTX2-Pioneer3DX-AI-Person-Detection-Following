python3 - <<'PY'
path = '/home/perception/person_detection/distance_estimation/person_detector_bbox_view_rotated.py'

with open(path, 'r') as f:
    code = f.read()

# 1. Rotate BEFORE inference
old = '''            detections = net.Detect(
                img, width, height
            )'''

new = '''            # Convert CUDA image to NumPy
            rgba = jetson.utils.cudaToNumpy(
                img, width, height, 4
            )

            # Rotate image by 180 degrees
            rotated = np.ascontiguousarray(
                np.rot90(rgba, 2)
            )

            # Convert back to CUDA for TensorRT
            rotated_img = jetson.utils.cudaFromNumpy(
                rotated
            )

            # Run detection on correctly oriented image
            detections = net.Detect(
                rotated_img, width, height
            )'''

if old not in code:
    raise SystemExit("ERROR: net.Detect section not found")

code = code.replace(old, new, 1)

# 2. Preview must use the rotated image
old = '''                make_preview(
                    img,
                    width,
                    height,
                    target
                )'''

new = '''                make_preview(
                    rotated_img,
                    width,
                    height,
                    target
                )'''

if old not in code:
    raise SystemExit("ERROR: make_preview section not found")

code = code.replace(old, new, 1)

# 3. Remove the previous preview-only rotation
old = '''    # Rotate the camera preview by 180 degrees
    preview = cv2.rotate(
        preview,
        cv2.ROTATE_180
    )'''

if old not in code:
    raise SystemExit("ERROR: preview rotation section not found")

code = code.replace(old, '', 1)

# 4. Bounding box coordinates now belong to rotated image
old = '''        # Bounding box coordinates after 180-degree rotation
        p1 = (
            int((width - right) * sx),
            int((height - bottom) * sy)
        )

        p2 = (
            int((width - left) * sx),
            int((height - top) * sy)
        )'''

new = '''        p1 = (
            int(left * sx),
            int(top * sy)
        )

        p2 = (
            int(right * sx),
            int(bottom * sy)
        )'''

if old not in code:
    raise SystemExit("ERROR: bounding box section not found")

code = code.replace(old, new, 1)

with open(path, 'w') as f:
    f.write(code)

print("Rotated detector created successfully!")
PY
