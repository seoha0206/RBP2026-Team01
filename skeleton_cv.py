#!/usr/bin/env python3

import cv2
import numpy as np


def detect_monitor(image):
    """
    Detect four corner points of the monitor (the largest rectangle) in the image.
    """

    top_left = np.array([631, 862], dtype="float32")
    top_right = np.array([1526, 346], dtype="float32")
    bottom_right = np.array([2034, 636], dtype="float32")
    bottom_left = np.array([1139, 1152], dtype="float32")

    return top_left, top_right, bottom_right, bottom_left


def rectify_monitor(image, top_left, top_right, bottom_right, bottom_left):
    """
    Warp the detected monitor to a front-facing rectangle.
    """

    width_top = np.linalg.norm(top_right - top_left)
    width_bottom = np.linalg.norm(bottom_right - bottom_left)
    max_width = int(max(width_top, width_bottom))

    height_right = np.linalg.norm(bottom_right - top_right)
    height_left = np.linalg.norm(bottom_left - top_left)
    max_height = int(max(height_right, height_left))

    src = np.array(
        [top_left, top_right, bottom_right, bottom_left],
        dtype="float32",
    )

    dst = np.array(
        [
            [0, 0],
            [max_width - 1, 0],
            [max_width - 1, max_height - 1],
            [0, max_height - 1],
        ],
        dtype="float32",
    )

    transform = cv2.getPerspectiveTransform(src, dst)

    rectified = cv2.warpPerspective(
        image,
        transform,
        (max_width, max_height),
    )

    return rectified


def detect_line(rectified):
    """
    Detect the dominant black line inside the rectified monitor using RANSAC.
    """

    gray = cv2.cvtColor(rectified, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    # Ignore borders
    margin_ratio = 0.05
    margin_x = int(w * margin_ratio)
    margin_y = int(h * margin_ratio)

    valid_region = gray[
        margin_y:h - margin_y,
        margin_x:w - margin_x,
    ]

    # Binary threshold for dark pixels
    _, binary = cv2.threshold(valid_region, 40, 255, cv2.THRESH_BINARY_INV)

    # Get coordinates of black pixels
    ys, xs = np.where(binary > 0)

    if len(xs) < 2:
        return None

    points = np.column_stack(
        (xs + margin_x, ys + margin_y)
    ).astype(np.float32)

    # -------------------------
    # RANSAC
    # -------------------------
    iterations = 500
    threshold = 2.0

    best_inliers = []
    best_model = None

    n = len(points)

    for _ in range(iterations):

        idx = np.random.choice(n, 2, replace=False)

        p1 = points[idx[0]]
        p2 = points[idx[1]]

        x1, y1 = p1
        x2, y2 = p2

        # Skip degenerate case
        if np.linalg.norm(p2 - p1) < 1e-6:
            continue

        # Line equation:
        # ax + by + c = 0
        a = y2 - y1
        b = x1 - x2
        c = x2 * y1 - x1 * y2

        norm = np.sqrt(a * a + b * b)
        if norm < 1e-6:
            continue

        # Distance from all points to line
        distances = np.abs(
            a * points[:, 0] + b * points[:, 1] + c
        ) / norm

        inlier_mask = distances < threshold
        inliers = points[inlier_mask]

        if len(inliers) > len(best_inliers):
            best_inliers = inliers
            best_model = (a, b, c)

    if len(best_inliers) < 2:
        return None

    # -------------------------
    # Refine using cv2.fitLine
    # -------------------------
    line = cv2.fitLine(
        best_inliers,
        cv2.DIST_L2,
        0,
        0.01,
        0.01,
    )

    vx, vy, x0, y0 = line.flatten()

    # Convert to drawable endpoints
    left_y = int(y0 + (-x0 * vy / vx))
    right_y = int(y0 + ((w - x0) * vy / vx))

    x1 = 0
    y1 = left_y

    x2 = w - 1
    y2 = right_y

    # Clamp
    y1 = np.clip(y1, 0, h - 1)
    y2 = np.clip(y2, 0, h - 1)

    best_line = (x1, y1, x2, y2)

    return best_line


def calculate_angle(line):
    """
    Calculate angle of the line.

    Definition:
    - 0 degree  : line points upward
    - Positive  : tilted to the left
    - Negative  : tilted to the right
    - Range     : (-90, 90]
    """

    if line is None:
        return None

    x1, y1, x2, y2 = line

    # Direction vector
    dx = x2 - x1
    dy = y2 - y1

    # Make the direction always point upward
    # (image y-axis increases downward)
    if dy > 0:
        dx = -dx
        dy = -dy

    # Angle relative to upward direction
    #
    # upward      -> 0°
    # left tilt   -> positive
    # right tilt  -> negative
    angle = np.degrees(np.arctan2(-dx, -dy))

    # Normalize to (-90, 90]
    if angle > 90:
        angle -= 180
    elif angle <= -90:
        angle += 180

    return angle


def line_point_distance(a, b, x, y):
    """
    Distance from point (x,y) to line y = ax + b
    """

    distance = abs(a * x - y + b) / np.sqrt(a * a + 1)

    return distance


def fit_line(points_list):
    """
    Fit line y = ax + b using least squares.
    """

    points = np.array(points_list, dtype=np.float32)

    x = points[:, 0]
    y = points[:, 1]

    if len(x) < 2:
        return None, None

    A = np.vstack([x, np.ones(len(x))]).T

    a, b = np.linalg.lstsq(A, y, rcond=None)[0]

    return a, b
