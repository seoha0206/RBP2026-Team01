#!/usr/bin/env python3

import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image
from std_msgs.msg import Float32
from cv_bridge import CvBridge

from skeleton_cv import *


class LineDetector(Node):
    def __init__(self):
        super().__init__("line_detector_node")

        self.bridge = CvBridge()

        self.image_sub = self.create_subscription(
            Image,
            "/image",
            self.image_callback,
            10,
        )

        self.angle_pub = self.create_publisher(
            Float32,
            "/angle",
            10,
        )

        # For Debugging
        self.monitor_pub = self.create_publisher(
            Image,
            "/debug/monitor",
            10,
        )

        self.rectified_pub = self.create_publisher(
            Image,
            "/debug/rectified",
            10,
        )

        self.line_pub = self.create_publisher(
            Image,
            "/debug/line",
            10,
        )

        self.get_logger().info("Line Detector Started.")

    def image_callback(self, msg):
        try:
            image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as e:
            self.get_logger().error(f"Failed to convert image: {e}")
            return
        
        ### 1. Detect monitor
        # Please check NOTE inside the detect_monitor() function.
        top_left, top_right, bottom_right, bottom_left = detect_monitor(image)

        if top_left is None or top_right is None or bottom_left is None or bottom_right is None:
            self.get_logger().warn("Monitor not detected.")
        else:
            self._debug_monitor(msg, image, top_left, top_right, bottom_right, bottom_left)

        ### 2. Rectify monitor
        # Please check NOTE inside the rectify_monitor() function.
        rectified = rectify_monitor(image, top_left, top_right, bottom_right, bottom_left)

        if rectified is None:
            self.get_logger().warn("Monitor not rectified.")
        else:
            self._debug_rectified(msg, rectified)

        ### 3. Detect line
        # TODO: Implement _detect_line() function.
        line = detect_line(rectified)

        if line is None:
            self.get_logger().warn("Line not detected.")
        else:
            self._debug_line(msg, rectified, line)

        ### 4. Calculate angle
        # TODO: Implement _calculate_angle() function.
        angle = calculate_angle(line)

        if angle is None:
            self.get_logger().warn("Angle not calculated.")
            angle = float('nan')

        # Publush angle
        angle_msg = Float32()
        angle_msg.data = float(angle)
        self.angle_pub.publish(angle_msg)

        self.get_logger().info(f"line angle: {angle:.2f} deg")

    def _debug_monitor(self, msg, image, top_left, top_right, bottom_left, bottom_right):
        debug_monitor = image.copy()

        monitor = np.array([top_left, top_right, bottom_left, bottom_right], dtype=np.float32)
        monitor_corners = monitor.astype(np.int32)

        cv2.polylines(
            debug_monitor,
            [monitor_corners],
            isClosed = True,
            color=(0,255,0),
            thickness=10,
        )

        for _, corner in enumerate(monitor_corners):
            x, y = corner
            cv2.circle(debug_monitor, (x, y), 10, (0, 0, 255), -1)
        
        debug_monitor_msg = self.bridge.cv2_to_imgmsg(debug_monitor, encoding="bgr8")
        debug_monitor_msg.header = msg.header
        self.monitor_pub.publish(debug_monitor_msg)

    def _debug_rectified(self, msg, rectified):
        debug_rectified_msg = self.bridge.cv2_to_imgmsg(rectified, encoding="bgr8")
        debug_rectified_msg.header = msg.header
        self.rectified_pub.publish(debug_rectified_msg)
        
    def _debug_line(self, msg, rectified, line):
        debug_line = rectified.copy()

        x1, y1, x2, y2 = line
        cv2.line(
            debug_line,
            (int(x1), int(y1)),
            (int(x2), int(y2)),
            (0, 0, 255),
            10,
        )

        debug_line_msg = self.bridge.cv2_to_imgmsg(debug_line, encoding="bgr8")
        debug_line_msg.header = msg.header
        self.line_pub.publish(debug_line_msg)

if __name__ == "__main__":
    rclpy.init()
    node = LineDetector()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
