#!/usr/bin/env python3

import math
import rclpy
from rclpy.node import Node

from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Quaternion
from tf_transformations import quaternion_from_euler


class PredictionNode(Node):

    def __init__(self):
        super().__init__('prediction_node')

        # -----------------------
        # Parameters
        # -----------------------
        self.declare_parameter('dt', 0.02)               # [s]
        self.declare_parameter('cmd_vel_topic', '/cmd_vel')
        self.declare_parameter('odom_topic', '/predicted_odom')
        self.declare_parameter('frame_id', 'odom')
        self.declare_parameter('child_frame_id', 'base_link')

        self.dt = self.get_parameter('dt').value

        # -----------------------
        # State
        # -----------------------
        self.x = 0.0
        self.y = 0.0
        self.yaw = 0.0

        self.v = 0.0
        self.w = 0.0

        # -----------------------
        # ROS Interfaces
        # -----------------------
        self.cmd_sub = self.create_subscription(
            Twist,
            self.get_parameter('cmd_vel_topic').value,
            self.cmd_callback,
            10
        )

        self.odom_pub = self.create_publisher(
            Odometry,
            self.get_parameter('odom_topic').value,
            10
        )

        self.timer = self.create_timer(self.dt, self.predict)

        self.get_logger().info('Prediction node started')

    # -----------------------
    # Callbacks
    # -----------------------
    def cmd_callback(self, msg: Twist):
        self.v = msg.linear.x
        self.w = msg.angular.z

    def predict(self):
        # Prediction step
        self.x += self.v * math.cos(self.yaw) * self.dt
        self.y += self.v * math.sin(self.yaw) * self.dt
        self.yaw += self.w * self.dt

        self.publish_odom()

    def publish_odom(self):
        odom = Odometry()

        odom.header.stamp = self.get_clock().now().to_msg()
        odom.header.frame_id = self.get_parameter('frame_id').value
        odom.child_frame_id = self.get_parameter('child_frame_id').value

        odom.pose.pose.position.x = self.x
        odom.pose.pose.position.y = self.y

        q = quaternion_from_euler(0.0, 0.0, self.yaw)
        odom.pose.pose.orientation = Quaternion(
            x=q[0],
            y=q[1],
            z=q[2],
            w=q[3]
        )

        odom.twist.twist.linear.x = self.v
        odom.twist.twist.angular.z = self.w

        self.odom_pub.publish(odom)


def main(args=None):
    rclpy.init(args=args)
    node = PredictionNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
