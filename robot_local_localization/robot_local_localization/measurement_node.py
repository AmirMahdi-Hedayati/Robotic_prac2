#!/usr/bin/env python3

import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Imu
from nav_msgs.msg import Odometry


class MeasurementNode(Node):

    def __init__(self):
        super().__init__('measurement_node')

        # -----------------------
        # Parameters
        # -----------------------
        self.declare_parameter('imu_topic', '/zed/zed_node/imu/data_raw')
        self.declare_parameter('vo_topic', '/vo/odom')
        self.declare_parameter('measurement_topic', '/measurement/odom')

        # -----------------------
        # Internal buffers
        # -----------------------
        self.last_vo = None
        self.last_imu = None

        # -----------------------
        # Subscribers
        # -----------------------
        self.imu_sub = self.create_subscription(
            Imu,
            self.get_parameter('imu_topic').value,
            self.imu_callback,
            10
        )

        self.vo_sub = self.create_subscription(
            Odometry,
            self.get_parameter('vo_topic').value,
            self.vo_callback,
            10
        )

        # -----------------------
        # Publisher
        # -----------------------
        self.meas_pub = self.create_publisher(
            Odometry,
            self.get_parameter('measurement_topic').value,
            10
        )

        self.get_logger().info('Measurement node started')

    # -----------------------
    # Callbacks
    # -----------------------
    def imu_callback(self, msg: Imu):
        self.last_imu = msg
        self.publish_measurement_if_ready()

    def vo_callback(self, msg: Odometry):
        self.last_vo = msg
        self.publish_measurement_if_ready()

    # -----------------------
    # Measurement construction
    # -----------------------
    def publish_measurement_if_ready(self):
        if self.last_vo is None or self.last_imu is None:
            return

        meas = Odometry()

        # Use Visual Odometry header and frames (RTAB-Map)
        meas.header = self.last_vo.header
        meas.child_frame_id = self.last_vo.child_frame_id

        # Pose from RTAB-Map Visual Odometry
        meas.pose = self.last_vo.pose

        # Yaw rate from IMU (gyro z)
        meas.twist.twist.angular.z = self.last_imu.angular_velocity.z

        # Simple covariance example (yaw rate variance)
        meas.twist.covariance[35] = 0.01

        self.meas_pub.publish(meas)


def main(args=None):
    rclpy.init(args=args)
    node = MeasurementNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
