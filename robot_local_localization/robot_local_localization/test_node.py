#!/usr/bin/env python3
import math
from typing import Optional

import rclpy
from rclpy.node import Node

from geometry_msgs.msg import Twist, PoseStamped
from nav_msgs.msg import Odometry, Path

from tf2_ros import Buffer, TransformListener
from tf_transformations import quaternion_from_euler


def yaw_to_quat(yaw: float):
    q = quaternion_from_euler(0.0, 0.0, yaw)
    return q  # [x,y,z,w]


class TestNode(Node):
    """
    - Publishes /cmd_vel to drive the robot in a rectangular path (open-loop timed).
    - Subscribes to:
        /predicted_odom      (motion model)
        /measurement/odom    (measurement model)
        /ekf/odom            (EKF fused)
      and also tries to compute "real" path using TF odom->base_link.
    - Publishes Path messages:
        /path/real
        /path/motion
        /path/meas
        /path/ekf
    """

    def __init__(self):
        super().__init__('test_node')

        # -----------------------
        # Parameters
        # -----------------------
        self.declare_parameter('cmd_vel_topic', '/cmd_vel')

        self.declare_parameter('motion_odom_topic', '/predicted_odom')
        self.declare_parameter('meas_odom_topic', '/measurement/odom')
        self.declare_parameter('ekf_odom_topic', '/ekf/odom')

        # If you want "real" from wheel encoder odom instead of TF, set it here
        self.declare_parameter('real_odom_topic', '/wheel_encoder/odom')
        self.declare_parameter('use_tf_as_real', True)

        self.declare_parameter('frame_id', 'odom')

        # Rectangle command profile (timed open-loop)
        self.declare_parameter('linear_speed', 0.20)   # m/s
        self.declare_parameter('angular_speed', 0.40)  # rad/s
        self.declare_parameter('side_time', 5.0)       # seconds moving straight
        self.declare_parameter('turn_time', 4.0)       # seconds turning (approx 90deg with w=0.4 -> 3.93s)
        self.declare_parameter('pause_time', 0.5)      # pause between segments
        self.declare_parameter('loops', 1)             # number of rectangles
        self.declare_parameter('publish_rate', 20.0)   # Hz command publishing

        # Path sampling
        self.declare_parameter('path_max_len', 5000)   # max poses in each Path
        self.declare_parameter('min_pose_dist', 0.02)  # meters
        self.declare_parameter('min_yaw_dist', 0.02)   # rad

        self.frame_id = self.get_parameter('frame_id').value
        self.use_tf_as_real = bool(self.get_parameter('use_tf_as_real').value)

        self.v = float(self.get_parameter('linear_speed').value)
        self.w = float(self.get_parameter('angular_speed').value)
        self.side_time = float(self.get_parameter('side_time').value)
        self.turn_time = float(self.get_parameter('turn_time').value)
        self.pause_time = float(self.get_parameter('pause_time').value)
        self.loops = int(self.get_parameter('loops').value)
        self.cmd_rate = float(self.get_parameter('publish_rate').value)

        self.path_max_len = int(self.get_parameter('path_max_len').value)
        self.min_pose_dist = float(self.get_parameter('min_pose_dist').value)
        self.min_yaw_dist = float(self.get_parameter('min_yaw_dist').value)

        # -----------------------
        # Publishers
        # -----------------------
        self.cmd_pub = self.create_publisher(Twist, self.get_parameter('cmd_vel_topic').value, 10)

        self.real_path_pub = self.create_publisher(Path, '/path/real', 10)
        self.motion_path_pub = self.create_publisher(Path, '/path/motion', 10)
        self.meas_path_pub = self.create_publisher(Path, '/path/meas', 10)
        self.ekf_path_pub = self.create_publisher(Path, '/path/ekf', 10)

        # -----------------------
        # Paths
        # -----------------------
        self.real_path = Path()
        self.motion_path = Path()
        self.meas_path = Path()
        self.ekf_path = Path()

        for p in (self.real_path, self.motion_path, self.meas_path, self.ekf_path):
            p.header.frame_id = self.frame_id

        # last appended poses for downsampling
        self._last_real = None
        self._last_motion = None
        self._last_meas = None
        self._last_ekf = None

        # -----------------------
        # Subscribers
        # -----------------------
        self.motion_sub = self.create_subscription(
            Odometry,
            self.get_parameter('motion_odom_topic').value,
            self._motion_cb,
            10
        )
        self.meas_sub = self.create_subscription(
            Odometry,
            self.get_parameter('meas_odom_topic').value,
            self._meas_cb,
            10
        )
        self.ekf_sub = self.create_subscription(
            Odometry,
            self.get_parameter('ekf_odom_topic').value,
            self._ekf_cb,
            10
        )

        # Optional real odom subscription if TF is disabled
        self.real_sub = None
        if not self.use_tf_as_real:
            self.real_sub = self.create_subscription(
                Odometry,
                self.get_parameter('real_odom_topic').value,
                self._real_odom_cb,
                10
            )

        # -----------------------
        # TF listener (for "real" path)
        # -----------------------
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # -----------------------
        # Command state machine
        # -----------------------
        self.segment = 0
        self.phase = 'pause'  # 'move', 'turn', 'pause', 'done'
        self.phase_time = 0.0
        self.loop_count = 0

        self.dt_cmd = 1.0 / max(self.cmd_rate, 1.0)
        self.timer = self.create_timer(self.dt_cmd, self._tick)

        self.get_logger().info('Test node started: will drive rectangular path and publish Paths.')

    # -----------------------
    # Helpers
    # -----------------------
    def _should_add(self, last, x, y, yaw):
        if last is None:
            return True
        lx, ly, lyaw = last
        d = math.hypot(x - lx, y - ly)
        dy = abs((yaw - lyaw + math.pi) % (2 * math.pi) - math.pi)
        return (d >= self.min_pose_dist) or (dy >= self.min_yaw_dist)

    def _append_pose(self, path: Path, last_store_name: str, x: float, y: float, yaw: float):
        last = getattr(self, last_store_name)
        if not self._should_add(last, x, y, yaw):
            return

        ps = PoseStamped()
        ps.header.stamp = self.get_clock().now().to_msg()
        ps.header.frame_id = self.frame_id
        ps.pose.position.x = x
        ps.pose.position.y = y
        ps.pose.position.z = 0.0
        q = yaw_to_quat(yaw)
        ps.pose.orientation.x = q[0]
        ps.pose.orientation.y = q[1]
        ps.pose.orientation.z = q[2]
        ps.pose.orientation.w = q[3]

        path.header.stamp = ps.header.stamp
        path.poses.append(ps)
        if len(path.poses) > self.path_max_len:
            path.poses.pop(0)

        setattr(self, last_store_name, (x, y, yaw))

    # -----------------------
    # Odom callbacks (paths)
    # -----------------------
    def _odom_to_xyyaw(self, msg: Odometry):
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        # yaw extraction
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        yaw = math.atan2(siny_cosp, cosy_cosp)
        return float(x), float(y), float(yaw)

    def _motion_cb(self, msg: Odometry):
        x, y, yaw = self._odom_to_xyyaw(msg)
        self._append_pose(self.motion_path, '_last_motion', x, y, yaw)
        self.motion_path_pub.publish(self.motion_path)

    def _meas_cb(self, msg: Odometry):
        x, y, yaw = self._odom_to_xyyaw(msg)
        self._append_pose(self.meas_path, '_last_meas', x, y, yaw)
        self.meas_path_pub.publish(self.meas_path)

    def _ekf_cb(self, msg: Odometry):
        x, y, yaw = self._odom_to_xyyaw(msg)
        self._append_pose(self.ekf_path, '_last_ekf', x, y, yaw)
        self.ekf_path_pub.publish(self.ekf_path)

    def _real_odom_cb(self, msg: Odometry):
        # only if TF disabled
        x, y, yaw = self._odom_to_xyyaw(msg)
        self._append_pose(self.real_path, '_last_real', x, y, yaw)
        self.real_path_pub.publish(self.real_path)

    # -----------------------
    # Main loop: command + TF real path
    # -----------------------
    def _tick(self):
        # 1) update "real" path from TF (if enabled)
        if self.use_tf_as_real:
            try:
                tfm = self.tf_buffer.lookup_transform(
                    self.frame_id,
                    'base_link',
                    rclpy.time.Time()  # latest
                )
                x = float(tfm.transform.translation.x)
                y = float(tfm.transform.translation.y)
                q = tfm.transform.rotation
                siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
                cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
                yaw = math.atan2(siny_cosp, cosy_cosp)

                self._append_pose(self.real_path, '_last_real', x, y, yaw)
                self.real_path_pub.publish(self.real_path)
            except Exception:
                # TF may not be available early; ignore quietly
                pass

        # 2) command state machine (rectangular path)
        if self.phase == 'done':
            self._publish_cmd(0.0, 0.0)
            return

        self.phase_time += self.dt_cmd

        if self.phase == 'pause':
            self._publish_cmd(0.0, 0.0)
            if self.phase_time >= self.pause_time:
                self.phase_time = 0.0
                # alternating: move then turn
                self.phase = 'move' if (self.segment % 2 == 0) else 'turn'
            return

        if self.phase == 'move':
            self._publish_cmd(self.v, 0.0)
            if self.phase_time >= self.side_time:
                self.phase_time = 0.0
                self.segment += 1
                self.phase = 'pause'
            return

        if self.phase == 'turn':
            self._publish_cmd(0.0, self.w)
            if self.phase_time >= self.turn_time:
                self.phase_time = 0.0
                self.segment += 1
                self.phase = 'pause'

                # after move+turn repeated 4 times: one rectangle finished
                # Each side uses: move,turn => 2 segments. For rectangle 4 sides => 8 segments.
                if self.segment >= 8:
                    self.segment = 0
                    self.loop_count += 1
                    if self.loop_count >= self.loops:
                        self.phase = 'done'
                        self.get_logger().info('Rectangular path finished. Holding stop command.')
            return

    def _publish_cmd(self, v: float, w: float):
        msg = Twist()
        msg.linear.x = float(v)
        msg.angular.z = float(w)
        self.cmd_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = TestNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
