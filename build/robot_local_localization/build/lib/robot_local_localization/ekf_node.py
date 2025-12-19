#!/usr/bin/env python3

import math
import rclpy
from rclpy.node import Node

from nav_msgs.msg import Odometry
from geometry_msgs.msg import TransformStamped

from tf2_ros import TransformBroadcaster
from tf_transformations import quaternion_from_euler


def wrap_to_pi(a: float) -> float:
    while a <= -math.pi:
        a += 2.0 * math.pi
    while a > math.pi:
        a -= 2.0 * math.pi
    return a


def yaw_from_quat(q) -> float:
    # q: geometry_msgs.msg.Quaternion
    # yaw from quaternion (assuming planar motion)
    # yaw = atan2(2(wz + xy), 1 - 2(y^2 + z^2))
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)


class EkfNode(Node):
    """
    State: x = [px, py, yaw]
    Predict: from motion_model Odometry.twist (v, w)
    Update:  from measurement_model Odometry.pose (px, py, yaw)
    Output:  /ekf/odom (nav_msgs/Odometry) + TF odom->base_link
    """

    def __init__(self):
        super().__init__('ekf_node')

        # -----------------------
        # Parameters
        # -----------------------
        self.declare_parameter('motion_odom_topic', '/predicted_odom')
        self.declare_parameter('meas_odom_topic', '/measurement/odom')
        self.declare_parameter('ekf_odom_topic', '/ekf/odom')

        self.declare_parameter('frame_id', 'odom')
        self.declare_parameter('child_frame_id', 'base_link')

        # Process noise (control noise)
        self.declare_parameter('sigma_v', 0.10)       # m/s
        self.declare_parameter('sigma_w', 0.10)       # rad/s

        # Measurement noise fallback (if measurement covariance is zero)
        self.declare_parameter('r_x', 0.05)           # m
        self.declare_parameter('r_y', 0.05)           # m
        self.declare_parameter('r_yaw', 0.10)         # rad

        self.sigma_v = float(self.get_parameter('sigma_v').value)
        self.sigma_w = float(self.get_parameter('sigma_w').value)

        self.r_x = float(self.get_parameter('r_x').value)
        self.r_y = float(self.get_parameter('r_y').value)
        self.r_yaw = float(self.get_parameter('r_yaw').value)

        # -----------------------
        # EKF state
        # -----------------------
        self.x = [0.0, 0.0, 0.0]  # px, py, yaw

        # Covariance P (3x3)
        self.P = [
            [0.1, 0.0, 0.0],
            [0.0, 0.1, 0.0],
            [0.0, 0.0, 0.2],
        ]

        self.last_pred_stamp = None  # rclpy.time.Time

        # Last control
        self.last_v = 0.0
        self.last_w = 0.0

        # -----------------------
        # ROS Interfaces
        # -----------------------
        self.motion_sub = self.create_subscription(
            Odometry,
            self.get_parameter('motion_odom_topic').value,
            self.motion_cb,
            10
        )

        self.meas_sub = self.create_subscription(
            Odometry,
            self.get_parameter('meas_odom_topic').value,
            self.meas_cb,
            10
        )

        self.odom_pub = self.create_publisher(
            Odometry,
            self.get_parameter('ekf_odom_topic').value,
            10
        )

        self.tf_broadcaster = TransformBroadcaster(self)

        self.get_logger().info('EKF node started')

    # -----------------------
    # Small 3x3 linear algebra helpers
    # -----------------------
    @staticmethod
    def mat33_mul(A, B):
        return [[sum(A[i][k] * B[k][j] for k in range(3)) for j in range(3)] for i in range(3)]

    @staticmethod
    def mat33_add(A, B):
        return [[A[i][j] + B[i][j] for j in range(3)] for i in range(3)]

    @staticmethod
    def mat33_sub(A, B):
        return [[A[i][j] - B[i][j] for j in range(3)] for i in range(3)]

    @staticmethod
    def mat33_T(A):
        return [[A[j][i] for j in range(3)] for i in range(3)]

    @staticmethod
    def mat33_eye():
        return [[1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0]]

    @staticmethod
    def mat33_inv(A):
        # Inverse of 3x3 using adjugate/determinant (OK for EKF small matrix)
        a, b, c = A[0]
        d, e, f = A[1]
        g, h, i = A[2]
        det = (a*(e*i - f*h) - b*(d*i - f*g) + c*(d*h - e*g))
        if abs(det) < 1e-12:
            return None
        invdet = 1.0 / det
        adj = [
            [(e*i - f*h), -(b*i - c*h),  (b*f - c*e)],
            [-(d*i - f*g), (a*i - c*g), -(a*f - c*d)],
            [(d*h - e*g), -(a*h - b*g),  (a*e - b*d)]
        ]
        return [[adj[r][c] * invdet for c in range(3)] for r in range(3)]

    @staticmethod
    def mat33_vec_mul(A, v):
        return [sum(A[i][k] * v[k] for k in range(3)) for i in range(3)]

    # -----------------------
    # EKF steps
    # -----------------------
    def predict(self, v: float, w: float, dt: float):
        px, py, yaw = self.x
        c = math.cos(yaw)
        s = math.sin(yaw)

        # State prediction
        px = px + v * c * dt
        py = py + v * s * dt
        yaw = wrap_to_pi(yaw + w * dt)
        self.x = [px, py, yaw]

        # Jacobian F = d f / d x
        F = [
            [1.0, 0.0, -v * s * dt],
            [0.0, 1.0,  v * c * dt],
            [0.0, 0.0,  1.0]
        ]

        # Control Jacobian G (3x2) -> build Q = G Qu G^T
        # G = [[c*dt, 0],
        #      [s*dt, 0],
        #      [0,   dt]]
        # Qu = diag(sigma_v^2, sigma_w^2)
        sv2 = self.sigma_v * self.sigma_v
        sw2 = self.sigma_w * self.sigma_w

        # Q = [[(c*dt)^2 sv2, (c*dt)(s*dt) sv2, 0],
        #      [(s*dt)(c*dt) sv2, (s*dt)^2 sv2, 0],
        #      [0, 0, (dt)^2 sw2]]
        Q = [
            [(c*dt)*(c*dt)*sv2, (c*dt)*(s*dt)*sv2, 0.0],
            [(s*dt)*(c*dt)*sv2, (s*dt)*(s*dt)*sv2, 0.0],
            [0.0,               0.0,               (dt*dt)*sw2]
        ]

        # P = F P F^T + Q
        FP = self.mat33_mul(F, self.P)
        FPFt = self.mat33_mul(FP, self.mat33_T(F))
        self.P = self.mat33_add(FPFt, Q)

    def update(self, z, R):
        # H = I (we measure [x, y, yaw] directly)
        # y = z - x (with yaw wrapped)
        y = [z[0] - self.x[0], z[1] - self.x[1], wrap_to_pi(z[2] - self.x[2])]

        # S = P + R
        S = self.mat33_add(self.P, R)
        S_inv = self.mat33_inv(S)
        if S_inv is None:
            self.get_logger().warn('S not invertible, skip update')
            return

        # K = P S^-1
        K = self.mat33_mul(self.P, S_inv)

        # x = x + K y
        Ky = self.mat33_vec_mul(K, y)
        self.x = [self.x[0] + Ky[0], self.x[1] + Ky[1], wrap_to_pi(self.x[2] + Ky[2])]

        # P = (I - K) P  (since H=I)
        I = self.mat33_eye()
        IminusK = self.mat33_sub(I, K)
        self.P = self.mat33_mul(IminusK, self.P)

    # -----------------------
    # Callbacks
    # -----------------------
    def motion_cb(self, msg: Odometry):
        # Use twist from predicted odom as control
        v = float(msg.twist.twist.linear.x)
        w = float(msg.twist.twist.angular.z)

        # Determine dt from stamps
        stamp = msg.header.stamp
        t = rclpy.time.Time.from_msg(stamp)

        if self.last_pred_stamp is None:
            self.last_pred_stamp = t
            self.last_v = v
            self.last_w = w
            return

        dt = (t - self.last_pred_stamp).nanoseconds * 1e-9
        if dt <= 1e-4:
            dt = 1e-4

        self.predict(v, w, dt)
        self.last_pred_stamp = t
        self.last_v = v
        self.last_w = w

        # publish after predict (so RViz shows something even before updates)
        self.publish_ekf(t)

    def meas_cb(self, msg: Odometry):
        # Measurement z from measurement model odom pose
        px = float(msg.pose.pose.position.x)
        py = float(msg.pose.pose.position.y)
        yaw = yaw_from_quat(msg.pose.pose.orientation)

        z = [px, py, yaw]

        # Build R from measurement covariance if valid, else fallback params
        cov = msg.pose.covariance
        # ROS pose.covariance indices: x=0, y=7, yaw(about z)=35
        rx = cov[0] if cov[0] > 0.0 else (self.r_x * self.r_x)
        ry = cov[7] if cov[7] > 0.0 else (self.r_y * self.r_y)
        ryaw = cov[35] if cov[35] > 0.0 else (self.r_yaw * self.r_yaw)

        R = [
            [rx, 0.0, 0.0],
            [0.0, ry, 0.0],
            [0.0, 0.0, ryaw]
        ]

        self.update(z, R)

        # publish at measurement time (or now)
        t = rclpy.time.Time.from_msg(msg.header.stamp)
        self.publish_ekf(t)

    # -----------------------
    # Publishing (Odometry + TF)
    # -----------------------
    def publish_ekf(self, t: rclpy.time.Time):
        odom = Odometry()
        odom.header.stamp = t.to_msg()
        odom.header.frame_id = self.get_parameter('frame_id').value
        odom.child_frame_id = self.get_parameter('child_frame_id').value

        odom.pose.pose.position.x = self.x[0]
        odom.pose.pose.position.y = self.x[1]
        odom.pose.pose.position.z = 0.0

        q = quaternion_from_euler(0.0, 0.0, self.x[2])
        odom.pose.pose.orientation.x = q[0]
        odom.pose.pose.orientation.y = q[1]
        odom.pose.pose.orientation.z = q[2]
        odom.pose.pose.orientation.w = q[3]

        # Put P into covariance (x,y,yaw)
        for i in range(36):
            odom.pose.covariance[i] = 0.0
        odom.pose.covariance[0] = self.P[0][0]
        odom.pose.covariance[7] = self.P[1][1]
        odom.pose.covariance[35] = self.P[2][2]

        odom.twist.twist.linear.x = self.last_v
        odom.twist.twist.angular.z = self.last_w

        self.odom_pub.publish(odom)

        # TF so RViz can visualize
        tfm = TransformStamped()
        tfm.header.stamp = odom.header.stamp
        tfm.header.frame_id = odom.header.frame_id
        tfm.child_frame_id = odom.child_frame_id
        tfm.transform.translation.x = self.x[0]
        tfm.transform.translation.y = self.x[1]
        tfm.transform.translation.z = 0.0
        tfm.transform.rotation.x = q[0]
        tfm.transform.rotation.y = q[1]
        tfm.transform.rotation.z = q[2]
        tfm.transform.rotation.w = q[3]
        self.tf_broadcaster.sendTransform(tfm)


def main(args=None):
    rclpy.init(args=args)
    node = EkfNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
