#include <chrono>
#include <cmath>
#include <memory>

#include "rclcpp/rclcpp.hpp"
#include "geometry_msgs/msg/twist.hpp"
#include "std_msgs/msg/float64.hpp"

using namespace std::chrono_literals;

class CmdVelToMotorRpmNode : public rclcpp::Node
{
public:
  CmdVelToMotorRpmNode()
  : Node("cmd_vel_to_motor_rpm_node")
  {
    // Parameters (match your URDF diff-drive values)
    wheel_separation_ = this->declare_parameter<double>("wheel_separation", 0.45); // meters
    wheel_radius_     = this->declare_parameter<double>("wheel_radius", 0.10);     // meters

    max_rpm_          = this->declare_parameter<double>("max_rpm", 0.0); // 0 => no clamp
    invert_left_      = this->declare_parameter<bool>("invert_left", false);
    invert_right_     = this->declare_parameter<bool>("invert_right", false);

    cmd_timeout_sec_  = this->declare_parameter<double>("cmd_timeout", 0.5); // seconds

    cmd_topic_        = this->declare_parameter<std::string>("cmd_topic", "/cmd_vel");
    left_topic_       = this->declare_parameter<std::string>("left_rpm_topic", "/left_motor_rpm");
    right_topic_      = this->declare_parameter<std::string>("right_rpm_topic", "/right_motor_rpm");

    // Publishers
    left_pub_ = this->create_publisher<std_msgs::msg::Float64>(left_topic_, rclcpp::QoS(10));
    right_pub_ = this->create_publisher<std_msgs::msg::Float64>(right_topic_, rclcpp::QoS(10));

    // Subscriber
    cmd_sub_ = this->create_subscription<geometry_msgs::msg::Twist>(
      cmd_topic_,
      rclcpp::QoS(10),
      std::bind(&CmdVelToMotorRpmNode::cmdVelCb, this, std::placeholders::_1)
    );

    last_cmd_time_ = this->now();

    // Watchdog timer: if cmd_vel stops -> publish zero
    watchdog_timer_ = this->create_wall_timer(
      50ms,
      std::bind(&CmdVelToMotorRpmNode::watchdog, this)
    );

    RCLCPP_INFO(this->get_logger(),
      "CmdVelToMotorRpmNode started. cmd_topic=%s left=%s right=%s wheel_sep=%.3f wheel_r=%.3f",
      cmd_topic_.c_str(), left_topic_.c_str(), right_topic_.c_str(),
      wheel_separation_, wheel_radius_);
  }

private:
  void cmdVelCb(const geometry_msgs::msg::Twist::SharedPtr msg)
  {
    last_cmd_time_ = this->now();

    const double v = msg->linear.x;   // m/s
    const double w = msg->angular.z;  // rad/s

    // Differential drive kinematics:
    // v_left  = v - w * (L/2)
    // v_right = v + w * (L/2)
    const double v_left = v - (w * wheel_separation_ * 0.5);
    const double v_right = v + (w * wheel_separation_ * 0.5);

    // Convert wheel linear speed (m/s) -> wheel angular speed (rad/s) -> RPM
    const double left_rpm  = radpsToRpm(v_left / wheel_radius_);
    const double right_rpm = radpsToRpm(v_right / wheel_radius_);

    publishRpm(left_rpm, right_rpm);
  }

  void watchdog()
  {
    const auto now = this->now();
    const double dt = (now - last_cmd_time_).seconds();

    if (dt > cmd_timeout_sec_) {
      // stop motors if command timed out
      publishRpm(0.0, 0.0);
    }
  }

  double radpsToRpm(double radps) const
  {
    return radps * (60.0 / (2.0 * M_PI));
  }

  double clamp(double x, double limit) const
  {
    if (limit <= 0.0) return x;
    if (x > limit) return limit;
    if (x < -limit) return -limit;
    return x;
  }

  void publishRpm(double left_rpm, double right_rpm)
  {
    if (invert_left_)  left_rpm = -left_rpm;
    if (invert_right_) right_rpm = -right_rpm;

    left_rpm = clamp(left_rpm, max_rpm_);
    right_rpm = clamp(right_rpm, max_rpm_);

    std_msgs::msg::Float64 lmsg;
    std_msgs::msg::Float64 rmsg;
    lmsg.data = left_rpm;
    rmsg.data = right_rpm;

    left_pub_->publish(lmsg);
    right_pub_->publish(rmsg);
  }

private:
  rclcpp::Subscription<geometry_msgs::msg::Twist>::SharedPtr cmd_sub_;
  rclcpp::Publisher<std_msgs::msg::Float64>::SharedPtr left_pub_;
  rclcpp::Publisher<std_msgs::msg::Float64>::SharedPtr right_pub_;
  rclcpp::TimerBase::SharedPtr watchdog_timer_;

  rclcpp::Time last_cmd_time_;

  double wheel_separation_{0.45};
  double wheel_radius_{0.10};
  double max_rpm_{0.0};
  bool invert_left_{false};
  bool invert_right_{false};
  double cmd_timeout_sec_{0.5};

  std::string cmd_topic_{"/cmd_vel"};
  std::string left_topic_{"/left_motor_rpm"};
  std::string right_topic_{"/right_motor_rpm"};
};

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<CmdVelToMotorRpmNode>());
  rclcpp::shutdown();
  return 0;
}
