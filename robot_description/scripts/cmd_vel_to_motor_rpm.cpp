#include <memory>
#include "rclcpp/rclcpp.hpp"
#include "geometry_msgs/msg/twist.hpp"
#include "std_msgs/msg/float64.hpp"

class MotorController : public rclcpp::Node
{
public:
    MotorController() : Node("motor_command_node")
    {
        // پارامترهای ربات - مقادیر بر اساس URDF شما
        this->declare_parameter("wheel_separation", 0.46); // 0.23 * 2
        this->declare_parameter("wheel_radius", 0.1);

        // سابسکرایبر برای دریافت سرعت خطی و زاویه‌ای
        cmd_vel_sub_ = this->create_subscription<geometry_msgs::msg::Twist>(
            "/cmd_vel", 10, std::bind(&MotorController::cmd_vel_callback, this, std::placeholders::_1));

        // پابلیشرها برای ارسال دستور به موتورها در Gazebo
        left_motor_pub_ = this->create_publisher<std_msgs::msg::Float64>("/left_motor_rpm", 10);
        right_motor_pub_ = this->create_publisher<std_msgs::msg::Float64>("/right_motor_rpm", 10);

        RCLCPP_INFO(this->get_logger(), "Motor Controller Node Started.");
    }

private:
    void cmd_vel_callback(const geometry_msgs::msg::Twist::SharedPtr msg)
    {
        double wheel_sep = this->get_parameter("wheel_separation").as_double();
        double wheel_rad = this->get_parameter("wheel_radius").as_double();

        // ۱. محاسبه سرعت خطی هر چرخ (m/s)
        // v_left = v - (omega * L / 2)
        // v_right = v + (omega * L / 2)
        double v_l = msg->linear.x - (msg->angular.z * wheel_sep / 2.0);
        double v_r = msg->linear.x + (msg->angular.z * wheel_sep / 2.0);

        // ۲. تبدیل سرعت خطی به سرعت زاویه‌ای چرخ (rad/s)
        // omega_wheel = v / R
        double omega_l = v_l / wheel_rad;
        double omega_r = v_r / wheel_rad;

        // نکته: اگر پلاگین Gazebo شما دقیقا RPM می‌خواهد، باید رادیان بر ثانیه را تبدیل کنید:
        // RPM = rad/s * 60 / (2 * PI)
        // اما معمولا JointController در Gazebo ورودی rad/s می‌گیرد.
        // با فرض اینکه نام تاپیک RPM است اما مقدار rad/s می‌پذیرد:
        
        auto left_msg = std_msgs::msg::Float64();
        auto right_msg = std_msgs::msg::Float64();

        left_msg.data = omega_l;
        right_msg.data = omega_r;

        left_motor_pub_->publish(left_msg);
        right_motor_pub_->publish(right_msg);
    }

    rclcpp::Subscription<geometry_msgs::msg::Twist>::SharedPtr cmd_vel_sub_;
    rclcpp::Publisher<std_msgs::msg::Float64>::SharedPtr left_motor_pub_;
    rclcpp::Publisher<std_msgs::msg::Float64>::SharedPtr right_motor_pub_;
};

int main(int argc, char * argv[])
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<MotorController>());
    rclcpp::shutdown();
    return 0;
}