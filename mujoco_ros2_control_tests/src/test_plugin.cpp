#include <pluginlib/class_list_macros.hpp>
#include <std_msgs/msg/u_int32.hpp>
#include "mujoco_ros2_control_plugins/mujoco_ros2_control_plugins_base.hpp"

namespace external_plugin_test
{

class TestPlugin : public mujoco_ros2_control_plugins::MuJoCoROS2ControlPluginBase
{
public:
  bool init(rclcpp::Node::SharedPtr node, const mjModel* model, mjData*) override
  {
    const int joint = mj_name2id(model, mjOBJ_JOINT, "plugin_driven");
    plugin_driven_dof_ = joint == -1 ? -1 : model->jnt_dofadr[joint];
    // Latched so a test subscribing after a reset still observes the count.
    world_reset_count_pub_ =
        node->create_publisher<std_msgs::msg::UInt32>("world_reset_count", rclcpp::QoS(1).transient_local());
    return true;
  }
  // Writes a force no component commands, so a test can check that staging keeps it.
  void update(const mjModel*, mjData* data) override
  {
    if (plugin_driven_dof_ != -1)
    {
      // test_plugin_output_staging.cpp expects this value.
      data->qfrc_applied[plugin_driven_dof_] = 5.0;
    }
  }
  void on_reset(mjData*) override
  {
    std_msgs::msg::UInt32 msg;
    msg.data = ++world_reset_count_;
    world_reset_count_pub_->publish(msg);
  }
  void cleanup() override
  {
    world_reset_count_pub_.reset();
  }

private:
  rclcpp::Publisher<std_msgs::msg::UInt32>::SharedPtr world_reset_count_pub_;
  uint32_t world_reset_count_{ 0 };
  int plugin_driven_dof_{ -1 };
};

}  // namespace external_plugin_test

PLUGINLIB_EXPORT_CLASS(external_plugin_test::TestPlugin, mujoco_ros2_control_plugins::MuJoCoROS2ControlPluginBase)
