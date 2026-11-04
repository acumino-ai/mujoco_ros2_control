// Copyright 2026 Acumino
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// A <position> actuator runs the MIT law kp*(q_des - q) + kd*(v_des - v) + tau_ff on whichever of
// position, velocity and effort are claimed together. A claim must add to the others: the exclusive
// switch would leave a hybrid controller that claims all three with only the last one. See
// write_mit_command() in mujoco_system_interface.cpp.

#include <gtest/gtest.h>

#include <chrono>
#include <filesystem>
#include <fstream>
#include <map>
#include <memory>
#include <string>
#include <thread>

#include <hardware_interface/hardware_info.hpp>
#include <hardware_interface/version.h>
#include <rclcpp/rclcpp.hpp>

#include <mujoco_ros2_control/mujoco_system_interface.hpp>

#define ROS_DISTRO_HUMBLE (HARDWARE_INTERFACE_VERSION_MAJOR < 3)

#if !ROS_DISTRO_HUMBLE
namespace
{

constexpr double kKp = 100.0;
constexpr double kKd = 10.0;

constexpr const char* kTestModel = R"(<?xml version="1.0"?>
<mujoco model="test_mit_position_actuator">
  <option timestep="0.002" gravity="0 0 0" integrator="implicitfast"/>
  <worldbody>
    <body name="link"><joint name="j" type="hinge" axis="0 0 1"/><geom type="box" size="0.1 0.02 0.02"/></body>
  </worldbody>
  <actuator>
    <position name="j" joint="j" kp="100" kv="10"/>
  </actuator>
</mujoco>
)";

const std::string kTestModelPath = "/tmp/test_mit_position_actuator_model.xml";

struct Staged
{
  double gain;
  double bias1;
  double bias2;
  double ctrl;
};

// The gains and ctrl the physics thread has applied, once a step has run after the write.
Staged applied(mujoco_ros2_control::MujocoSystemInterface& component)
{
  std::this_thread::sleep_for(std::chrono::milliseconds(50));
  mjModel* model = nullptr;
  mjData* data = nullptr;
  component.get_model(model);
  component.get_data(data);
  const Staged staged{ model->actuator_gainprm[0], model->actuator_biasprm[1], model->actuator_biasprm[2],
                       data->ctrl[0] };
  mj_deleteData(data);
  mj_deleteModel(model);
  return staged;
}

}  // namespace

TEST(MitPositionActuator, ClaimedInterfacesCombineIntoOneLaw)
{
  rclcpp::init(0, nullptr);
  {
    std::ofstream(kTestModelPath) << kTestModel;

    hardware_interface::ComponentInfo joint;
    joint.name = "j";
    joint.type = "joint";
    for (const char* name : { "position", "velocity", "effort" })
    {
      joint.command_interfaces.push_back(hardware_interface::InterfaceInfo{ .name = name });
      joint.state_interfaces.push_back(hardware_interface::InterfaceInfo{ .name = name });
    }
    hardware_interface::HardwareInfo info;
    info.name = "mit";
    info.type = "system";
    info.joints.push_back(joint);
    info.hardware_parameters["mujoco_model"] = kTestModelPath;
    info.hardware_parameters["headless"] = "true";
    hardware_interface::HardwareComponentInterfaceParams params;
    params.hardware_info = info;

    mujoco_ros2_control::MujocoSystemInterface component;
    ASSERT_EQ(component.on_init(params), hardware_interface::CallbackReturn::SUCCESS);
    std::map<std::string, hardware_interface::CommandInterface::SharedPtr> commands;
    for (const auto& command : component.on_export_command_interfaces())
    {
      commands[command->get_interface_name()] = command;
    }
    const double q_des = 0.3;
    const double v_des = 0.2;
    const double tau_ff = 1.5;
    ASSERT_TRUE(commands.at("position")->set_value(q_des));
    ASSERT_TRUE(commands.at("velocity")->set_value(v_des));
    ASSERT_TRUE(commands.at("effort")->set_value(tau_ff));
    const rclcpp::Duration period(0, 2000000);

    // All three, as hybrid_compliance_controller claims them: the full law.
    ASSERT_EQ(component.perform_command_mode_switch({ "j/position", "j/velocity", "j/effort" }, {}),
              hardware_interface::return_type::OK);
    ASSERT_EQ(component.write(rclcpp::Time(0), period), hardware_interface::return_type::OK);
    Staged staged = applied(component);
    EXPECT_DOUBLE_EQ(staged.gain, kKp);
    EXPECT_DOUBLE_EQ(staged.bias1, -kKp);
    EXPECT_DOUBLE_EQ(staged.bias2, -kKd);
    EXPECT_DOUBLE_EQ(staged.ctrl, q_des + (kKd * v_des + tau_ff) / kKp);

    // Effort alone, as forward_effort_controller claims it: a plain motor.
    ASSERT_EQ(component.perform_command_mode_switch({}, { "j/position", "j/velocity" }),
              hardware_interface::return_type::OK);
    ASSERT_EQ(component.write(rclcpp::Time(0), period), hardware_interface::return_type::OK);
    staged = applied(component);
    EXPECT_DOUBLE_EQ(staged.gain, 1.0);
    EXPECT_DOUBLE_EQ(staged.bias1, 0.0);
    EXPECT_DOUBLE_EQ(staged.bias2, 0.0);
    EXPECT_DOUBLE_EQ(staged.ctrl, tau_ff);

    // Velocity alone: kd servos the velocity.
    ASSERT_EQ(component.perform_command_mode_switch({ "j/velocity" }, { "j/effort" }),
              hardware_interface::return_type::OK);
    ASSERT_EQ(component.write(rclcpp::Time(0), period), hardware_interface::return_type::OK);
    staged = applied(component);
    EXPECT_DOUBLE_EQ(staged.gain, 1.0);
    EXPECT_DOUBLE_EQ(staged.bias1, 0.0);
    EXPECT_DOUBLE_EQ(staged.bias2, -kKd);
    EXPECT_DOUBLE_EQ(staged.ctrl, kKd * v_des);
  }
  std::filesystem::remove(kTestModelPath);
  rclcpp::shutdown();
}
#endif
