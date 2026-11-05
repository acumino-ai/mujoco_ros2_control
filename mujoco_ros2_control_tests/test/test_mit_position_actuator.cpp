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
#include <vector>

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

// One joint with the given command interfaces (all three by default), fresh from on_init, commands set to
// the values below.
struct Rig
{
  static constexpr double q_des = 0.3;
  static constexpr double v_des = 0.2;
  static constexpr double tau_ff = 1.5;
  const rclcpp::Duration period{ 0, 2000000 };
  mujoco_ros2_control::MujocoSystemInterface component;
  std::map<std::string, hardware_interface::CommandInterface::SharedPtr> commands;
  hardware_interface::CallbackReturn init_result;

  explicit Rig(const std::string& model = kTestModel,
               const std::vector<std::string>& interfaces = { "position", "velocity", "effort" })
  {
    std::ofstream(kTestModelPath) << model;
    hardware_interface::ComponentInfo joint;
    joint.name = "j";
    joint.type = "joint";
    for (const auto& name : interfaces)
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

    init_result = component.on_init(params);
    if (init_result != hardware_interface::CallbackReturn::SUCCESS)
    {
      return;
    }
    for (const auto& command : component.on_export_command_interfaces())
    {
      commands[command->get_interface_name()] = command;
    }
    const std::map<std::string, double> values{ { "position", q_des }, { "velocity", v_des }, { "effort", tau_ff } };
    for (const auto& [name, command] : commands)
    {
      EXPECT_TRUE(command->set_value(values.at(name)));
    }
  }

  ~Rig() { std::filesystem::remove(kTestModelPath); }

  Staged switch_and_write(const std::vector<std::string>& start, const std::vector<std::string>& stop)
  {
    EXPECT_EQ(component.perform_command_mode_switch(start, stop), hardware_interface::return_type::OK);
    EXPECT_EQ(component.write(rclcpp::Time(0), period), hardware_interface::return_type::OK);
    return applied(component);
  }
};

}  // namespace

class MitPositionActuator : public ::testing::Test
{
protected:
  static void SetUpTestSuite() { rclcpp::init(0, nullptr); }
  static void TearDownTestSuite() { rclcpp::shutdown(); }
};

TEST_F(MitPositionActuator, ClaimedInterfacesCombineIntoOneLaw)
{
  Rig rig;
  ASSERT_EQ(rig.init_result, hardware_interface::CallbackReturn::SUCCESS);

  // All three, as hybrid_compliance_controller claims them: the full law.
  Staged staged = rig.switch_and_write({ "j/position", "j/velocity", "j/effort" }, {});
  EXPECT_DOUBLE_EQ(staged.gain, kKp);
  EXPECT_DOUBLE_EQ(staged.bias1, -kKp);
  EXPECT_DOUBLE_EQ(staged.bias2, -kKd);
  EXPECT_DOUBLE_EQ(staged.ctrl, Rig::q_des + (kKd * Rig::v_des + Rig::tau_ff) / kKp);

  // Effort alone, as forward_effort_controller claims it: a plain motor.
  staged = rig.switch_and_write({}, { "j/position", "j/velocity" });
  EXPECT_DOUBLE_EQ(staged.gain, 1.0);
  EXPECT_DOUBLE_EQ(staged.bias1, 0.0);
  EXPECT_DOUBLE_EQ(staged.bias2, 0.0);
  EXPECT_DOUBLE_EQ(staged.ctrl, Rig::tau_ff);

  // Velocity alone: kd servos the velocity.
  staged = rig.switch_and_write({ "j/velocity" }, { "j/effort" });
  EXPECT_DOUBLE_EQ(staged.gain, 1.0);
  EXPECT_DOUBLE_EQ(staged.bias1, 0.0);
  EXPECT_DOUBLE_EQ(staged.bias2, -kKd);
  EXPECT_DOUBLE_EQ(staged.ctrl, kKd * Rig::v_des);
}
#endif
