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
#include <utility>
#include <vector>

#include <hardware_interface/hardware_info.hpp>
#include <hardware_interface/version.h>
#include <rclcpp/rclcpp.hpp>

#include <mujoco_ros2_control/mujoco_system_interface.hpp>
#include <mujoco_ros2_control_msgs/srv/reset_world.hpp>

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

// Two joints, for two components that share the scene.
constexpr const char* kTwoJointModel = R"(<?xml version="1.0"?>
<mujoco model="test_mit_position_actuator">
  <option timestep="0.002" gravity="0 0 0" integrator="implicitfast"/>
  <worldbody>
    <body name="link"><joint name="j" type="hinge" axis="0 0 1"/><geom type="box" size="0.1 0.02 0.02"/></body>
    <body name="link_k" pos="0 0.5 0"><joint name="k" type="hinge" axis="0 0 1"/><geom type="box" size="0.1 0.02 0.02"/></body>
  </worldbody>
  <actuator>
    <position name="j" joint="j" kp="100" kv="10"/>
    <position name="k" joint="k" kp="100" kv="10"/>
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
               const std::vector<std::string>& interfaces = { "position", "velocity", "effort" },
               const std::string& joint_name = "j")
  {
    std::ofstream(kTestModelPath) << model;
    hardware_interface::ComponentInfo joint;
    joint.name = joint_name;
    joint.type = "joint";
    for (const auto& name : interfaces)
    {
      joint.command_interfaces.push_back(hardware_interface::InterfaceInfo{ .name = name });
      joint.state_interfaces.push_back(hardware_interface::InterfaceInfo{ .name = name });
    }
    hardware_interface::HardwareInfo info;
    info.name = "mit_" + joint_name;
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

  // Calls the reset_world service. Returns what the physics thread applies before the next write.
  Staged reset_world()
  {
    const auto node = std::make_shared<rclcpp::Node>("mit_reset_client");
    const auto client =
        node->create_client<mujoco_ros2_control_msgs::srv::ResetWorld>("/mujoco_ros2_control_node/reset_world");
    EXPECT_TRUE(client->wait_for_service(std::chrono::seconds(5)));
    auto future = client->async_send_request(std::make_shared<mujoco_ros2_control_msgs::srv::ResetWorld::Request>());
    EXPECT_EQ(rclcpp::spin_until_future_complete(node, future, std::chrono::seconds(5)),
              rclcpp::FutureReturnCode::SUCCESS);
    EXPECT_TRUE(future.get()->success);
    return applied(component);
  }

  // Calls the reset_world service, then writes again with no mode switch, as after a real reset.
  Staged reset_world_and_write()
  {
    reset_world();
    // The reset set the commands to the reset state; the active controllers write theirs again.
    EXPECT_TRUE(commands.at("position")->set_value(q_des));
    EXPECT_TRUE(commands.at("velocity")->set_value(v_des));
    EXPECT_TRUE(commands.at("effort")->set_value(tau_ff));
    EXPECT_EQ(component.write(rclcpp::Time(0), period), hardware_interface::return_type::OK);
    return applied(component);
  }

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

// The start-up hold is not a position claim: the first claim on the joint ends it, so an effort
// controller started from start-up gets a plain motor and not kp*(q_start - q) on top of its torque.
TEST_F(MitPositionActuator, FirstClaimEndsStartupHold)
{
  Rig rig;
  ASSERT_EQ(rig.init_result, hardware_interface::CallbackReturn::SUCCESS);
  const Staged staged = rig.switch_and_write({ "j/effort" }, {});
  EXPECT_DOUBLE_EQ(staged.gain, 1.0);
  EXPECT_DOUBLE_EQ(staged.bias1, 0.0);
  EXPECT_DOUBLE_EQ(staged.bias2, 0.0);
  EXPECT_DOUBLE_EQ(staged.ctrl, Rig::tau_ff);
}

// MuJoCo clamps ctrl, which carries the velocity and effort terms: a ctrlrange would cut them silently.
// Position alone stays claimable, as its ctrl is q_des.
TEST_F(MitPositionActuator, CtrlrangeRefusesVelocityAndEffortClaims)
{
  std::string model = kTestModel;
  const std::string actuator = R"(kv="10")";
  model.replace(model.find(actuator), actuator.size(), R"(kv="10" ctrlrange="-0.5 0.5")");
  Rig rig(model);
  ASSERT_EQ(rig.init_result, hardware_interface::CallbackReturn::SUCCESS);
  EXPECT_EQ(rig.component.prepare_command_mode_switch({ "j/effort" }, {}), hardware_interface::return_type::ERROR);
  EXPECT_EQ(rig.component.prepare_command_mode_switch({ "j/velocity" }, {}), hardware_interface::return_type::ERROR);
  EXPECT_EQ(rig.component.prepare_command_mode_switch({ "j/position" }, {}), hardware_interface::return_type::OK);
}

// Position is optional: a joint that exports only effort initializes and drives as a plain motor.
TEST_F(MitPositionActuator, EffortOnlyJointDrivesAsMotor)
{
  Rig rig(kTestModel, { "effort" });
  ASSERT_EQ(rig.init_result, hardware_interface::CallbackReturn::SUCCESS);
  const Staged staged = rig.switch_and_write({ "j/effort" }, {});
  EXPECT_DOUBLE_EQ(staged.gain, 1.0);
  EXPECT_DOUBLE_EQ(staged.bias1, 0.0);
  EXPECT_DOUBLE_EQ(staged.bias2, 0.0);
  EXPECT_DOUBLE_EQ(staged.ctrl, Rig::tau_ff);
}

// A reset keeps the claims: the controllers that made them stay active, and no mode switch follows.
TEST_F(MitPositionActuator, ResetWorldKeepsClaims)
{
  Rig rig;
  ASSERT_EQ(rig.init_result, hardware_interface::CallbackReturn::SUCCESS);
  rig.switch_and_write({ "j/position", "j/velocity", "j/effort" }, {});
  Staged staged = rig.reset_world_and_write();
  EXPECT_DOUBLE_EQ(staged.gain, kKp);
  EXPECT_DOUBLE_EQ(staged.bias1, -kKp);
  EXPECT_DOUBLE_EQ(staged.bias2, -kKd);
  EXPECT_DOUBLE_EQ(staged.ctrl, Rig::q_des + (kKd * Rig::v_des + Rig::tau_ff) / kKp);

  // Effort alone gets no start-pose hold back from the reset.
  rig.switch_and_write({}, { "j/position", "j/velocity" });
  staged = rig.reset_world_and_write();
  EXPECT_DOUBLE_EQ(staged.gain, 1.0);
  EXPECT_DOUBLE_EQ(staged.bias1, 0.0);
  EXPECT_DOUBLE_EQ(staged.bias2, 0.0);
  EXPECT_DOUBLE_EQ(staged.ctrl, Rig::tau_ff);
}

// Until the next write, the reset ctrl is a start position. Gains an effort claim left would apply it as
// torque, so the reset restores the MJCF servo the ctrl belongs to.
TEST_F(MitPositionActuator, ResetWorldPairsStartCtrlWithMjcfGains)
{
  Rig rig;
  ASSERT_EQ(rig.init_result, hardware_interface::CallbackReturn::SUCCESS);
  rig.switch_and_write({ "j/effort" }, {});
  const Staged staged = rig.reset_world();
  EXPECT_DOUBLE_EQ(staged.gain, kKp);
  EXPECT_DOUBLE_EQ(staged.bias1, -kKp);
  EXPECT_DOUBLE_EQ(staged.bias2, -kKd);
}

// After a reset every component holds the position actuators it does not own. Its hold gains must not
// replace the owner's law: the owner's cached gains match, so it would never stage them again.
TEST_F(MitPositionActuator, ResetKeepsOtherComponentsGainsOffOwnedActuator)
{
  Rig owner(kTwoJointModel, { "position", "velocity", "effort" }, "j");
  ASSERT_EQ(owner.init_result, hardware_interface::CallbackReturn::SUCCESS);
  Rig other(kTwoJointModel, { "position", "velocity", "effort" }, "k");
  ASSERT_EQ(other.init_result, hardware_interface::CallbackReturn::SUCCESS);
  owner.switch_and_write({ "j/effort" }, {});
  owner.reset_world();
  ASSERT_EQ(owner.component.write(rclcpp::Time(0), owner.period), hardware_interface::return_type::OK);
  ASSERT_EQ(other.component.write(rclcpp::Time(0), other.period), hardware_interface::return_type::OK);
  const Staged staged = applied(owner.component);
  EXPECT_DOUBLE_EQ(staged.gain, 1.0);
  EXPECT_DOUBLE_EQ(staged.bias1, 0.0);
  EXPECT_DOUBLE_EQ(staged.bias2, 0.0);
}

// An effort-only joint exports no position state, and the scene has no keyframe: the hold must still
// target the scene's start pose, before the first write and after it.
TEST_F(MitPositionActuator, EffortOnlyJointHoldsSceneStartPose)
{
  std::string model = kTestModel;
  const std::string joint = R"(type="hinge")";
  model.replace(model.find(joint), joint.size(), R"(type="hinge" ref="0.5")");
  const std::string option = "<option";
  model.replace(model.find(option), option.size(), R"(<compiler angle="radian"/><option)");
  // Gravity on an offset link pulls it off the start pose. Following qpos would let it fall.
  for (const auto& [from, to] : std::vector<std::pair<std::string, std::string>>{
           { R"(gravity="0 0 0")", R"(gravity="0 0 -9.81")" },
           { R"(axis="0 0 1")", R"(axis="0 1 0")" },
           { R"(<geom type="box")", R"(<geom type="box" pos="0.1 0 0")" } })
  {
    model.replace(model.find(from), from.size(), to);
  }
  Rig rig(model, { "effort" });
  ASSERT_EQ(rig.init_result, hardware_interface::CallbackReturn::SUCCESS);
  EXPECT_DOUBLE_EQ(applied(rig.component).ctrl, 0.5);

  EXPECT_TRUE(rig.commands.at("effort")->set_value(0.0));
  for (int i = 0; i < 25; ++i)
  {
    EXPECT_EQ(rig.component.read(rclcpp::Time(0), rig.period), hardware_interface::return_type::OK);
    EXPECT_EQ(rig.component.write(rclcpp::Time(0), rig.period), hardware_interface::return_type::OK);
    std::this_thread::sleep_for(std::chrono::milliseconds(2));
  }
  EXPECT_DOUBLE_EQ(applied(rig.component).ctrl, 0.5);
}
#endif
