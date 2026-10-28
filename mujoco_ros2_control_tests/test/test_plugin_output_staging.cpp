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

// Two components share one scene and only the first runs plugins. TestPlugin writes a force on the
// plugin_driven joint. The second component's write() must not stage its own stale value over that
// force, also when it drives that joint. See stage_control_data() in mujoco_system_interface.cpp.

#include <gtest/gtest.h>

#include <chrono>
#include <filesystem>
#include <fstream>
#include <memory>
#include <string>
#include <thread>

#include <hardware_interface/version.h>
#include <hardware_interface/hardware_info.hpp>
#include <rclcpp/rclcpp.hpp>

#include <mujoco_ros2_control/mujoco_system_interface.hpp>

#define ROS_DISTRO_HUMBLE (HARDWARE_INTERFACE_VERSION_MAJOR < 3)

namespace
{

constexpr const char* kTestModel = R"(<?xml version="1.0"?>
<mujoco model="test_plugin_output_staging">
  <option timestep="0.002" gravity="0 0 0"/>
  <worldbody>
    <body name="a_link"><joint name="a" type="slide" axis="1 0 0"/><geom type="box" size="0.02 0.02 0.02"/></body>
    <body name="b_link" pos="0 0.1 0"><joint name="b" type="slide" axis="1 0 0"/><geom type="box" size="0.02 0.02 0.02"/></body>
    <body name="c_link" pos="0 0.2 0">
      <joint name="plugin_driven" type="slide" axis="1 0 0"/><geom type="box" size="0.02 0.02 0.02"/>
    </body>
  </worldbody>
  <actuator>
    <motor name="a" joint="a"/>
    <motor name="b" joint="b"/>
    <motor name="plugin_driven" joint="plugin_driven"/>
  </actuator>
</mujoco>
)";

constexpr const char* kPluginParams = R"(/**:
  ros__parameters:
    mujoco_plugins:
      test_plugin:
        type: "external_plugin_test/TestPlugin"
)";

// TestPlugin writes this force on the plugin_driven joint.
constexpr double kPluginForce = 5.0;

const std::string kTestModelPath = "/tmp/test_plugin_output_staging_model.xml";
const std::string kPluginParamsPath = "/tmp/test_plugin_output_staging_params.yaml";

void write_file(const std::string& path, const char* content)
{
  std::ofstream file(path);
  file << content;
}

hardware_interface::CallbackReturn init_component(mujoco_ros2_control::MujocoSystemInterface& component,
                                                  const std::string& joint_name)
{
  hardware_interface::ComponentInfo joint;
  joint.name = joint_name;
  joint.type = "joint";
  joint.command_interfaces.push_back(hardware_interface::InterfaceInfo{ .name = "effort" });
  joint.state_interfaces.push_back(hardware_interface::InterfaceInfo{ .name = "position" });

  hardware_interface::HardwareInfo info;
  info.name = joint_name;
  info.type = "system";
  info.joints.push_back(joint);
  info.hardware_parameters["mujoco_model"] = kTestModelPath;
  info.hardware_parameters["headless"] = "true";
  info.hardware_parameters["pids_config_file"] = kPluginParamsPath;
#if ROS_DISTRO_HUMBLE
  return component.on_init(info);
#else
  hardware_interface::HardwareComponentInterfaceParams params;
  params.hardware_info = info;
  return component.on_init(params);
#endif
}

}  // namespace

namespace
{

// The first component drives joint a and runs the plugins. The second drives `second_joint`.
void expect_plugin_force_survives(const std::string& second_joint)
{
  rclcpp::init(0, nullptr);
  write_file(kTestModelPath, kTestModel);
  write_file(kPluginParamsPath, kPluginParams);
  {
    mujoco_ros2_control::MujocoSystemInterface first;
    ASSERT_EQ(init_component(first, "a"), hardware_interface::CallbackReturn::SUCCESS);
    mujoco_ros2_control::MujocoSystemInterface second;
    ASSERT_EQ(init_component(second, second_joint), hardware_interface::CallbackReturn::SUCCESS);

    // The first write claims the plugin driver. The second write is the one that used to clear the force.
    const rclcpp::Duration period(0, 2000000);
    ASSERT_EQ(first.write(rclcpp::Time(0), period), hardware_interface::return_type::OK);
    ASSERT_EQ(second.write(rclcpp::Time(0), period), hardware_interface::return_type::OK);
    std::this_thread::sleep_for(std::chrono::milliseconds(100));

    mjData* data = nullptr;
    mjModel* model = nullptr;
    first.get_model(model);
    first.get_data(data);
    const int dof = model->jnt_dofadr[mj_name2id(model, mjOBJ_JOINT, "plugin_driven")];
    EXPECT_DOUBLE_EQ(data->qfrc_applied[dof], kPluginForce);
    mj_deleteData(data);
    mj_deleteModel(model);
  }
  std::filesystem::remove(kTestModelPath);
  std::filesystem::remove(kPluginParamsPath);
  rclcpp::shutdown();
}

}  // namespace

TEST(PluginOutputStaging, ComponentWithoutPluginsKeepsPluginForceOnUnownedJoint)
{
  expect_plugin_force_survives("b");
}

// A component once claimed the qfrc_applied of the joints it drives and staged its stale value there.
TEST(PluginOutputStaging, ComponentWithoutPluginsKeepsPluginForceOnItsOwnJoint)
{
  expect_plugin_force_survives("plugin_driven");
}
