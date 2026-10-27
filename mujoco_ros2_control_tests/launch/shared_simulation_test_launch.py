# Copyright 2026 Acumino
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction, Shutdown, TimerAction
from launch.substitutions import Command, FindExecutable, LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterFile, ParameterValue
from launch_ros.substitutions import FindPackageShare

DEVICES = ("device_a", "device_b")


def launch_setup(context, *args, **kwargs):
    pkg_share = FindPackageShare("mujoco_ros2_control_tests")
    parameters_file = PathJoinSubstitution([pkg_share, "config", "shared_simulation_controllers.yaml"])

    # One process, one controller_manager per device, one simulation shared by both.
    nodes = [
        Node(
            package="mujoco_ros2_control",
            executable="ros2_control_node",
            arguments=["--cm-namespaces", ",".join(DEVICES)],
            emulate_tty=True,
            output="both",
            parameters=[{"use_sim_time": True}, ParameterFile(parameters_file)],
            on_exit=Shutdown(),
        )
    ]

    for device in DEVICES:
        robot_description_content = Command(
            [
                PathJoinSubstitution([FindExecutable(name="xacro")]),
                " ",
                PathJoinSubstitution([pkg_share, "test_resources", "robot", "shared_simulation_robot.urdf"]),
                " device:=",
                device,
                " headless:=",
                LaunchConfiguration("headless"),
            ]
        )
        robot_description = {
            "robot_description": ParameterValue(value=robot_description_content.perform(context), value_type=str)
        }
        # Each manager reads its description from its own namespace. Humble's manager listens on
        # ~/robot_description instead, so point this device's publisher there.
        publisher = Node(
                package="robot_state_publisher",
                executable="robot_state_publisher",
                namespace=device,
                output="both",
                parameters=[robot_description, {"use_sim_time": True}],
                remappings=(
                    [("robot_description", "controller_manager/robot_description")]
                    if os.environ.get("ROS_DISTRO") == "humble"
                    else []
                ),
            )
        # device_b attaches after physics has run long enough for its uncontrolled joint to fall, so
        # the tests see whether it seeds and stores its start pose from the keyframe or from the fall.
        nodes.append(publisher if device == DEVICES[0] else TimerAction(period=2.0, actions=[publisher]))
        nodes.append(
            Node(
                package="controller_manager",
                executable="spawner",
                arguments=["-c", f"/{device}/controller_manager", "joint_state_broadcaster", "position_controller"],
                output="both",
            )
        )

    return nodes


def generate_launch_description():
    return LaunchDescription(
        [
            DeclareLaunchArgument("headless", default_value="true"),
            OpaqueFunction(function=launch_setup),
        ]
    )
