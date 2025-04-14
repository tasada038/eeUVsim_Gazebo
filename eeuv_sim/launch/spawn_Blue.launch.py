#!/usr/bin/env python3
"""
UCAT simulation launch file.

@author: Yuya Hamamatsu
@contact: yuya.hamamatsu@ŧaltech.ee
"""

import os

from ament_index_python.packages import get_package_share_directory
import launch_ros
import launch
from launch import LaunchDescription
from launch.actions import ExecuteProcess
from launch.substitutions import PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare
from launch_ros.actions import Node
import xacro


def generate_launch_description():

    # Robot model name
    robot_model = 'Blue'

    # Path to package definition
    uw_gazebo_path = get_package_share_directory('eeuv_sim')

    # Setting GAZEBO_MODEL_PATH so that Gazebo knows where to find the defined models
    os.environ['GAZEBO_MODEL_PATH'] = uw_gazebo_path + '/models'

    # World path definition
    world_path = os.path.join(uw_gazebo_path,
                              'worlds', 'diver_world.world')

    # Loading urdf robot_description
    # ---------------------------------------------
    xacro_file = os.path.join(uw_gazebo_path,
                              'urdf', robot_model,
                              'base.xacro')

    doc = xacro.parse(open(xacro_file))
    xacro.process_doc(doc)
    params = {'robot_description': doc.toxml()}
    # ---------------------------------------------


    node_robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        output='screen',
        parameters=[params]
    )

    # Spawning robot model in Gazebo
    spawn_entity = Node(package='gazebo_ros', executable='spawn_entity.py',
                        arguments=['-topic', 'robot_description',
                                   '-entity', robot_model,
                                   '-x', '0.0',
                                   '-y', '0.0',
                                   '-z', '-5.0',
                                   ],
                        output='screen')


    # Node to compute auv dynamics
    auv_motion = Node(package='eeuv_sim', executable='AUVMotion.py',
                        parameters=[{'yaml_dynamics': "BlueDynamics.yaml",
                                     'robot_model': robot_model}],
                        output='screen')
    
    
    # Node to move thrusters
    move_thruster = Node(package='eeuv_sim', executable='moveThruster.py',
                        parameters=[{'yaml_dynamics': "BlueDynamics.yaml"}],
                        output='screen')
    

    # Node to compute auv dynamics
    pressure = Node(package='eeuv_sim', executable='pressure.py',
                        parameters=[{'robot_model': robot_model}],
                        output='screen')

    # Node to convert forces to fin kinematic parameters
    return LaunchDescription([
        ExecuteProcess(
            cmd=['gazebo', '--verbose', '-s', 'libgazebo_ros_factory.so',  '-s', 'libgazebo_ros_init.so',
            '-s', 'libgazebo_ros_force_system.so', world_path],
            output='log'
        ),
        node_robot_state_publisher,
        spawn_entity,
        # pressure,
        auv_motion,
        move_thruster,
        ])