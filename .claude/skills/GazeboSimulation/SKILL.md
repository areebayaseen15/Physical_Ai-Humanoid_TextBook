---
name: GazeboSimulation
description: Generates Gazebo simulation environments, robot models (URDF/SDF), and integrates them with ROS 2 for realistic robot testing.
version: 1.0.0
---

### When to Use This Skill

Use this skill when:
- You need to create a new Gazebo simulation world or environment for robotic experiments.
- You are defining or modifying a robot's URDF/SDF model for simulation in Gazebo.
- You want to integrate ROS 2 control and sensor data with a Gazebo simulation.
- You are creating textbook content explaining Gazebo simulation principles and examples.

### How This Skill Works

This skill focuses on creating and managing Gazebo simulation assets and their ROS 2 integration:

1.  **World Generation:** Creates `.world` files for Gazebo, defining environments with obstacles, terrains, and light sources.
2.  **Robot Model Definition:** Generates or modifies URDF (Unified Robot Description Format) or SDF (Simulation Description Format) files to define robot kinematics, dynamics, visuals, and collisions.
3.  **Sensor Integration:** Adds virtual sensors (e.g., cameras, LiDAR, IMU) to robot models within Gazebo and configures their ROS 2 interfaces.
4.  **ROS 2 Control Integration:** Provides code and configurations for launching ROS 2 nodes that communicate with Gazebo, sending commands to joints and receiving sensor feedback.
5.  **Tutorial and Explanation:** Produces detailed markdown explanations of Gazebo concepts, model creation, and ROS 2 integration, suitable for textbook inclusion.
6.  **Launch File Generation:** Creates ROS 2 launch files to easily bring up Gazebo worlds, spawn robots, and start necessary ROS 2 nodes.

### Output Format

The output will typically include:
- `.world` files for Gazebo simulation environments.
- `.urdf` or `.sdf` files for robot models.
- ROS 2 configuration files and launch scripts.
- Python/C++ code snippets for ROS 2 Gazebo plugins or controllers.
- Markdown formatted explanations and usage examples.

### Example Input/Output

**Example Input:**

```
Create a simple Gazebo world with a flat ground plane and a single box obstacle.
Generate a basic URDF for a differential drive robot named 'my_diff_robot'.
```

**Example Output:**

```
<command-message>Running GazeboSimulation skill...</command-message>

<commentary>
The skill would then proceed to:
1. Create a `simple_world.world` file.
2. Generate a `my_diff_robot.urdf` file.
3. Provide markdown explanation and instructions.
</commentary>

# Gazebo Simulation Setup Report

## `simple_world.world` Content:
```xml
<?xml version="1.0" ?>
<sdf version="1.6">
  <world name="simple_world">
    <include>
      <uri>model://sun</uri>
    </include>
    <include>
      <uri>model://ground_plane</uri>
    </include>

    <model name="box_obstacle">
      <pose>2 0 0.5 0 0 0</pose>
      <link name="box_link">
        <collision name="collision">
          <geometry>
            <box><size>1 1 1</size></box>
          </geometry>
        </collision>
        <visual name="visual">
          <geometry>
            <box><size>1 1 1</size></box>
          </geometry>
          <material>
            <ambient>0.8 0.1 0.1 1</ambient>
            <diffuse>0.8 0.1 0.1 1</diffuse>
          </material>
        </visual>
      </link>
    </model>
  </world>
</sdf>
```

## `my_diff_robot.urdf` Content:
```xml
<?xml version="1.0" ?>
<robot name="my_diff_robot">
  <link name="base_link">
    <visual>
      <geometry>
        <box size="0.3 0.2 0.1"/>
      </geometry>
    </visual>
    <collision>
      <geometry>
        <box size="0.3 0.2 0.1"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="10"/>
      <inertia ixx="1" ixy="0" ixz="0" iyy="1" iyz="0" izz="1"/>
    </inertial>
  </link>

  <!-- Left Wheel -->
  <link name="left_wheel_link">
    <visual>
      <origin xyz="0 0.12 0" rpy="1.57079632679 0 0"/>
      <geometry>
        <cylinder radius="0.05" length="0.02"/>
      </geometry>
    </visual>
    <collision>
      <origin xyz="0 0.12 0" rpy="1.57079632679 0 0"/>
      <geometry>
        <cylinder radius="0.05" length="0.02"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="0.1"/>
      <inertia ixx="0.0001" ixy="0" ixz="0" iyy="0.0001" iyz="0" izz="0.0001"/>
    </inertial>
  </link>
  <joint name="left_wheel_joint" type="continuous">
    <parent link="base_link"/>
    <child link="left_wheel_link"/>
    <origin xyz="-0.1 0.12 0" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
  </joint>

  <!-- Right Wheel -->
  <link name="right_wheel_link">
    <visual>
      <origin xyz="0 -0.12 0" rpy="1.57079632679 0 0"/>
      <geometry>
        <cylinder radius="0.05" length="0.02"/>
      </geometry>
    </visual>
    <collision>
      <origin xyz="0 -0.12 0" rpy="1.57079632679 0 0"/>
      <geometry>
        <cylinder radius="0.05" length="0.02"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="0.1"/>
      <inertia ixx="0.0001" ixy="0" ixz="0" iyy="0.0001" iyz="0" izz="0.0001"/>
    </inertial>
  </link>
  <joint name="right_wheel_joint" type="continuous">
    <parent link="base_link"/>
    <child link="right_wheel_link"/>
    <origin xyz="-0.1 -0.12 0" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
  </joint>

  <!-- Gazebo specific tags for differential drive control -->
  <gazebo>
    <plugin name="differential_drive_controller" filename="libgazebo_ros_diff_drive.so">
      <ros>\n        <namespace>/my_diff_robot</namespace>\n        <remapping>cmd_vel:=cmd_vel</remapping>\n        <remapping>odom:=odom</remapping>\n      </ros>
      <left_joint>left_wheel_joint</left_joint>
      <right_joint>right_wheel_joint</right_joint>
      <wheel_separation>0.24</wheel_separation>
      <wheel_radius>0.05</wheel_radius>
      <publish_odom>true</publish_odom>
      <publish_wheel_tf>true</publish_wheel_tf>
      <odometry_frame>odom</odometry_frame>
      <robot_base_frame>base_link</robot_base_base_frame>
    </plugin>
  </gazebo>

</robot>
```

## Usage Instructions:

1.  **Save the files:** Save the `.world` and `.urdf` content into appropriate ROS 2 package directories (e.g., `my_robot_description/worlds` and `my_robot_description/urdf`).
2.  **Launch Gazebo with the world:**
    ```bash
    gazebo --verbose simple_world.world
    ```
3.  **Spawn the robot in Gazebo (requires `ros2_gazebo` package):**
    ```bash
    ros2 run gazebo_ros spawn_entity.py -entity my_diff_robot -file my_diff_robot.urdf -x 0 -y 0 -z 0.1
    ```
4.  **Send Twist commands (e.g., using `ROS2RobotControl` skill output):**
    ```bash
    ros2 topic pub /my_diff_robot/cmd_vel geometry_msgs/msg/Twist '{linear: {x: 0.1, y: 0.0, z: 0.0}, angular: {x: 0.0, y: 0.0, z: 0.5}}' --once
    ```

## Next Steps:
- Add more complex models and environments.
- Integrate advanced sensor types and their plugins.
- Develop custom Gazebo plugins for specific behaviors.
```