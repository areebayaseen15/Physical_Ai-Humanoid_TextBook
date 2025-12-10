---
name: ROS2RobotControl
description: Generates ROS 2 packages, nodes, and code for controlling and simulating robotic systems.
version: 1.0.0
---

### When to Use This Skill

Use this skill when:
- You need to generate a new ROS 2 package or components (nodes, topics, services, actions).
- You are creating code examples for ROS 2 robot control, navigation, or perception within the textbook.
- You need to explain and demonstrate specific ROS 2 concepts with practical code.
- You are debugging or analyzing existing ROS 2 code for textbook content.

### How This Skill Works

This skill focuses on generating and explaining ROS 2 related content and code:

1.  **Package Generation:** Creates the basic structure of a ROS 2 package, including `package.xml`, `CMakeLists.txt`, and source directories for C++ or Python nodes.
2.  **Node/Component Development:** Generates boilerplate or functional code for ROS 2 nodes, publishers, subscribers, service servers/clients, and action servers/clients.
3.  **Robot Control Logic:** Provides code snippets and explanations for common robot control tasks, such as:
    *   Publishing commands to robot actuators.
    *   Subscribing to sensor data (e.g., IMU, LiDAR, camera).
    *   Implementing simple control loops or state machines.
4.  **Simulation Integration:** Generates code and instructions for integrating ROS 2 with simulation environments like Gazebo (e.g., publishing `JointState` messages, reading sensor data).
5.  **Tutorial and Explanation:** Produces detailed markdown explanations of ROS 2 concepts alongside the generated code, making it suitable for direct inclusion in the textbook.
6.  **Testing and Verification:** Includes basic testing mechanisms or suggestions to verify the generated ROS 2 code functionality.

### Output Format

The output will typically include:
- ROS 2 package directory structure with necessary configuration files.
- C++ or Python source files for ROS 2 nodes.
- Markdown formatted explanations and code usage examples.
- Command-line instructions for building and running the ROS 2 package.

### Example Input/Output

**Example Input:**

```
Generate a ROS 2 Python package named 'simple_mover' with a node that publishes Twist messages to control a mobile robot.
```

**Example Output:**

```
<command-message>Running ROS2RobotControl skill...</command-message>

<commentary>
The skill would then proceed to:
1. Create the `simple_mover` ROS 2 package structure.
2. Generate a Python node (`cmd_vel_publisher.py`) that publishes `geometry_msgs/msg/Twist`.
3. Provide markdown explanation of the code and how to run it.
</commentary>

# ROS 2 Package: simple_mover Setup Report

## Package Structure Created:
```
.claude/skills/ROS2RobotControl/simple_mover/
├── CMakeLists.txt
├── package.xml
└── simple_mover/
    ├── __init__.py
    └── cmd_vel_publisher.py
```

## `cmd_vel_publisher.py` (Python Node):
```python
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist

class CmdVelPublisher(Node):

    def __init__(self):
        super().__init__('cmd_vel_publisher')
        self.publisher_ = self.create_publisher(Twist, 'cmd_vel', 10)
        timer_period = 0.5  # seconds
        self.timer = self.create_timer(timer_period, self.timer_callback)
        self.linear_speed = 0.2  # m/s
        self.angular_speed = 0.0 # rad/s

    def timer_callback(self):
        msg = Twist()
        msg.linear.x = self.linear_speed
        msg.angular.z = self.angular_speed
        self.publisher_.publish(msg)
        self.get_logger().info(f'Publishing Linear: {msg.linear.x}, Angular: {msg.angular.z}')

def main(args=None):
    rclpy.init(args=args)
    node = CmdVelPublisher()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## `package.xml`:
```xml
<!-- ... (standard package.xml content) ... -->
<name>simple_mover</name>
<version>0.0.1</version>
<description>A simple ROS 2 package to control a mobile robot.</description>
<maintainer email="user@example.com">User</maintainer>
<license>Apache-2.0</license>

<buildtool_depend>ament_python</buildtool_depend>

<depend>rclpy</depend>
<depend>geometry_msgs</depend>
<test_depend>ament_copyright</test_depend>
<test_depend>ament_flake8</test_depend>
<test_depend>ament_pep257</test_depend>
<test_depend>python3-pytest</test_depend>

<export>
  <build_type>ament_python</build_type>
</export>
```

## Build and Run Instructions:

1.  **Navigate to your workspace:**
    ```bash
    cd ~/ros2_ws/src
    ```
2.  **Clone or create the package:** (If you created it manually, skip this step)
    ```bash
    # If cloning from a repo
    # git clone <your_repo_url>
    # If creating new
    # ros2 pkg create --build-type ament_python simple_mover
    ```
3.  **Copy the generated files:** Place `cmd_vel_publisher.py` in `simple_mover/simple_mover/` and update `package.xml` and `CMakeLists.txt` accordingly.
4.  **Install dependencies (if any new ones):**
    ```bash
    rosdep install -i --from-path src --rosdistro humble -y
    ```
5.  **Build the workspace:**
    ```bash
    cd ~/ros2_ws
    colcon build --packages-select simple_mover
    ```
6.  **Source the setup files:**
    ```bash
    source install/setup.bash
    ```
7.  **Run the node:**
    ```bash
    ros2 run simple_mover cmd_vel_publisher
    ```

## Next Steps:
- Integrate this package with a simulated robot in Gazebo.
- Add subscriber for odometry feedback.
- Implement a more sophisticated control algorithm.
```