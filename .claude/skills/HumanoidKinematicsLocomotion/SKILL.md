---
name: HumanoidKinematicsLocomotion
description: Generates content, code, and explanations for humanoid robot kinematics (forward/inverse) and locomotion control strategies.
version: 1.0.0
---

### When to Use This Skill

Use this skill when:
- You need to generate explanations or code examples for humanoid robot kinematics (e.g., forward kinematics, inverse kinematics).
- You are developing or explaining locomotion control strategies for humanoid robots (e.g., walking, balancing).
- You are creating textbook content focused on the theoretical and practical aspects of humanoid movement.
- You need to visualize or simulate humanoid robot motion.

### How This Skill Works

This skill focuses on generating and explaining humanoid robot kinematics and locomotion:

1.  **Kinematics Explanations:** Provides detailed markdown explanations of forward kinematics (FK) and inverse kinematics (IK) concepts, including Denavit-Hartenberg parameters, Jacobians, and common IK solvers.
2.  **Kinematics Code Generation:** Generates Python or C++ code snippets to compute FK and IK for simplified humanoid leg or arm structures.
3.  **Locomotion Strategies:** Explains various humanoid locomotion techniques, such as Zero Moment Point (ZMP), Capture Point, and whole-body control, with theoretical background.
4.  **Balance and Stability:** Provides content on humanoid balance control, including strategies for maintaining stability during walking and disturbance rejection.
5.  **Trajectory Generation:** Generates simple walking trajectories or motion sequences for humanoid robots, with explanations of interpolation methods.
6.  **Simulation Integration:** Offers guidance or code examples for simulating humanoid robot kinematics and locomotion in environments like Gazebo or Isaac Sim (though the actual simulation setup might be handled by other skills).
7.  **Tutorial and Explanation:** Produces comprehensive markdown content suitable for direct inclusion in the textbook, combining theory with practical implementation.

### Output Format

The output will typically include:
- Python/C++ code for forward and inverse kinematics calculations.
- Markdown formatted explanations, mathematical derivations, and conceptual diagrams.
- Code snippets for basic locomotion control loops or trajectory generation.
- Usage examples and instructions for running the generated code.

### Example Input/Output

**Example Input:**

```
Explain forward kinematics for a 2-DOF planar humanoid arm and provide a Python code example.
```

**Example Output:**

```
<command-message>Running HumanoidKinematicsLocomotion skill...</command-message>

<commentary>
The skill would then proceed to:
1. Explain the FK concept for a 2-DOF arm.
2. Provide Python code for FK calculation.
3. Include a simple usage example.
</commentary>

# Humanoid Robot Kinematics & Locomotion Report

## Forward Kinematics for a 2-DOF Planar Arm

Forward Kinematics (FK) is the process of calculating the position and orientation of a robot's end-effector given the joint angles. For a simple 2-Degrees Of Freedom (2-DOF) planar arm, we consider two links connected by two revolute joints, moving in a 2D plane.

### Mathematical Derivation:

Let:
- `L1`, `L2` be the lengths of the first and second links.
- `θ1`, `θ2` be the joint angles of the first and second joints (relative to the previous link or a fixed reference).
- `(x, y)` be the coordinates of the end-effector.

The end-effector position can be calculated as:

`x = L1 * cos(θ1) + L2 * cos(θ1 + θ2)`
`y = L1 * sin(θ1) + L2 * sin(θ1 + θ2)`

### Python Code Example (`two_dof_arm_fk.py`):
```python
import numpy as np

def forward_kinematics_2dof(L1, L2, theta1_rad, theta2_rad):
    """
    Calculates the forward kinematics for a 2-DOF planar arm.

    Args:
        L1 (float): Length of the first link.
        L2 (float): Length of the second link.
        theta1_rad (float): Angle of the first joint in radians.
        theta2_rad (float): Angle of the second joint in radians.

    Returns:
        tuple: (x, y) coordinates of the end-effector.
    """
    x = L1 * np.cos(theta1_rad) + L2 * np.cos(theta1_rad + theta2_rad)
    y = L1 * np.sin(theta1_rad) + L2 * np.sin(theta1_rad + theta2_rad)
    return x, y

if __name__ == "__main__":
    # Example Usage:
    link1_length = 1.0
    link2_length = 0.8

    # Joint angles (in degrees) -> convert to radians for numpy
    joint1_angle_deg = 30
    joint2_angle_deg = 60

    theta1 = np.radians(joint1_angle_deg)
    theta2 = np.radians(joint2_angle_deg)

    end_effector_pos = forward_kinematics_2dof(link1_length, link2_length, theta1, theta2)
    print(f"For L1={link1_length}, L2={link2_length}, θ1={joint1_angle_deg}°, θ2={joint2_angle_deg}°:")
    print(f"End-effector position (x, y): ({end_effector_pos[0]:.3f}, {end_effector_pos[1]:.3f})")

    # Another example
    theta1 = np.radians(90)
    theta2 = np.radians(0)
    end_effector_pos = forward_kinematics_2dof(link1_length, link2_length, theta1, theta2)
    print(f"\nFor L1={link1_length}, L2={link2_length}, θ1={90}°, θ2={0}°:")
    print(f"End-effector position (x, y): ({end_effector_pos[0]:.3f}, {end_effector_pos[1]:.3f})")
```

## Usage Instructions:

1.  **Save the code:** Save the Python code as `two_dof_arm_fk.py`.
2.  **Run from terminal:**
    ```bash
    python two_dof_arm_fk.py
    ```

## Next Steps:
- Implement inverse kinematics for the 2-DOF planar arm.
- Extend to 3D kinematics for more complex humanoid structures.
- Integrate with a visualization tool to see the arm's movement.
```