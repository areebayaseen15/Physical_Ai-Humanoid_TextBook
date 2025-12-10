---
name: NVIDIAIsaacAIManipulation
description: Generates content, code, and configurations related to NVIDIA Isaac Sim and Isaac ROS for AI-driven perception and robot manipulation.
version: 1.0.0
---

### When to Use This Skill

Use this skill when:
- You need to generate explanations or code examples for NVIDIA Isaac Sim or Isaac ROS.
- You are implementing AI-driven perception tasks (e.g., object detection, segmentation) within Isaac.
- You are developing robot manipulation strategies (e.g., pick-and-place, grasping) using Isaac's tools.
- You are creating textbook content focused on NVIDIA's robotics platforms and AI capabilities.

### How This Skill Works

This skill focuses on generating and explaining NVIDIA Isaac-related concepts, code, and configurations:

1.  **Isaac Sim Environment Setup:** Provides instructions and configurations for setting up simulation environments in Isaac Sim, including assets, sensors, and lighting.
2.  **Isaac ROS Perception Pipeline:** Generates code and configurations for common perception tasks using Isaac ROS modules, such as:
    *   Camera-based object detection and tracking.
    *   3D point cloud processing and segmentation.
    *   Integration with deep learning models.
3.  **Robot Manipulation Scripting:** Provides Python scripts and explanations for controlling robot arms and grippers in Isaac Sim, implementing tasks like pick-and-place, and motion planning.
4.  **ROS 2 Bridge Integration:** Explains how to leverage the Isaac ROS Bridge to connect Isaac Sim simulations with the broader ROS 2 ecosystem for control and data exchange.
5.  **Tutorial and Explanation:** Produces detailed markdown explanations of NVIDIA Isaac concepts, API usage, and best practices, suitable for direct inclusion in the textbook.
6.  **Code Examples:** Generates functional Python scripts that demonstrate key Isaac Sim and Isaac ROS functionalities.

### Output Format

The output will typically include:
- Python scripts for Isaac Sim scenes, robot control, and perception tasks.
- ROS 2 launch files and configuration for Isaac ROS integration.
- Markdown formatted explanations, tutorials, and usage examples.
- Instructions for setting up and running Isaac Sim/ROS projects.

### Example Input/Output

**Example Input:**

```
Generate a Python script for Isaac Sim to simulate a UR5e robot performing a simple pick-and-place task.
```

**Example Output:**

```
<command-message>Running NVIDIAIsaacAIManipulation skill...</command-message>

<commentary>
The skill would then proceed to:
1. Generate a Python script that sets up a basic Isaac Sim scene.
2. Add a UR5e robot and a target object.
3. Implement a pick-and-place routine using Isaac Sim's APIs.
4. Provide markdown explanation of the script and how to run it.
</commentary>

# NVIDIA Isaac AI Perception & Manipulation Setup Report

## `isaac_pick_place.py` (Isaac Sim Python Script):
```python
from omni.isaac.kit import SimulationApp

# Start Isaac Sim
kit = SimulationApp({"headless": False})

import os
import numpy as np
from omni.isaac.core import World
from omni.isaac.urdf import RigidPrim
from omni.isaac.core.objects import DynamicCuboid
from pxr import Gf

class PickAndPlace(object):
    def __init__(self):
        self.world = World(stage_units_in_meters=1.0)
        self.world.scene.add_default_ground_plane()

        # Add UR5e robot
        self.robot = self.world.scene.add(RigidPrim(
            prim_path="/World/ur5e",
            name="ur5e",
            usd_path=os.path.join(kit.get_extension_path("omni.isaac.urdf"), "urdf", "ur5e_with_gripper.urdf"),
            position=Gf.Vec3d(0, 0, 0.9))
        )
        self.robot.initialize()

        # Add a cube to pick
        self.cube = self.world.scene.add(DynamicCuboid(
            prim_path="/World/cube",
            name="cube",
            position=np.array([0.5, 0.2, 0.05]),
            scale=np.array([0.1, 0.1, 0.1]),
            color=np.array([0, 0, 1]))
        )
        self.cube.initialize()

        self.world.reset()

    async def run_pick_and_place(self):
        # Simple pick and place logic (replace with actual inverse kinematics/motion planning)
        print("Simulating pick and place...")
        await self.world.step_async()

        # Move to above cube (simplified)
        print("Moving to pick pose...")
        await kit.update()

        # Pick (simplified - attach cube to gripper)
        print("Picking cube...")
        await kit.update()

        # Move to place pose (simplified)
        print("Moving to place pose...")
        await kit.update()

        # Place (simplified - detach cube)
        print("Placing cube...")
        await kit.update()

        print("Pick and place completed.")

    def cleanup(self):
        kit.close()

async def main():
    task = PickAndPlace()
    await task.run_pick_place()
    task.cleanup()

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
```

## Usage Instructions:

1.  **Save the script:** Save the Python code as `isaac_pick_place.py` in your Isaac Sim project directory.
2.  **Ensure dependencies:** Make sure you have Isaac Sim installed and configured, along with the `omni.isaac.urdf` extension.
3.  **Run the script:**
    ```bash
    python isaac_pick_place.py
    ```
    (Note: This requires an active Isaac Sim environment. You might need to run this from the Isaac Sim terminal or with appropriate environment setup.)

## Next Steps:
- Integrate actual inverse kinematics (IK) and motion planning for the UR5e.
- Add more sophisticated grasping techniques.
- Implement vision-based perception using Isaac ROS for object detection and pose estimation.
- Create a more complex scene with multiple objects.
```