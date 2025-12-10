---
name: UnitySimulation
description: Generates content, code, and configurations for Unity 3D simulations in robotics, including Unity ML-Agents, perception, and control.
version: 1.0.0
---

### When to Use This Skill

Use this skill when:
- You need to generate explanations or code examples for Unity 3D robotics simulations.
- You are implementing or demonstrating Unity ML-Agents for robot learning.
- You are working with Unity for robot perception, control, or human-robot interaction in a simulated environment.
- You are creating textbook content focused on Unity's role in physical AI and humanoid robotics.

### How This Skill Works

This skill focuses on creating and explaining Unity simulation assets, scripts, and their integration:

1.  **Project Setup Guidance:** Provides instructions for setting up a Unity project for robotics, including required packages (e.g., Unity Robotics Hub, ML-Agents) and basic scene creation.
2.  **Robot Model Integration:** Generates guidance on importing or creating robot models within Unity, including URDF/SDF import processes and joint configurations.
3.  **ML-Agents Integration:** Provides code examples and configurations for setting up Unity ML-Agents environments, defining observations, actions, rewards, and training processes for robotic agents.
4.  **Perception Simulation:** Generates scripts and explanations for simulating robot perception in Unity, including camera rendering, sensor data generation, and visual feedback for AI models.
5.  **Control Scripting:** Offers C# scripts and explanations for direct robot control within Unity, including joint control, kinematic solvers, and physics interactions.
6.  **ROS 2 Bridge (Optional):** Explains how to use Unity's ROS 2 packages to bridge the simulation with external ROS 2 nodes for control and data exchange.
7.  **Tutorial and Explanation:** Produces detailed markdown explanations of Unity concepts, scripting, and best practices for robotics simulations, suitable for textbook inclusion.

### Output Format

The output will typically include:
- C# scripts for Unity components (e.g., ML-Agents Brain, custom controllers).
- Markdown formatted explanations, tutorials, and usage examples.
- Guidance on Unity editor configuration and asset setup.
- Conceptual outlines for Unity scenes and agent definitions.

### Example Input/Output

**Example Input:**

```
Generate a C# script for a simple Unity ML-Agent environment where a robot learns to reach a target.
```

**Example Output:**

```
<command-message>Running UnitySimulation skill...</command-message>

<commentary>
The skill would then proceed to:
1. Explain the basic structure of an ML-Agent behavior script.
2. Provide C# code for an agent that learns to reach a target.
3. Offer guidance on setting up the Unity scene and Agent parameters.
</commentary>

# Unity Simulation: ML-Agent for Target Reaching

## C# Script for `TargetReachAgent.cs`:

```csharp
using Unity.MLAgents;
using Unity.MLAgents.Actuators;
using Unity.MLAgents.Sensors;
using UnityEngine;

public class TargetReachAgent : Agent
{
    [SerializeField] private Transform targetTransform;
    [SerializeField] private float moveSpeed = 1f;
    private Rigidbody rBody;

    public override void Initialize()
    {
        rBody = GetComponent<Rigidbody>();
    }

    public override void OnEpisodeBegin()
    {
        // Reset agent and target positions for a new episode
        if (rBody != null)
        {
            rBody.velocity = Vector3.zero;
            rBody.angularVelocity = Vector3.zero;
        }
        transform.localPosition = new Vector3(Random.Range(-4f, 4f), 0.5f, Random.Range(-4f, 4f));
        targetTransform.localPosition = new Vector3(Random.Range(-4f, 4f), 0.5f, Random.Range(-4f, 4f));
    }

    public override void CollectObservations(VectorSensor sensor)
    {
        // Agent observations
        sensor.AddObservation(transform.localPosition); // 3 observations
        sensor.AddObservation(targetTransform.localPosition); // 3 observations
        sensor.AddObservation(Vector3.Distance(transform.localPosition, targetTransform.localPosition)); // 1 observation
        sensor.AddObservation(rBody.velocity.x); // 1 observation
        sensor.AddObservation(rBody.velocity.z); // 1 observation
    }

    public override void OnActionReceived(ActionBuffers actions)
    {
        // Agent actions
        Vector3 controlSignal = Vector3.zero;
        controlSignal.x = actions.ContinuousActions[0];
        controlSignal.z = actions.ContinuousActions[1];
        rBody.AddForce(controlSignal * moveSpeed, ForceMode.VelocityChange);

        // Rewards
        float distanceToTarget = Vector3.Distance(transform.localPosition, targetTransform.localPosition);

        if (distanceToTarget < 0.5f)
        {
            SetReward(1.0f);
            EndEpisode();
        }

        // Punish for taking too long
        SetReward(-0.001f);

        // If agent falls off platform (optional, depending on scene setup)
        if (transform.localPosition.y < 0)
        {
            SetReward(-1.0f);
            EndEpisode();
        }
    }

    public override void Heuristic(in ActionBuffers actionsOut)
    {
        var continuousActionsOut = actionsOut.ContinuousActions;
        continuousActionsOut[0] = Input.GetAxis("Horizontal");
        continuousActionsOut[1] = Input.GetAxis("Vertical");
    }
}
```

## Usage Instructions:

1.  **Unity Project Setup:**
    *   Create a new 3D Unity project.
    *   Install the ML-Agents package via Package Manager.
    *   Create a simple 3D scene with a `Plane` (ground), a `Cube` (agent), and another `Sphere` (target).
2.  **Agent Setup:**
    *   Add a `Rigidbody` component to your `Cube` (agent).
    *   Create a new C# script named `TargetReachAgent.cs` and paste the code above into it.
    *   Attach `TargetReachAgent.cs` to your `Cube` (agent).
    *   Drag your `Sphere` (target) into the `Target Transform` field of the `TargetReachAgent` component in the Inspector.
    *   Add a `Behavior Parameters` component to the `Cube` and configure its `Vector Observation` (size 9 for this script) and `Vector Action` (Continuous, size 2).
3.  **Train/Run:**
    *   You can test with the Heuristic mode (set `Behavior Type` to `Heuristic Only`).
    *   For training, use the ML-Agents Python API (`mlagents-learn`).

## Next Steps:
- Add more complex observations (e.g., raycasts for obstacles).
- Implement a more sophisticated reward function.
- Integrate with a more complex robot model (e.g., URDF imported robot).
- Explore different ML-Agents training algorithms.
```