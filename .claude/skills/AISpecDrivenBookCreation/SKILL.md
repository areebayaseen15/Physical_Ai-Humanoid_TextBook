---
name: AISpecDrivenBookCreation
description: Automates the creation of textbook content following a spec-driven development approach, from outlining to content generation, review, and editing.
version: 1.0.0
---

### When to Use This Skill

Use this skill when:
- You need to generate a new chapter or a specific section of the "Physical AI & Humanoid Robotics" textbook.
- You have a clear specification (e.g., chapter title, learning objectives, outline) for the content to be created.
- You want to refine or review existing book content for accuracy, clarity, and adherence to Docusaurus standards.
- You need to generate code examples, diagrams, or structured explanations for technical robotics concepts.

### How This Skill Works

This skill orchestrates a multi-step process using various subagents and internal tools to produce high-quality, spec-driven textbook content:

1.  **Input Analysis:** It takes a detailed prompt describing the desired content, including chapter title, learning objectives, and any specific technical requirements or constraints.
2.  **Planning (via `Agent:ProjectPlanner`):** An internal planning subagent is invoked to create a detailed outline or implementation plan for the content, ensuring it aligns with Docusaurus structure and learning goals.
3.  **Content Generation (via `Agent:BookWriter` and specialized technical skills):**
    *   Content is generated section by section, leveraging specialized `Technical:*` skills (e.g., `Technical:ROS2Content`, `Technical:HumanoidContent`) for accurate and detailed explanations.
    *   `Book:CodeExampleGenerator` is used to create executable code snippets.
    *   `Book:DiagramGenerator` is used for visual explanations.
4.  **Review (via `Agent:ContentReviewer`):** The generated content undergoes a thorough review for technical accuracy, clarity, completeness, grammar, and adherence to the project's style guide.
5.  **Editing (via `Book:Editor`):** Based on review feedback or direct user input, the content is iteratively refined and edited until it meets the required standards.
6.  **Docusaurus Integration:** Ensures the output is correctly formatted for Docusaurus, including proper Markdown, file paths, and integration into the sidebar/navigation if specified.

### Output Format

The output will primarily consist of:
- Markdown (`.md` or `.mdx`) files for the generated chapter or section content, ready to be placed in your Docusaurus project structure.
- Potentially, separate files for code examples or diagram definitions (e.g., `.py`, `.cpp`, `.mermaid`).
- A review report if a review step was part of the process, detailing suggested improvements.

### Example Input/Output

**Example Input:**

```
Generate the content for Chapter 3: "ROS 2 Fundamentals".
Learning Objectives:
- Understand ROS 2 architecture and core concepts.
- Learn to create and manage ROS 2 nodes, topics, and services.
- Explore ROS 2 launch files and debugging tools.
Target Audience: Intermediate robotics students with some programming experience.
```

**Example Output:**

```
<command-message>Running AISpecDrivenBookCreation skill...</command-message>

<commentary>
The skill would then proceed to:
1. Generate an outline for "ROS 2 Fundamentals".
2. Write content for each section (e.g., "ROS 2 Architecture", "Nodes and Topics", "Services and Actions", "Launch Files", "Debugging").
3. Include relevant ROS 2 code examples and diagrams.
4. Review and edit the content.
</commentary>

# Chapter 3: ROS 2 Fundamentals

## 3.1 ROS 2 Architecture and Core Concepts

Robot Operating System 2 (ROS 2) is a flexible framework for writing robot software. It is a collection of tools, libraries, and conventions that aim to simplify the task of creating complex and robust robot behaviors across a wide variety of robotic platforms.

### Key Concepts

*   **Nodes:** Executable processes that perform computation.
*   **Topics:** Named buses for nodes to exchange messages.
*   **Services:** Request/reply mechanism for nodes to communicate.
*   **Actions:** Long-running, preemptable tasks.
*   **Parameters:** Dynamic configuration values for nodes.

```python
# Example: a simple ROS 2 publisher node (src/my_package/publisher_node.py)
import rclpy
from rclpy.node import Node
from std_msgs.msg import String

class MinimalPublisher(Node):

    def __init__(self):
        super().__init__('minimal_publisher')
        self.publisher_ = self.create_publisher(String, 'topic', 10)
        timer_period = 0.5  # seconds
        self.timer = self.create_timer(timer_period, self.timer_callback)
        self.i = 0

    def timer_callback(self):
        msg = String()
        msg.data = f'Hello ROS 2: {self.i}'
        self.publisher_.publish(msg)
        self.get_logger().info(f'Publishing: "{msg.data}"')
        self.i += 1

def main(args=None):
    rclpy.init(args=args)
    minimal_publisher = MinimalPublisher()
    rclpy.spin(minimal_publisher)
    minimal_publisher.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

... (rest of the chapter content)
