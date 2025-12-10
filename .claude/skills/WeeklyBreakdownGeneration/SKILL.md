---
name: WeeklyBreakdownGeneration
description: Organizes the entire textbook content into a structured weekly learning schedule, outlining topics and assignments for each week.
version: 1.0.0
---

### When to Use This Skill

Use this skill when:
- You need to create a weekly syllabus or learning schedule for the "Physical AI & Humanoid Robotics" course based on the textbook content.
- You want to distribute chapters, topics, and practical exercises across a defined number of weeks.
- You are planning the structure of a course or workshop that utilizes the textbook.
- You need to update or modify the existing weekly breakdown.

### How This Skill Works

This skill focuses on generating a structured weekly learning plan from the textbook content:

1.  **Content Inventory:** Analyzes the available textbook chapters and sections, identifying key topics and their estimated learning complexity/duration.
2.  **Schedule Generation:** Takes the total number of weeks available for the course and intelligently distributes the textbook content across these weeks. It aims for a logical flow and balanced workload per week.
3.  **Topic Allocation:** Assigns specific chapters, sections, and sub-topics to each week.
4.  **Activity Suggestions (Optional):** Can suggest complementary activities or assignments for each week, such as:
    *   Reading assignments.
    *   Practical exercises (e.g., ROS 2 coding tasks, Gazebo simulations).
    *   Quiz/assessment pointers.
    *   Discussion topics.
5.  **Output Formatting:** Formats the weekly breakdown into a clear, readable markdown table or list, suitable for a course syllabus or Docusaurus integration.
6.  **Customization:** Allows for parameters like emphasis on certain topics or a lighter/heavier workload in specific weeks.

### Output Format

The output will typically include:
- Markdown formatted weekly schedule, listing week numbers, topics covered, and suggested activities.
- A summary of the course structure.
- (Optional) Generated placeholders for assignments or quizzes.

### Example Input/Output

**Example Input:**

```
Generate a 12-week breakdown for the "Physical AI & Humanoid Robotics" textbook.
Prioritize early introduction to ROS 2 and later focus on Humanoid topics.
```

**Example Output:**

```
<command-message>Running WeeklyBreakdownGeneration skill...</command-message>

<commentary>
The skill would then proceed to:
1. Analyze all available chapters.
2. Create a 12-week schedule, balancing content.
3. Prioritize ROS 2 in early weeks and Humanoids later.
4. Format as a markdown table.
</commentary>

# 12-Week Course Breakdown: Physical AI & Humanoid Robotics

This is a suggested 12-week breakdown for the "Physical AI & Humanoid Robotics" course, designed to progressively build knowledge from foundational concepts to advanced topics, with an emphasis on ROS 2 early on and humanoid robotics later.

## Course Overview
This course covers fundamental concepts in physical AI and humanoid robotics, including simulation, control, perception, and advanced AI models. Practical exercises will utilize ROS 2, Gazebo, Unity, and NVIDIA Isaac platforms.

## Weekly Schedule

| Week | Main Topics                                   | Textbook Chapters/Sections           | Suggested Activities/Assignments                                |
| :--- | :-------------------------------------------- | :----------------------------------- | :-------------------------------------------------------------- |
| 1    | Introduction to Physical AI & Robotics, Course Setup | Chapter 1 (Intro), Course Logistics  | Environment setup, ROS 2 installation, Git basics               |
| 2    | ROS 2 Fundamentals: Architecture & Communication | Chapter 2 (ROS 2 Basics)             | Create first ROS 2 publisher/subscriber nodes                   |
| 3    | ROS 2 Tools: Launch Files, Rviz, rqt             | Chapter 3 (ROS 2 Tools)              | ROS 2 launch file exercise, data visualization with Rviz        |
| 4    | Gazebo Simulation: Worlds & Models             | Chapter 4 (Gazebo Intro), URDF/SDF   | Build a simple Gazebo world, create a basic robot URDF        |
| 5    | ROS 2 & Gazebo Integration                     | Chapter 5 (ROS-Gazebo Interface)     | Control simulated robot in Gazebo via ROS 2 topics            |
| 6    | Unity Simulation & ML-Agents                   | Chapter 6 (Unity Robotics)           | Implement a simple ML-Agent for robot control in Unity          |
| 7    | NVIDIA Isaac Sim & Isaac ROS: Basics           | Chapter 7 (Isaac Sim Intro)          | Explore Isaac Sim environment, run first Isaac ROS example      |
| 8    | NVIDIA Isaac: AI Perception & Manipulation     | Chapter 8 (Isaac Perception/Manip)   | Object detection or simple pick-and-place in Isaac Sim        |
| 9    | Humanoid Locomotion: Kinematics & Planning     | Chapter 9 (Humanoid Kinematics)      | Forward/Inverse Kinematics exercise for a humanoid arm/leg      |
| 10   | Humanoid Balance & Control                     | Chapter 10 (Humanoid Control)        | ZMP concept application, basic balance control simulation       |
| 11   | Conversational Robotics & VLA Models           | Chapter 11 (Conversational AI), VLA  | Design a simple voice command interface for a robot             |
| 12   | Advanced Topics, Project Work & Review         | Chapter 12 (Advanced/Future), Project | Final project presentation, course review, feedback submission |

## Next Steps:
- Integrate this breakdown into a Docusaurus page.
- Create specific assignment descriptions for each week.
- Link directly to relevant textbook sections within the breakdown.
```