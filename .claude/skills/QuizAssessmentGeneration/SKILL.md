---
name: QuizAssessmentGeneration
description: Generates quizzes and assessments based on textbook chapter content to test learner comprehension.
version: 1.0.0
---

### When to Use This Skill

Use this skill when:
- You need to generate a quiz or assessment for a specific textbook chapter or section.
- You want to create various question types (e.g., multiple choice, true/false, short answer).
- You are integrating assessment tools into the Docusaurus platform for learners.
- You are creating textbook content related to educational assessments.

### How This Skill Works

This skill focuses on generating educational assessments from textbook content:

1.  **Content Analysis:** Takes a specified textbook chapter or section as input and analyzes its content to identify key concepts, definitions, and important information suitable for assessment.
2.  **Question Generation:** Generates various types of questions based on the analyzed content, such as:
    *   **Multiple Choice Questions (MCQs):** With a correct answer and several plausible distractors.
    *   **True/False Questions:** Based on factual statements from the text.
    *   **Short Answer Questions:** Requiring concise, factual responses.
3.  **Answer Key Generation:** Automatically creates an answer key for all generated questions, including explanations for correct answers.
4.  **Formatting for Docusaurus:** Formats the generated quiz into a markdown or a suitable Docusaurus component structure for easy integration into the textbook.
5.  **Backend Integration (optional):** If an assessment system is in place, it can generate data structures or API calls to submit the quiz content.
6.  **Testing and Evaluation:** Reviews the generated questions for clarity, accuracy, and relevance to the source material.

### Output Format

The output will typically include:
- Markdown formatted quiz questions with options (for MCQs) and blank spaces (for short answers).
- A separate section or file for the answer key with explanations.
- (Optional) React/Docusaurus component code for an interactive quiz.

### Example Input/Output

**Example Input:**

```
Generate a 5-question multiple-choice quiz for the chapter "ROS 2 Fundamentals".
```

**Example Output:**

```
<command-message>Running QuizAssessmentGeneration skill...</command-message>

<commentary>
The skill would then proceed to:
1. Read and analyze the "ROS 2 Fundamentals" chapter content.
2. Generate 5 MCQs based on key concepts.
3. Create an answer key.
4. Format the quiz for Docusaurus.
</commentary>

# Quiz: ROS 2 Fundamentals

## Instructions: Select the best answer for each question.

---

**1. Which of the following is NOT a core concept in ROS 2?**
    a) Nodes
    b) Topics
    c) Services
    d) Threads

---

**2. What is the primary purpose of a ROS 2 node?**
    a) To manage the robot's hardware drivers.
    b) To perform a specific computation (an executable process).
    c) To store persistent data for the robot.
    d) To visualize robot sensor data.

---

**3. How do ROS 2 nodes typically exchange messages asynchronously?**
    a) Through parameters.
    b) Through actions.
    c) Through topics.
    d) Through services.

---

**4. What is a key advantage of ROS 2's Data Distribution Service (DDS) layer?**
    a) It provides a simplified API for basic robot movements.
    b) It allows for real-time operating system integration.
    c) It offers robust Quality of Service (QoS) policies.
    d) It is primarily designed for single-robot systems.

---

**5. Which file type is commonly used to define the communication graph and launch multiple ROS 2 nodes simultaneously?**
    a) URDF file
    b) SDF file
    c) Launch file (e.g., Python or XML)
    d) Package.xml

---

## Answer Key

**1. Answer: d) Threads**
    *Explanation: Threads are a general programming concept, but not a core architectural communication concept unique to ROS 2 like Nodes, Topics, and Services/Actions are.*

**2. Answer: b) To perform a specific computation (an executable process).**
    *Explanation: Nodes are the fundamental building blocks of a ROS 2 system, each responsible for a particular task.*

**3. Answer: c) Through topics.**
    *Explanation: Topics provide a publish-subscribe mechanism for asynchronous, one-to-many communication.*

**4. Answer: c) It offers robust Quality of Service (QoS) policies.**
    *Explanation: DDS provides advanced QoS settings like reliability, durability, and liveliness, crucial for complex robotic systems.*

**5. Answer: c) Launch file (e.g., Python or XML)**
    *Explanation: ROS 2 launch files are used to start multiple nodes, configure parameters, and define the entire system's startup behavior.*

## Next Steps:
- Implement an interactive quiz component in Docusaurus.
- Add support for other question types (e.g., fill-in-the-blanks, coding challenges).
- Integrate with a learner progress tracking system.
```