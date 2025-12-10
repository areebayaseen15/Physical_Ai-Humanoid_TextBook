---
name: FeedbackEvaluation
description: Provides mechanisms for learners to submit feedback and for the system to evaluate their progress and provide constructive insights.
version: 1.0.0
---

### When to Use This Skill

Use this skill when:
- You need to implement a feedback submission mechanism for learners (e.g., bug reports, content suggestions).
- You are designing a system to evaluate learner performance on quizzes, exercises, or personalized content.
- You want to provide automated, constructive feedback to learners based on their interactions or assessment results.
- You are integrating feedback and evaluation tools into the Docusaurus platform.

### How This Skill Works

This skill focuses on enabling feedback collection and learner evaluation:

1.  **Feedback Form Generation:** Creates Docusaurus React components for a user-friendly feedback form, allowing learners to submit textual feedback, rate content, or report issues.
2.  **Feedback Submission API:** Develops or updates FastAPI endpoints to receive and store learner feedback in NeonDB, potentially categorized by chapter, user, or feedback type.
3.  **Learner Performance Evaluation:** Integrates with quiz results (`QuizAssessmentGeneration` skill) and content interaction data (`Personalization:ProfileManager` for tracking) to assess learner comprehension and progress.
4.  **Automated Feedback Generation:** Uses AI models to generate personalized, constructive feedback for learners based on their assessment scores, common mistakes, or areas of difficulty.
5.  **Progress Tracking:** Stores and visualizes learner progress (e.g., completed chapters, quiz scores, areas for improvement) within the user's profile.
6.  **UI Integration:** Provides mechanisms to display evaluation results and automated feedback to learners within the textbook interface.
7.  **Tutorial and Explanation:** Produces markdown content explaining the importance of feedback, different evaluation metrics, and how learners can utilize the system.

### Output Format

The output will typically include:
- React/Docusaurus component code for feedback forms and evaluation dashboards.
- Python code for FastAPI endpoints to handle feedback submission and evaluation logic.
- Database schema updates for feedback storage and learner progress tracking in NeonDB.
- Examples of automated feedback generated for learners.

### Example Input/Output

**Example Input:**

```
Implement a feedback form at the end of each chapter.
The form should allow users to submit text feedback and a content rating (1-5 stars).
```

**Example Output:**

```
<command-message>Running FeedbackEvaluation skill...</command-message>

<commentary>
The skill would then proceed to:
1. Generate a Docusaurus React component for the feedback form.
2. Create a FastAPI endpoint to receive feedback.
3. Define a table in NeonDB to store feedback.
</commentary>

# Feedback & Evaluation Setup Report

## Feedback Form Integration
- `src/components/ChapterFeedbackForm.js` generated for Docusaurus.
- Includes text area for feedback and a 5-star rating component.
- Example usage provided to embed the form at the end of chapters.

## Backend API for Feedback
- `app/api/feedback.py` created with a `POST /feedback` endpoint.
- Endpoint handles `chapter_id`, `user_id`, `rating`, and `feedback_text`.

## NeonDB Schema for Feedback
- `feedback` table created with `id`, `chapter_id`, `user_id`, `rating`, `feedback_text`, `timestamp` fields.

## Next Steps:
- Implement AI-driven analysis of feedback for common themes or issues.
- Develop a learner dashboard to display progress and personalized insights.
- Integrate with quiz results for comprehensive evaluation.
```