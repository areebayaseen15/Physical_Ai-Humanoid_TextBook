---
name: ContentPersonalization
description: Enables dynamic adaptation of textbook content based on individual user profiles, learning styles, and progress.
version: 1.0.0
---

### When to Use This Skill

Use this skill when:
- You need to implement the core logic for personalizing textbook chapters or sections for a user.
- You are defining how user profiles (learning style, background, pace) influence content delivery.
- You need to integrate personalization features into the Docusaurus frontend, such as a "Personalize Chapter" button.
- You are updating or refining the algorithms or rules for content adaptation.

### How This Skill Works

This skill orchestrates the process of fetching user data, adapting content, and integrating personalization into the user interface:

1.  **User Profile Management (via `Personalization:ProfileManager`):** Retrieves and updates user-specific data from NeonDB, including preferences, learning history, and proficiency levels.
2.  **Content Adaptation Logic (via `Personalization:ContentAdapter`):** Takes original textbook content and, based on the active user's profile, modifies it. This could involve:
    *   Adjusting the level of detail (e.g., simplifying explanations for beginners, providing deeper dives for advanced users).
    *   Swapping examples or analogies to be more relevant to the user's declared interests.
    *   Highlighting key concepts or pre-requisites based on learning gaps.
3.  **UI Integration (via `Personalization:UIIntegration`):** Generates or integrates Docusaurus React components for a per-chapter "Personalization" button. When clicked, this button triggers the content adaptation process for the current chapter.
4.  **Backend Integration (via `RAG:FastAPIBuilder`):** Develops or updates FastAPI endpoints that serve personalized content by interacting with the profile manager and content adapter.
5.  **Testing and Evaluation:** Ensures the personalization logic correctly adapts content as expected and the UI integration is seamless.

### Output Format

The output will primarily consist of:
- Python code for content adaptation logic and FastAPI endpoints.
- Database schema updates for user profiles in NeonDB (if not already handled by `UserAuthIntegration`).
- React/Docusaurus component code for the personalization button and dynamic content display.
- Example of personalized content based on a sample user profile.

### Example Input/Output

**Example Input:**

```
Implement per-chapter content personalization.
Users should have a button to personalize the current chapter.
Personalization should adjust detail level based on a 'skill_level' in the user profile (beginner, intermediate, advanced).
```

**Example Output:**

```
<command-message>Running ContentPersonalization skill...</command-message>

<commentary>
The skill would then proceed to:
1. Define `skill_level` field in user profile schema.
2. Create FastAPI endpoint for `/personalize/{chapter_id}`.
3. Implement content adaptation logic based on `skill_level`.
4. Generate a Docusaurus React component for a "Personalize Chapter" button.
</commentary>

# Content Personalization Setup Report

## Backend Logic
- `app/services/personalization.py` created with `adapt_content_for_user` function.
- FastAPI endpoint `/api/personalize/{chapter_id}` added to handle requests.
- Integration with `NeonDBManager` for fetching user profiles.

## UI Integration
- `src/components/PersonalizationButton.js` generated for Docusaurus.
- Example usage provided to integrate the button into chapter pages.

## Example Personalization (for a 'beginner' user):

**Original Content Snippet:**

```markdown
ROS 2 utilizes a DDS (Data Distribution Service) layer for inter-process communication, offering robust quality of service (QoS) policies like reliability, durability, and liveliness.
```

**Personalized Content Snippet (for beginner):**

```markdown
ROS 2 uses a special communication system called DDS. Think of DDS as a very smart mail delivery service for your robot's software parts. It makes sure messages are sent reliably and even keeps them safe if a part isn't ready yet. This is important for making sure your robot's brains talk to each other correctly.
```

## Next Steps:
- Expand personalization parameters (e.g., preferred examples, learning pace).
- Integrate A/B testing for different personalization strategies.
- Develop UI for user profile settings.
```