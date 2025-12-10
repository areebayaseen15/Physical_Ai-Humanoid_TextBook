---
name: UrduTranslation
description: Provides per-chapter translation functionality, specifically for converting textbook content into Urdu.
version: 1.0.0
---

### When to Use This Skill

Use this skill when:
- You need to enable a per-chapter Urdu translation feature within the Docusaurus textbook.
- You are integrating the translation button and display logic into the frontend.
- You are defining the backend API for handling translation requests.
- You need to update or refine the translation mechanism or integrate a new translation service.

### How This Skill Works

This skill orchestrates the translation process and its integration into the textbook platform:

1.  **Translation Logic (via `Translation:TextTranslator`):** Takes a block of text (e.g., a chapter section) and translates it into Urdu using an underlying translation model or service.
2.  **UI Integration (via `Translation:UIIntegration`):** Generates or integrates Docusaurus React components for a per-chapter "Urdu Translation" button. When clicked, this button triggers the translation of the current chapter's content.
3.  **Backend Integration (via `RAG:FastAPIBuilder`):** Develops or updates FastAPI endpoints that handle translation requests, calling the `Translation:TextTranslator` and returning the translated content.
4.  **Display Mechanism:** Ensures the translated content can be displayed seamlessly alongside or in place of the original English content within Docusaurus.
5.  **Testing and Evaluation:** Verifies the accuracy and fluency of the Urdu translation and confirms the UI integration functions correctly.

### Output Format

The output will primarily consist of:
- Python code for translation logic and FastAPI endpoints.
- React/Docusaurus component code for the translation button and dynamic translated content display.
- An example of translated content for a sample chapter section.

### Example Input/Output

**Example Input:**

```
Implement a per-chapter Urdu translation button.
The button should translate the visible chapter content into Urdu.
```

**Example Output:**

```
<command-message>Running UrduTranslation skill...</command-message>

<commentary>
The skill would then proceed to:
1. Create FastAPI endpoint for `/translate/{chapter_id}` with `target_language='ur'`.
2. Implement the core text translation logic (assuming an external service or model).
3. Generate a Docusaurus React component for a "Urdu Translate" button.
</commentary>

# Urdu Translation Setup Report

## Backend Logic
- `app/services/translation.py` created with `translate_text_to_urdu` function.
- FastAPI endpoint `/api/translate/{chapter_id}` added to handle Urdu translation requests.

## UI Integration
- `src/components/UrduTranslationButton.js` generated for Docusaurus.
- Example usage provided to integrate the button into chapter pages.

## Example Translation (for a chapter heading):

**Original Content Snippet:**

```markdown
# Chapter 5: Humanoid Locomotion
```

**Translated Content Snippet (Urdu):**

```markdown
# باب 5: ہیومنائیڈ کی نقل و حرکت
```

## Next Steps:
- Integrate with a robust Urdu translation API or local model.
- Handle different translation scopes (e.g., paragraphs, entire sections).
- Refine UI for toggling between original and translated content.
```