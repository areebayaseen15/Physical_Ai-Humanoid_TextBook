---
name: RAGChatbot
description: Builds and integrates a Retrieval-Augmented Generation (RAG) chatbot for the textbook, enabling Q&A functionality over the book's content.
version: 1.0.0
---

### When to Use This Skill

Use this skill when:
- You need to set up the RAG chatbot for the "Physical AI & Humanoid Robotics" textbook.
- You want to index new book content or update the existing knowledge base for the chatbot.
- You need to extend or modify the chatbot's API endpoints or AI integration (e.g., OpenAI agents).
- You are integrating the chatbot's UI components into the Docusaurus frontend.

### How This Skill Works

This skill orchestrates the entire process of building and integrating the RAG chatbot:

1.  **Data Ingestion (via `RAG:DataIngestor`):** It processes textbook chapters and other relevant data sources (e.g., PDFs, web pages) into a structured format suitable for vectorization.
2.  **Vector Indexing (via `RAG:QdrantIndexer`):** The processed data is indexed into a Qdrant vector database, creating vector embeddings for efficient retrieval.
3.  **Database Management (via `RAG:NeonDBManager`):** Manages relational data such as chat history, user metadata, or specific configurations within NeonDB.
4.  **API Development (via `RAG:FastAPIBuilder`):** Generates and maintains FastAPI endpoints to serve the RAG chatbot's functionalities, including query handling, personalization hooks, and translation triggers.
5.  **AI Integration (via `RAG:OpenAIAgentIntegrator`):** Integrates OpenAI agents (e.g., Assistants API, function calling) to enhance conversational capabilities, answer complex questions, and manage dialogue flow.
6.  **UI Integration (via `RAG:ChatbotUIBuilder`):** Generates and integrates Docusaurus React components for the chatbot interface within the textbook, ensuring a seamless user experience.
7.  **Testing and Verification:** Ensures all components of the RAG system are working correctly and the chatbot can accurately answer questions based on the textbook content.

### Output Format

The output will include:
- Python code for FastAPI application and OpenAI agent integration.
- Configuration files for Qdrant and NeonDB setup.
- React/Docusaurus components for the chatbot UI.
- Confirmation of successful data ingestion and indexing.

### Example Input/Output

**Example Input:**

```
Set up the RAG chatbot system.
Index all existing textbook chapters for Q&A.
Integrate an OpenAI agent for advanced query handling.
```

**Example Output:**

```
<command-message>Running RAGChatbot skill...</command-message>

<commentary>
The skill would then proceed to:
1. Ingest existing textbook chapters.
2. Index data into Qdrant.
3. Set up necessary NeonDB tables.
4. Generate FastAPI boilerplate.
5. Integrate OpenAI agent configuration.
6. Generate basic Docusaurus chatbot UI components.
</commentary>

# RAG Chatbot Setup Report

## Data Ingestion
- All existing chapters in 'content/docs' ingested.
- Processed 15 markdown files.

## Qdrant Indexing
- 'textbook_content' collection created in Qdrant.
- 1500 vectors indexed.

## FastAPI Endpoints
- `main.py` created with `/chat` and `/ingest` endpoints.
- Placeholder for OpenAI agent integration in `app/services/chatbot.py`.

## UI Integration
- `src/components/Chatbot/index.js` generated.
- Basic chat input and display integrated into Docusaurus.

## Next Steps:
- Review and refine generated FastAPI code for specific logic.
- Implement detailed OpenAI agent functions.
- Customize Chatbot UI.
```