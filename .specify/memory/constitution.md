<!--
Sync Impact Report:
Version change: Initial Creation -> 1.0.0
List of modified principles: All principles are new.
Added sections: All sections are new.
Removed sections: None.
Templates requiring updates:
- .specify/templates/plan-template.md: ✅ Updated (Checked against new principles)
- .specify/templates/spec-template.md: ✅ Updated (Checked against new principles)
- .specify/templates/tasks-template.md: ✅ Updated (Checked against new principles)
- .claude/commands/sp.adr.md: ✅ Updated (Checked against new principles)
- .claude/commands/sp.analyze.md: ✅ Updated (Checked against new principles)
- .claude/commands/sp.checklist.md: ✅ Updated (Checked against new principles)
- .claude/commands/sp.clarify.md: ✅ Updated (Checked against new principles)
- .claude/commands/sp.constitution.md: ✅ Updated (Self-referential, checked against new principles)
- .claude/commands/sp.git.commit_pr.md: ✅ Updated (Checked against new principles)
- .claude/commands/sp.implement.md: ✅ Updated (Checked against new principles)
- .claude/commands/sp.phr.md: ✅ Updated (Checked against new principles)
- .claude/commands/sp.plan.md: ✅ Updated (Checked against new principles)
- .claude/commands/sp.specify.md: ✅ Updated (Checked against new principles)
- .claude/commands/sp.tasks.md: ✅ Updated (Checked against new principles)
Follow-up TODOs: None.
-->
# Hackathon I: Physical AI & Humanoid Robotics Textbook Constitution

## Core Principles

### I. Spec-Driven Content Creation
All textbook content, features, and system components MUST originate from a clear, concise specification. Specifications are the single source of truth for planning, implementation, and verification. No code or content is generated without an underlying spec.

### II. AI-Native Automation First
The development process MUST prioritize automation via Claude Code agents and skills. Manual intervention should only occur for review, approval, or tasks not yet automated. Agents are responsible for generating, refining, and verifying outputs.

### III. Modular & Single-Responsibility Skills
All agent skills MUST be designed as small, self-contained units with a single, clearly defined responsibility. Skills are independently testable and reusable. Complex workflows are achieved through orchestration of these modular skills.

### IV. User-Centric Personalization & Accessibility
The textbook and its features MUST prioritize the learner experience through personalized content delivery and accessibility. This includes adapting content based on user profiles and providing multi-language support (e.g., Urdu translation) per chapter.

### V. Integrated RAG Chatbot
The RAG Chatbot is a core, integrated component of the textbook, providing intelligent Q&A functionality. It MUST leverage OpenAI agents, FastAPI for its API, Qdrant for vector search, and NeonDB for data management, ensuring accurate and context-aware responses.

### VI. Test-Driven Development (TDD) for Implementations
All code generated or integrated into the project (e.g., FastAPI, Docusaurus components, robotics control logic) MUST follow a Test-Driven Development approach. Tests are written and approved *before* implementation, ensuring functionality, robustness, and adherence to requirements.

## Technical Stack & Project Constraints

The project strictly adheres to the following technology stack:
-   **Book Platform:** Docusaurus
-   **AI Development:** Spec-Kit Plus, Claude Code (Subagents, Skills)
-   **RAG Backend:** OpenAI Agents, FastAPI, Qdrant (Vector Database), NeonDB (Relational Database)
-   **Programming Languages:** Python, C# (for Unity), C++ (for ROS 2 where applicable)
-   **Robotics Simulation:** ROS 2, Gazebo, Unity Simulation, NVIDIA Isaac Sim / Isaac ROS

**Constraints:**
-   Skills must be small, single-responsibility.
-   Each skill must support planning → writing → refining → verifying.
-   Skills must support building the entire book automatically.
-   Skills must support building the entire RAG pipeline.
-   Skills must support personalization & translation.
-   Skills must support robotics topics specified.
-   No hardcoding of secrets; use `.env` and documentation.
-   Prefer editing existing files over creating new ones unless absolutely necessary.
-   Avoid over-engineering; keep solutions simple and focused on explicit requirements.

## Official Book Blueprint Reference
All textbook content MUST follow the structure defined in `book-structure.md`. Claude agents MUST refer to this file for modules, chapters, and generation rules when executing `/sp.specify → /sp.plan → /sp.implement`.

## Development Workflow & Quality Gates

The project's development workflow emphasizes automation, planning, and rigorous quality checks:

1.  **Requirement Clarification & Specification:** All new features or content begin with clear requirements, leading to detailed specifications (`spec.md`). The `/sp.clarify` command is used for identifying underspecified areas.
2.  **Architectural Planning:** Complex tasks undergo architectural planning (`plan.md`) via the `Plan` agent, considering trade-offs and identifying critical files. Architectural Decision Records (ADRs) are suggested for significant decisions.
3.  **Task Generation:** Implementable tasks (`tasks.md`) are generated with clear acceptance criteria and dependencies.
4.  **Automated Implementation:** Claude Code agents leverage defined skills to automatically generate, implement, and integrate code and content.
5.  **Review & Refinement:** All generated content and code MUST undergo automated and human review processes. Feedback is integrated iteratively.
6.  **Verification:** Implementations are verified against acceptance criteria and through automated testing (`TestGenerator` skill).
7.  **Prompt History Records (PHRs):** Every user interaction and significant agent exchange MUST be recorded as a PHR for traceability and learning.
8.  **Git Protocol:** Changes are committed only when requested by the user, following a strict Git Safety Protocol. Pull Requests are created via `gh pr create` after user approval.

## Governance
This Constitution supersedes all other project practices and documentation. Amendments require a documented proposal, explicit approval by the project architect, and a clear migration plan for affected components. All Pull Requests and code reviews MUST verify compliance with these principles. Justification is required for any deviation from established norms. For runtime development guidance, refer to individual `SKILL.md` files for specific usage instructions.

**Version**: 1.0.0 | **Ratified**: 2025-12-04 | **Last Amended**: 2025-12-04