---
name: UserAuthIntegration
description: Implements robust user signup and signin functionalities with enhanced authentication integration for personalized textbook access.
version: 1.0.0
---

### When to Use This Skill

Use this skill when:
- You need to set up or modify user authentication for the textbook platform.
- You are integrating new authentication providers or enhancing existing security measures.
- You need to manage user sessions, profile creation, and access control for personalized features.
- You are developing frontend components for user registration, login, and profile management.

### How This Skill Works

This skill orchestrates the development and integration of user authentication features:

1.  **Backend API Development (via `RAG:FastAPIBuilder`):** Generates and configures FastAPI endpoints for user registration, login, logout, and session management. This includes handling password hashing, JWT token generation, and secure API communication.
2.  **Database Management (via `RAG:NeonDBManager`):** Manages user data storage in NeonDB, including user credentials (hashed), profile information, and session tokens. It ensures secure storage and retrieval of sensitive user data.
3.  **Authentication Logic:** Implements core authentication logic, including user validation, session creation, and authorization checks for protected routes and personalized content.
4.  **Frontend UI Integration:** Generates or integrates Docusaurus React components for user-facing elements like signup forms, login pages, and user profile dashboards.
5.  **Security Best Practices:** Ensures adherence to security best practices such as input validation, secure password storage, protection against common web vulnerabilities (e.g., XSS, CSRF), and rate limiting.
6.  **Testing and Verification:** Develops and runs tests to ensure the authentication system is secure, functional, and provides a smooth user experience.

### Output Format

The output will include:
- Python code for FastAPI authentication routes and logic.
- Database schema and migration scripts for user tables in NeonDB.
- React/Docusaurus components for user signup, signin, and profile management.
- Confirmation of successful authentication system setup.

### Example Input/Output

**Example Input:**

```
Implement user signup and signin functionality.
Store user profiles in NeonDB.
Generate basic login and registration forms for Docusaurus.
```

**Example Output:**

```
<command-message>Running UserAuthIntegration skill...</command-message>

<commentary>
The skill would then proceed to:
1. Generate FastAPI endpoints for /register and /login.
2. Create 'users' table in NeonDB.
3. Implement password hashing and JWT token generation.
4. Generate Docusaurus React components for login and signup forms.
</commentary>

# User Authentication Setup Report

## FastAPI Endpoints
- `app/api/auth.py` created with `register` and `login` routes.
- JWT token generation and validation implemented.

## NeonDB Schema
- `users` table created with `id`, `username`, `email`, `password_hash` fields.
- Basic user profile fields added.

## UI Integration
- `src/components/Auth/SignUpForm.js` and `SignInForm.js` generated.
- Integration points for Docusaurus pages provided.

## Next Steps:
- Implement password reset functionality.
- Integrate third-party authentication (e.g., Google, GitHub) if required.
- Refine UI/UX for user authentication flow.
```