# Agents Visual Studio Code Extension - Quick Start Guide

## 5-Minute Setup

### 1. Install the Extension

```
Ctrl+Shift+X (Windows/Linux) or Cmd+Shift+X (Mac)
Search: "Agents Visual Studio Code"
Click Install
```

### 2. Get Your API Key

**OpenAI:**
- Visit https://platform.openai.com/api-keys
- Create a new API key
- Copy it

**Anthropic:**
- Visit https://console.anthropic.com/
- Create a new API key
- Copy it

### 3. Configure Extension

```
Ctrl+, (Windows/Linux) or Cmd+, (Mac)
Search: "Agents LLM API Key"
Paste your API key
```

### 4. Test It Out

```
Ctrl+Shift+P (Windows/Linux) or Cmd+Shift+P (Mac)
Type: "Agents: Scaffold Feature"
Describe a feature: "Create login form with email validation"
Press Enter
```

---

## Your First Feature

### Step-by-Step Example: User Authentication

#### Step 1: Create a New Project

```bash
mkdir my-project
cd my-project
code .
```

#### Step 2: Initialize Node/TypeScript

```bash
npm init -y
npm install --save-dev typescript @types/node
npx tsc --init
```

#### Step 3: Open Agents Palette

```
Ctrl+Shift+P
Type "Agents: Scaffold Feature"
```

#### Step 4: Describe Your Feature

```
Feature Description:
"Create a user authentication module with the following:
- JWT token generation
- Password hashing with bcrypt
- Login endpoint with validation
- Refresh token mechanism
- Error handling and logging"
```

#### Step 5: Review Generated Code

The agent will generate:
- `src/auth/jwt.ts` - JWT utilities
- `src/auth/password.ts` - Password hashing
- `src/auth/service.ts` - Authentication service
- `src/auth/types.ts` - TypeScript types
- `test/auth.test.ts` - Automated tests

#### Step 6: Run Tests

```bash
npm test
```

#### Step 7: Review and Commit

```bash
git add .
git commit -m "feat: Add user authentication module"
```

---

## Common Commands

### Feature Scaffolding

```
Ctrl+Shift+P → "Agents: Scaffold Feature"
```

Describe what you want:
- "Create REST API endpoint for user profiles"
- "Add database migrations for user table"
- "Build WebSocket connection handler"

### Generate Tests

```
Ctrl+Shift+P → "Agents: Generate Tests"
```

Select code file and let the agent create comprehensive tests.

### Generate Documentation

```
Ctrl+Shift+P → "Agents: Generate Documentation"
```

Auto-generate API docs, README, and architecture docs.

### Analyze Code

```
Ctrl+Shift+P → "Agents: Analyze Code"
```

Get insights on code quality, complexity, and suggestions.

### Refactor Code

```
Ctrl+Shift+P → "Agents: Refactor Code"
```

Ask the agent to improve existing code:
- "Make this function more efficient"
- "Improve error handling"
- "Add TypeScript types"

### Connect MCP Server

```
Ctrl+Shift+P → "Agents: Connect MCP Server"
```

Add external tools and capabilities.

### Custom Instructions

```
Ctrl+Shift+P → "Agents: Add Custom Instruction"
```

Define team conventions and standards.

---

## Real-World Scenarios

### Scenario 1: Build a REST API

```
1. Scaffold Feature: "Create Express REST API with routes:
   - GET /users
   - POST /users
   - PUT /users/:id
   - DELETE /users/:id"

2. Generate Tests: Select the generated files

3. Add Custom Instruction:
   "Always use error handling middleware"
   "Always validate input with zod"
   "Add rate limiting"

4. Refactor: "Add logging and monitoring"

5. Generate Docs: API documentation with examples
```

### Scenario 2: Create a React Component

```
1. Scaffold Feature: "Create React component for user profile:
   - Display user information
   - Edit profile form
   - Upload profile picture
   - Show activity feed"

2. Generate Tests: Unit and integration tests

3. Custom Instruction: "Use React hooks", "Use TypeScript"

4. Generate Docs: Component API documentation
```

### Scenario 3: Database Schema

```
1. Scaffold Feature: "Create database schema for e-commerce:
   - Users table
   - Products table
   - Orders table
   - Order items
   - Relationships and indexes"

2. Generate Tests: Schema validation tests

3. Add Migration: Generate migration files

4. Generate Docs: Schema documentation
```

---

## Tips & Tricks

### 💡 Tip 1: Use Context

Provide existing code context for better generation:

```
"Based on our existing User model in src/models/User.ts,
create an admin user management system with:
- Batch user operations
- Bulk imports
- Permission management"
```

### 💡 Tip 2: Iterate Quickly

Refine results by requesting changes:

```
Initial: "Create login form"
Refine 1: "Add password strength indicator"
Refine 2: "Add remember me checkbox"
Refine 3: "Make it mobile responsive"
```

### 💡 Tip 3: Define Standards

Use custom instructions for consistency:

```json
{
  "id": "code-standards",
  "instruction": "Always:
    - Use TypeScript with strict mode
    - Write JSDoc comments
    - Include error handling
    - Add unit tests
    - Use meaningful variable names"
}
```

### 💡 Tip 4: Review Generated Code

Always review AI-generated code:
- ✅ Check logic correctness
- ✅ Review security implications
- ✅ Verify performance
- ✅ Test edge cases
- ✅ Run security scans

### 💡 Tip 5: Use MCP Servers

Extend capabilities with MCP servers:

```json
{
  "agents.mcpServers": [
    {
      "id": "code-analyzer",
      "name": "Code Quality Analyzer",
      "enabled": true
    },
    {
      "id": "git-assistant",
      "name": "Git Operations Helper",
      "enabled": true
    }
  ]
}
```

---

## Troubleshooting

### Problem: "API Key Not Found"

**Solution:**
```
Settings → Search "Agents LLM API Key"
Paste your API key
Reload VS Code
```

### Problem: "Generation Too Slow"

**Solution:**
- Use faster model: `gpt-3.5-turbo` instead of `gpt-4`
- Reduce `maxTokens` in settings
- Close other applications

### Problem: "Generated Code Has Errors"

**Solution:**
1. Review the generated code
2. Provide more specific instructions
3. Add custom instructions for your patterns
4. Use custom instructions for standards

### Problem: "Tests Keep Failing"

**Solution:**
1. Check generated test setup
2. Verify dependencies are installed
3. Review test structure
4. Refine feature description

---

## Next Steps

1. ✅ Try the [5-Minute Setup](#5-minute-setup)
2. ✅ Follow [Your First Feature](#your-first-feature)
3. ✅ Explore [Real-World Scenarios](#real-world-scenarios)
4. ✅ Read [API Documentation](./API_DOCUMENTATION.md)
5. ✅ Review [Architecture Guide](./ARCHITECTURE_DOCUMENTATION.md)
6. ✅ Check [Release Notes](./AGENTS_VS_CODE_RELEASE_NOTES.md)

---

## Resources

- 📖 **Full Documentation**: https://github.com/AnticipatedD/AnticipatedD
- 💬 **Discussions**: https://github.com/AnticipatedD/AnticipatedD/discussions
- 🐛 **Report Issues**: https://github.com/AnticipatedD/AnticipatedD/issues
- 📧 **Email Support**: harigov63@gmail.com
- 🌐 **Website**: https://vane-enterprise.github.io

---

## Video Tutorials

- [Installation & Setup](https://youtube.com/...) - 3 min
- [First Feature Creation](https://youtube.com/...) - 10 min
- [Advanced Workflows](https://youtube.com/...) - 15 min
- [MCP Server Integration](https://youtube.com/...) - 12 min

---

## Sample Projects

- [Basic Todo App](https://github.com/...)
- [REST API Starter](https://github.com/...)
- [React Dashboard](https://github.com/...)
- [Full-Stack E-Commerce](https://github.com/...)

---

*Last Updated: June 6, 2026*