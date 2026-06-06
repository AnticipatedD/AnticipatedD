# Agents Visual Studio Code Extension - Installation & Setup Guide

## Prerequisites

Before installing the Agents VS Code Extension, ensure you have:

### Required
- ✅ Visual Studio Code 1.90.0 or later
- ✅ Node.js 16.x or later
- ✅ npm 7.x or later
- ✅ Python 3.8+ (for AI/ML components)

### Recommended
- Git 2.30+
- Docker (for containerized MCP servers)
- GitHub account (for integration features)

### Optional
- Alternative LLM API keys (OpenAI, Anthropic, etc.)
- MCP server credentials

---

## Installation Methods

### Method 1: Install from VS Code Marketplace (Recommended)

1. **Open VS Code Extension Marketplace**
   - Press `Ctrl+Shift+X` (Windows/Linux) or `Cmd+Shift+X` (Mac)
   - Or click the Extensions icon in the Activity Bar

2. **Search for "Agents"**
   - Type "Agents Visual Studio Code" in the search box
   - Look for the official extension by AnticipatedD/Vane-Guard

3. **Click Install**
   - Click the "Install" button
   - Wait for the installation to complete

4. **Reload VS Code**
   - Click "Reload" when prompted
   - The extension will activate automatically

---

### Method 2: Install from GitHub Releases

1. **Download the Extension**
   ```bash
   # Clone the repository
   git clone https://github.com/AnticipatedD/AnticipatedD.git
   cd AnticipatedD
   
   # Install dependencies
   npm install
   
   # Build the extension
   npm run build
   ```

2. **Package the Extension**
   ```bash
   # Install vsce (Visual Studio Code Extension Manager)
   npm install -g vsce
   
   # Package the extension
   vsce package
   ```

3. **Install the .vsix File**
   - Press `Ctrl+Shift+X` to open Extensions
   - Click the three dots menu (⋯) at the top
   - Select "Install from VSIX"
   - Navigate to and select the `.vsix` file
   - Click "Install"

---

### Method 3: Install from Source (Development)

1. **Clone Repository**
   ```bash
   git clone https://github.com/AnticipatedD/AnticipatedD.git
   cd AnticipatedD
   ```

2. **Install Dependencies**
   ```bash
   npm install
   pip install -r requirements.txt
   ```

3. **Open in VS Code**
   ```bash
   code .
   ```

4. **Launch Development Mode**
   - Press `F5` or go to Run → Start Debugging
   - This opens a new VS Code window with the extension active
   - Changes are automatically reloaded

---

## Initial Configuration

### Step 1: Open Settings

1. Press `Ctrl+,` (Windows/Linux) or `Cmd+,` (Mac)
2. Or go to File → Preferences → Settings
3. Search for "Agents"

### Step 2: Configure LLM Provider

**Option A: OpenAI API**
```json
{
  "agents.llm.provider": "openai",
  "agents.llm.apiKey": "your-api-key-here",
  "agents.llm.model": "gpt-4",
  "agents.llm.temperature": 0.7,
  "agents.llm.maxTokens": 2000
}
```

**Option B: Anthropic Claude**
```json
{
  "agents.llm.provider": "anthropic",
  "agents.llm.apiKey": "your-anthropic-key",
  "agents.llm.model": "claude-3-opus",
  "agents.llm.temperature": 0.7
}
```

**Option C: Local LLM (Ollama)**
```json
{
  "agents.llm.provider": "local",
  "agents.llm.endpoint": "http://localhost:11434",
  "agents.llm.model": "neural-chat",
  "agents.llm.temperature": 0.7
}
```

### Step 3: Set Up MCP Servers (Optional)

Add MCP server configurations to your workspace settings:

**Workspace Settings (.vscode/settings.json)**
```json
{
  "agents.mcpServers": [
    {
      "id": "code-analyzer",
      "name": "Code Analyzer",
      "type": "stdio",
      "connection": {
        "command": "python",
        "args": ["-m", "code_analyzer"]
      },
      "enabled": true
    },
    {
      "id": "git-helper",
      "name": "Git Helper",
      "type": "stdio",
      "connection": {
        "command": "node",
        "args": ["./mcp-servers/git-helper.js"]
      },
      "enabled": true
    }
  ]
}
```

### Step 4: Create Custom Instructions (Optional)

1. Create `.agents/instructions.json` in your workspace:

```json
{
  "instructions": [
    {
      "id": "typescript-strict",
      "name": "TypeScript Strict Mode",
      "instruction": "Always use strict: true in tsconfig.json",
      "scope": "repository",
      "priority": 10,
      "tags": ["typescript", "quality"],
      "appliedTo": [
        {
          "type": "language",
          "value": "typescript"
        }
      ],
      "enabled": true
    },
    {
      "id": "naming-conventions",
      "name": "Naming Conventions",
      "instruction": "Use camelCase for variables and functions, PascalCase for classes",
      "scope": "repository",
      "priority": 8,
      "tags": ["coding-standards"],
      "appliedTo": [
        {
          "type": "language",
          "value": "typescript"
        },
        {
          "type": "language",
          "value": "javascript"
        }
      ],
      "enabled": true
    }
  ]
}
```

### Step 5: Verify Installation

1. **Check Extension Status**
   - Open Command Palette: `Ctrl+Shift+P` (Windows/Linux) or `Cmd+Shift+P` (Mac)
   - Type "Agents: Show Status"
   - Should show "✓ Extension initialized successfully"

2. **Test Agent Functionality**
   - Open any code file
   - Open Command Palette
   - Type "Agents: Scaffold Feature"
   - Follow the prompts

---

## Configuration Reference

### Extension Settings

| Setting | Type | Default | Description |
|---------|------|---------|-------------|
| `agents.enabled` | boolean | `true` | Enable/disable extension |
| `agents.llm.provider` | string | `"openai"` | LLM provider (openai, anthropic, local) |
| `agents.llm.apiKey` | string | `""` | API key for LLM provider |
| `agents.llm.model` | string | `"gpt-4"` | LLM model name |
| `agents.llm.temperature` | number | `0.7` | Model temperature (0-1) |
| `agents.llm.maxTokens` | number | `2000` | Maximum tokens per request |
| `agents.mcpServers` | array | `[]` | MCP server configurations |
| `agents.customInstructions` | array | `[]` | Custom instruction definitions |
| `agents.marketplace.autoUpdate` | boolean | `false` | Auto-update marketplace packages |
| `agents.logging.level` | string | `"info"` | Logging level (debug, info, warn, error) |
| `agents.experimental.enabled` | boolean | `false` | Enable experimental features |

---

## Troubleshooting

### Issue: "LLM API Key Not Found"

**Solution:**
```bash
# Set API key via environment variable
export OPENAI_API_KEY="your-key-here"

# Or in VS Code settings
# Settings → Agents: Llm API Key
```

### Issue: "MCP Server Connection Failed"

**Solution:**
1. Verify the server command is correct
2. Check server logs: View → Output → Select "Agents: MCP Servers"
3. Ensure required dependencies are installed

### Issue: "Extension Not Activating"

**Solution:**
```bash
# Reinstall the extension
# 1. Disable the extension
# 2. Reload VS Code
# 3. Re-enable and reload

# Or from CLI:
code --disable-extension AnticipatedD.agents
code --enable-extension AnticipatedD.agents
```

### Issue: "Performance Degradation"

**Solution:**
- Reduce `agents.llm.maxTokens`
- Disable unused MCP servers
- Enable logging to identify bottlenecks:
  ```json
  {
    "agents.logging.level": "debug",
    "agents.logging.file": "./agents-debug.log"
  }
  ```

---

## Environment Variables

You can set these environment variables instead of using settings:

```bash
# LLM Configuration
export AGENTS_LLM_PROVIDER="openai"
export AGENTS_LLM_API_KEY="your-key"
export AGENTS_LLM_MODEL="gpt-4"
export AGENTS_LLM_TEMPERATURE="0.7"
export AGENTS_LLM_MAX_TOKENS="2000"

# Workspace Path
export AGENTS_WORKSPACE_PATH="/path/to/workspace"

# Logging
export AGENTS_LOG_LEVEL="info"
```

---

## Security Best Practices

1. **Never commit API keys** to version control
   - Use `.env` files (add to `.gitignore`)
   - Use VS Code secrets storage
   - Use environment variables

2. **Use workspace settings carefully**
   - Store sensitive data in user settings, not workspace settings
   - Workspace settings are often shared in git

3. **Validate MCP server sources**
   - Only use MCP servers from trusted sources
   - Review MCP server code before using

4. **Review generated code**
   - Always review AI-generated code
   - Run security scans
   - Test thoroughly before production

---

## Performance Optimization

### For Large Projects

```json
{
  "agents.caching.enabled": true,
  "agents.caching.ttl": 3600,
  "agents.indexing.enabled": true,
  "agents.indexing.excludePatterns": [
    "**/node_modules/**",
    "**/.git/**",
    "**/dist/**"
  ]
}
```

### For Resource-Constrained Systems

```json
{
  "agents.llm.maxTokens": 1000,
  "agents.llm.temperature": 0.5,
  "agents.caching.enabled": true,
  "agents.mcpServers": [],
  "agents.experimental.enabled": false
}
```

---

## Next Steps

1. ✅ Read the [Quick Start Guide](./QUICK_START.md)
2. ✅ Review [Architecture Documentation](./ARCHITECTURE_DOCUMENTATION.md)
3. ✅ Explore [API Documentation](./API_DOCUMENTATION.md)
4. ✅ Check out [Release Notes](./AGENTS_VS_CODE_RELEASE_NOTES.md)

---

## Support

- 📧 **Email:** harigov63@gmail.com
- 🌐 **Website:** https://vane-enterprise.github.io
- 🐙 **GitHub:** https://github.com/AnticipatedD/AnticipatedD
- 💬 **Discussions:** https://github.com/AnticipatedD/AnticipatedD/discussions
- 🐛 **Issues:** https://github.com/AnticipatedD/AnticipatedD/issues

---

*Last Updated: June 6, 2026*