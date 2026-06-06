# Agents Visual Studio Code Extension - API Documentation

## Overview

The Agents VS Code Extension provides a comprehensive API for developers and team leads to integrate intelligent agent workflows into their development process. This API enables custom agent configurations, workflow automation, and seamless integration with existing development tools.

## Table of Contents
1. [Core Modules](#core-modules)
2. [Agent Interface](#agent-interface)
3. [Workflow API](#workflow-api)
4. [MCP Server Integration](#mcp-server-integration)
5. [Custom Instructions API](#custom-instructions-api)
6. [Marketplace API](#marketplace-api)
7. [Error Handling](#error-handling)
8. [Examples](#examples)

---

## Core Modules

### 1. Agent Manager

The central module for managing agent lifecycle and operations.

```typescript
interface AgentManager {
  // Initialize agent with configuration
  initialize(config: AgentConfig): Promise<Agent>;
  
  // List all active agents
  listAgents(): Agent[];
  
  // Get specific agent by ID
  getAgent(agentId: string): Agent | undefined;
  
  // Create new agent instance
  createAgent(config: AgentConfig): Promise<Agent>;
  
  // Delete agent instance
  deleteAgent(agentId: string): Promise<boolean>;
  
  // Execute agent workflow
  executeWorkflow(agentId: string, workflow: Workflow): Promise<WorkflowResult>;
}
```

### 2. Code Generation Engine

Responsible for intelligent code scaffolding and generation.

```typescript
interface CodeGenerator {
  // Generate code from natural language description
  generateFromDescription(description: string, context: CodeContext): Promise<GeneratedCode>;
  
  // Scaffold feature structure
  scaffoldFeature(featureName: string, template: TemplateType): Promise<ScaffoldResult>;
  
  // Apply multi-file changes
  applyMultiFileChanges(changes: CodeChange[]): Promise<ApplyResult>;
  
  // Validate generated code
  validateCode(code: string, rules: ValidationRule[]): ValidationResult;
}

interface GeneratedCode {
  files: FileContent[];
  metadata: CodeMetadata;
  confidence: number; // 0-1 confidence score
  sources: string[]; // Source references
}
```

### 3. Test Verification Engine

Automated test creation, execution, and validation.

```typescript
interface TestEngine {
  // Create tests from code
  generateTests(code: string, language: string): Promise<TestSuite>;
  
  // Run test suite
  runTests(testSuite: TestSuite): Promise<TestResult>;
  
  // Validate test coverage
  validateCoverage(coverage: CoverageReport, threshold: number): boolean;
  
  // Generate coverage report
  generateCoverageReport(testResults: TestResult[]): CoverageReport;
}

interface TestResult {
  passed: number;
  failed: number;
  skipped: number;
  duration: number; // milliseconds
  details: TestDetails[];
}
```

---

## Agent Interface

### AgentConfig

```typescript
interface AgentConfig {
  // Unique identifier
  id: string;
  
  // Agent name and description
  name: string;
  description: string;
  
  // Agent type (developer, architect, reviewer, etc.)
  type: 'developer' | 'architect' | 'reviewer' | 'custom';
  
  // Custom instructions for agent behavior
  customInstructions?: CustomInstruction[];
  
  // MCP servers to connect
  mcpServers?: MCPServerConfig[];
  
  // LLM configuration
  llm: {
    model: string;
    provider: 'openai' | 'anthropic' | 'local';
    temperature: number;
    maxTokens: number;
  };
  
  // Workflow configuration
  workflows: WorkflowConfig[];
}
```

### Agent Methods

```typescript
interface Agent {
  config: AgentConfig;
  
  // Start agent
  start(): Promise<void>;
  
  // Stop agent
  stop(): Promise<void>;
  
  // Process user request
  processRequest(request: AgentRequest): Promise<AgentResponse>;
  
  // Get agent status
  getStatus(): AgentStatus;
  
  // Update configuration
  updateConfig(config: Partial<AgentConfig>): Promise<void>;
  
  // Listen to agent events
  on(event: string, handler: Function): void;
  off(event: string, handler: Function): void;
}

interface AgentRequest {
  type: 'feature' | 'refactor' | 'document' | 'test' | 'custom';
  description: string;
  context?: CodeContext;
  options?: Record<string, any>;
}

interface AgentResponse {
  status: 'success' | 'error' | 'partial';
  result: any;
  metadata: {
    executionTime: number;
    tokensUsed: number;
    confidence: number;
  };
  errors?: ErrorDetail[];
}
```

---

## Workflow API

### Workflow Definition

```typescript
interface Workflow {
  // Unique workflow identifier
  id: string;
  
  // Workflow name
  name: string;
  
  // Description of what workflow does
  description: string;
  
  // Sequence of steps
  steps: WorkflowStep[];
  
  // Trigger conditions
  triggers: WorkflowTrigger[];
  
  // Success/failure handlers
  handlers: WorkflowHandler[];
}

interface WorkflowStep {
  id: string;
  name: string;
  action: WorkflowAction;
  inputs?: Record<string, any>;
  outputs?: string[];
  errorHandling?: 'continue' | 'retry' | 'fail';
  retryPolicy?: RetryPolicy;
}

type WorkflowAction = 
  | 'generate_code'
  | 'run_tests'
  | 'validate'
  | 'commit'
  | 'deploy'
  | 'document'
  | 'custom';

interface WorkflowTrigger {
  type: 'manual' | 'scheduled' | 'event';
  condition?: string;
}
```

### Workflow Execution

```typescript
interface WorkflowExecutor {
  // Execute workflow
  execute(workflow: Workflow, inputs: Record<string, any>): Promise<WorkflowResult>;
  
  // Execute workflow with progress tracking
  executeWithProgress(
    workflow: Workflow,
    inputs: Record<string, any>,
    onProgress: (progress: WorkflowProgress) => void
  ): Promise<WorkflowResult>;
  
  // Get execution history
  getHistory(workflowId: string): WorkflowExecution[];
  
  // Cancel running workflow
  cancel(executionId: string): Promise<void>;
}

interface WorkflowResult {
  executionId: string;
  workflowId: string;
  status: 'success' | 'failed' | 'cancelled';
  startTime: number;
  endTime: number;
  duration: number;
  outputs: Record<string, any>;
  stepResults: StepResult[];
  errors?: ErrorDetail[];
}
```

---

## MCP Server Integration

### MCP Server Configuration

```typescript
interface MCPServerConfig {
  // Server identifier
  id: string;
  
  // Server name
  name: string;
  
  // Server type
  type: 'stdio' | 'sse' | 'websocket' | 'custom';
  
  // Connection details
  connection: {
    host?: string;
    port?: number;
    command?: string;
    args?: string[];
  };
  
  // Authentication
  auth?: {
    type: 'none' | 'token' | 'oauth';
    credentials: Record<string, string>;
  };
  
  // Server capabilities
  capabilities?: string[];
}
```

### MCP Client Interface

```typescript
interface MCPClient {
  // Connect to MCP server
  connect(config: MCPServerConfig): Promise<MCPConnection>;
  
  // Disconnect from server
  disconnect(serverId: string): Promise<void>;
  
  // List available tools
  listTools(serverId: string): Promise<Tool[]>;
  
  // Call MCP tool
  callTool(
    serverId: string,
    toolName: string,
    args: Record<string, any>
  ): Promise<ToolResult>;
  
  // Get server status
  getStatus(serverId: string): ServerStatus;
}

interface Tool {
  name: string;
  description: string;
  inputSchema: JSONSchema;
  outputSchema: JSONSchema;
}
```

---

## Custom Instructions API

### Custom Instruction Definition

```typescript
interface CustomInstruction {
  // Instruction identifier
  id: string;
  
  // Instruction name
  name: string;
  
  // The instruction text
  instruction: string;
  
  // Scope: where this applies
  scope: 'global' | 'repository' | 'project' | 'team';
  
  // Priority (higher = more important)
  priority: number;
  
  // Tags for categorization
  tags: string[];
  
  // When to apply
  appliedTo: ApplyCondition[];
  
  // Status
  enabled: boolean;
}

interface ApplyCondition {
  type: 'language' | 'pattern' | 'module' | 'custom';
  value: string;
  regex?: boolean;
}

interface CustomInstructionManager {
  // Create new instruction
  create(instruction: CustomInstruction): Promise<CustomInstruction>;
  
  // Update instruction
  update(id: string, updates: Partial<CustomInstruction>): Promise<CustomInstruction>;
  
  // Delete instruction
  delete(id: string): Promise<boolean>;
  
  // List instructions
  list(scope?: string): CustomInstruction[];
  
  // Get applicable instructions for context
  getApplicable(context: CodeContext): CustomInstruction[];
  
  // Enable/disable instruction
  setEnabled(id: string, enabled: boolean): Promise<void>;
}
```

---

## Marketplace API

### Marketplace Package Structure

```typescript
interface MarketplacePackage {
  // Package metadata
  id: string;
  name: string;
  version: string;
  author: string;
  description: string;
  keywords: string[];
  
  // Package type
  type: 'instruction' | 'template' | 'tool' | 'integration';
  
  // Dependencies
  dependencies: Record<string, string>;
  
  // Manifest
  manifest: PackageManifest;
  
  // Ratings and reviews
  rating: number; // 0-5
  downloads: number;
  
  // Repository
  repository?: string;
  license: string;
}

interface PackageManifest {
  // Main entry point
  main: string;
  
  // Configuration schema
  configSchema: JSONSchema;
  
  // Required permissions
  permissions: string[];
  
  // Activation events
  activationEvents: string[];
}
```

### Marketplace Client

```typescript
interface MarketplaceClient {
  // Search packages
  search(
    query: string,
    filters?: SearchFilters
  ): Promise<MarketplacePackage[]>;
  
  // Get package details
  getPackage(packageId: string): Promise<MarketplacePackage>;
  
  // Install package
  install(
    packageId: string,
    version?: string
  ): Promise<InstallResult>;
  
  // Uninstall package
  uninstall(packageId: string): Promise<void>;
  
  // Get installed packages
  getInstalledPackages(): MarketplacePackage[];
  
  // Check for updates
  checkUpdates(): Promise<UpdateAvailable[]>;
  
  // Update package
  update(packageId: string, version: string): Promise<UpdateResult>;
  
  // Publish package
  publish(packagePath: string): Promise<PublishResult>;
}
```

---

## Error Handling

### Error Types

```typescript
class AgentError extends Error {
  code: string;
  context?: Record<string, any>;
  recoverable: boolean;
}

class ValidationError extends AgentError {
  validationErrors: ValidationErrorDetail[];
}

class MCPConnectionError extends AgentError {
  serverId: string;
  originalError: Error;
}

class WorkflowExecutionError extends AgentError {
  stepId: string;
  stepName: string;
  originalError: Error;
}

interface ErrorDetail {
  code: string;
  message: string;
  stack?: string;
  context?: Record<string, any>;
}
```

### Error Handling Best Practices

```typescript
try {
  const result = await agent.processRequest({
    type: 'feature',
    description: 'Add user authentication'
  });
} catch (error) {
  if (error instanceof ValidationError) {
    console.error('Validation failed:', error.validationErrors);
  } else if (error instanceof MCPConnectionError) {
    console.error('MCP server error:', error.serverId);
  } else if (error instanceof AgentError && error.recoverable) {
    // Retry logic
  }
}
```

---

## Examples

### Example 1: Create and Execute an Agent

```typescript
import { AgentManager } from 'agents-vscode';

const manager = new AgentManager();

const agent = await manager.createAgent({
  id: 'dev-agent-1',
  name: 'Development Agent',
  type: 'developer',
  llm: {
    model: 'gpt-4',
    provider: 'openai',
    temperature: 0.7,
    maxTokens: 2000
  },
  workflows: []
});

const response = await agent.processRequest({
  type: 'feature',
  description: 'Create user authentication module with JWT'
});

if (response.status === 'success') {
  console.log('Generated files:', response.result.files);
}
```

### Example 2: Configure Custom Instructions

```typescript
import { CustomInstructionManager } from 'agents-vscode';

const manager = new CustomInstructionManager();

await manager.create({
  id: 'typescript-best-practices',
  name: 'TypeScript Best Practices',
  instruction: 'Always use strict type checking and avoid any types',
  scope: 'repository',
  priority: 10,
  tags: ['typescript', 'quality'],
  appliedTo: [
    { type: 'language', value: 'typescript' }
  ],
  enabled: true
});
```

### Example 3: MCP Server Integration

```typescript
import { MCPClientFactory } from 'agents-vscode';

const mcpClient = MCPClientFactory.create();

const connection = await mcpClient.connect({
  id: 'code-analyzer',
  name: 'Code Analyzer MCP',
  type: 'stdio',
  connection: {
    command: 'python',
    args: ['-m', 'code_analyzer']
  }
});

const tools = await mcpClient.listTools('code-analyzer');
const result = await mcpClient.callTool(
  'code-analyzer',
  'analyze_complexity',
  { filePath: 'src/index.ts' }
);
```

### Example 4: Workflow Execution

```typescript
import { WorkflowExecutor } from 'agents-vscode';

const executor = new WorkflowExecutor();

const workflow: Workflow = {
  id: 'feature-workflow',
  name: 'Complete Feature Development',
  description: 'Generate, test, and document a new feature',
  steps: [
    {
      id: 'generate',
      name: 'Generate Code',
      action: 'generate_code'
    },
    {
      id: 'test',
      name: 'Run Tests',
      action: 'run_tests'
    },
    {
      id: 'document',
      name: 'Generate Documentation',
      action: 'document'
    }
  ],
  triggers: [{ type: 'manual' }],
  handlers: []
};

const result = await executor.executeWithProgress(
  workflow,
  { featureDescription: 'User profile management' },
  (progress) => {
    console.log(`Progress: ${progress.currentStep}/${progress.totalSteps}`);
  }
);
```

---

## Support & Resources

- 📚 **Full Documentation:** https://github.com/AnticipatedD/AnticipatedD
- 🐛 **Issue Tracking:** https://github.com/AnticipatedD/AnticipatedD/issues
- 💬 **Discussions:** https://github.com/AnticipatedD/AnticipatedD/discussions
- 📧 **Contact:** harigov63@gmail.com

---

*Last Updated: June 6, 2026*