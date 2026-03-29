# Development Guidelines

## Code Quality Checks

After completing each task, always run both linters to ensure code quality:

```bash
uv run --extra dev ruff check .

uv run --extra dev ty check kp_regression
```

Both commands should pass with no errors before considering a task complete.

## Error Handling Philosophy

- **Do not be overly cautious** - avoid wrapping everything in try/except blocks
- Let the code fail loudly when there are actual errors
- Only add exception handling when:
  - You expect specific, recoverable errors
  - You need to clean up resources
  - You want to provide better error messages
- Prefer explicit error checking over broad exception catching

## Inspecting Code and APIs

When unsure about methods or attributes of a class:

- **Do NOT use `getattr()` or `hasattr()` to check attributes**
- **Do NOT guess or assume behavior**
- Instead:
  1. Read the source code directly
  2. Use web search to find official documentation
  3. Inspect the actual implementation

## Code Comments

- **Do NOT add comments to code unless explicitly requested**
