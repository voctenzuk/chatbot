---
name: verify-imports
description: Verify all Python imports resolve correctly after file moves or refactoring
disable-model-invocation: true
---

# Verify Imports

Check that all Python imports in the project resolve correctly. Use after moving files, renaming modules, or restructuring packages.

## Steps

1. Run pyright to catch unresolved imports:
   ```bash
   uv run pyright 2>&1 | grep -E "import|cannot be resolved|not found"
   ```

2. Grep for imports of old paths that should no longer exist (if cleaning up shims):
   ```bash
   # Check for imports from bot.services.* that have been moved
   grep -rn "from bot\.services\." src/bot/ tests/ --include="*.py" | grep -v "services/__init__" | grep -v "__pycache__"
   ```

3. Check for circular imports by attempting to import the main package:
   ```bash
   uv run python -c "import bot; import bot.conversation; import bot.memory; import bot.llm; import bot.media; import bot.infra; print('All imports OK')"
   ```

4. Run the full test suite to catch runtime import issues:
   ```bash
   uv run pytest --tb=short -q
   ```

5. Report:
   - Number of broken imports found
   - Files with broken imports (file:line)
   - Circular import chains (if any)
   - Test failures related to imports

## When to use

- After any `/refactor-phase` execution
- After removing backward-compatible shims
- After bulk import path updates
- When pyright reports unexplained errors

## Gotchas

- Re-export shims (`from bot.X import *`) will mask broken imports — pyright may not catch issues hidden behind `*` imports
- `__init__.py` files with try/except ImportError blocks will silently swallow import failures
- pytest may pass even with broken imports if the broken module is not imported in any test
