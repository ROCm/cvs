@AGENTS.md

## Config generation

When creating or customizing CVS `config.json` files (single, distributed, or disaggregated topologies), follow the **cvs-config-generator** skill at `.claude/skills/cvs-config-generator/SKILL.md` (Claude Code, Cursor, and other AI IDEs).

Invoke explicitly, e.g. *"Follow cvs-config-generator and create a SGLang distributed config for …"*.

Without an AI IDE, read `.claude/skills/cvs-config-generator/frameworks.md` and run:

```bash
python .claude/skills/cvs-config-generator/scripts/validate_config.py path/to/config.json
```
