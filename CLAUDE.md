@AGENTS.md

## Config generation

When creating or customizing CVS `config.json` files (single, distributed, or disaggregated topologies), follow the **cvs-config-generator** skill:

- **Claude Code:** `.claude/skills/cvs-config-generator/SKILL.md`


Without an AI IDE, read `.claude/skills/cvs-config-generator/frameworks.md` and run:

```bash
python .claude/skills/cvs-config-generator/scripts/validate_xdit_config.py path/to/config.json
```
