# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

Claude Code plugin (v1.0.0) providing Git operations and GitHub CLI workflows through:
- **2 Agents**: git-ops (Haiku, lightweight Git commands) and github-cli (Sonnet, GitHub CLI operations)
- **2 Skills**: gh-cli (GitHub CLI utilities) and github-actions-writer (CI/CD workflow generation)

## Dependencies

- Python 3.9+
- gh CLI (authenticated via `gh auth login`)
- PyYAML (for workflow validation scripts)

## Quick Reference

### gh-cli Scripts

```bash
# Search GitHub code with filtering
python3 skills/gh-cli/scripts/gh_code_search.py "pattern" --language python --format pretty

# Analyze most recent failed GitHub Actions run
python3 skills/gh-cli/scripts/gh_failed_run.py

# Deploy to GitHub Pages
python3 skills/gh-cli/scripts/gh_pages_deploy.py --help
```

### github-actions-writer Scripts

```bash
# Validate workflow YAML
python3 skills/github-actions-writer/scripts/validate_workflow.py .github/workflows/ci.yml

# Security audit (use in CI with --fail-on)
python3 skills/github-actions-writer/scripts/security_audit.py .github/workflows/ci.yml --fail-on=high
```

## Plugin Structure

```
version-control/
├── .claude-plugin/plugin.json    # Plugin manifest
├── agents/
│   ├── git-ops.md                # Git operations (Haiku model)
│   └── github-cli.md             # GitHub CLI operations (Sonnet model)
└── skills/
    ├── gh-cli/                   # GitHub CLI utilities
    │   ├── SKILL.md
    │   └── scripts/              # gh_code_search, gh_failed_run, gh_pages_deploy
    └── github-actions-writer/    # Workflow generation
        ├── SKILL.md
        ├── scripts/              # validate_workflow, security_audit
        ├── assets/templates/     # CI/CD/security workflow templates
        └── references/           # Syntax and security guides
```

## Skill Documentation

- [gh-cli/SKILL.md](skills/gh-cli/SKILL.md) - GitHub CLI utilities and script usage
- [github-actions-writer/SKILL.md](skills/github-actions-writer/SKILL.md) - Workflow generation, templates, validation

## GitHub Actions Best Practices

The github-actions-writer skill enforces:
- Explicit minimal permissions (never `write-all`)
- OIDC for cloud authentication (no hardcoded credentials)
- Action version pinning with SHA hashes
- Input sanitization to prevent command injection
