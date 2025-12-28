# Version Control

![Version](https://img.shields.io/badge/version-1.0.0-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Python](https://img.shields.io/badge/python-3.9+-yellow)

A Claude Code plugin for Git operations and GitHub CLI workflows.

## Quick Start

```bash
# Add the marketplace
/plugin marketplace add /path/to/claude-plugins

# Install the plugin
/plugin install version-control@plugins-by-james
```

## Table of Contents

- [Features](#features)
- [Agents](#agents)
- [Skills](#skills)
- [Usage Examples](#usage-examples)
- [Dependencies](#dependencies)
- [Security Best Practices](#security-best-practices)
- [Limitations](#limitations)
- [License](#license)

## Features

- **2 Agents** for task delegation:
  - `git-ops` - Lightweight Git operations (Haiku model)
  - `github-cli` - Full GitHub CLI workflows (Sonnet model)

- **2 Skills** for specialized capabilities:
  - `gh-cli` - Code search, workflow debugging, Pages deployment
  - `github-actions-writer` - CI/CD workflow generation with templates

## Agents

### git-ops

Lightweight Git specialist using the Haiku model. Ideal for:

- Status checks, branch operations, commits
- Conventional commit format: `type(scope): description`
- Safe operations with pre-flight checks

**Safety rules:**
- Always checks `git status` first
- Verifies before destructive operations
- Never force pushes without confirmation

### github-cli

Full GitHub CLI specialist using the Sonnet model. Handles:

- Issues & PRs: create, list, view, edit, comment, merge, review
- Repositories: create, clone, fork, settings, deploy keys
- Releases: create, edit, upload/download assets
- Workflows: trigger, view, cancel, rerun, watch
- Projects: create, edit, manage fields and items
- Search: code, commits, issues, PRs, repositories
- API access: `gh api` for any GitHub operation

**Best practices:**
- Uses `--json` with `--jq` for structured output
- Uses `-R owner/repo` when not in git directory
- Verifies command success before proceeding

## Skills

### gh-cli

Three Python utilities for common GitHub workflows:

| Script | Purpose |
|--------|---------|
| `gh_code_search.py` | Advanced code search with filtering, formatting, sorting |
| `gh_failed_run.py` | Extract errors from most recent failed Actions run |
| `gh_pages_deploy.py` | GitHub Pages automation (enable, status, deploy) |

```bash
# Search GitHub code
python3 skills/gh-cli/scripts/gh_code_search.py "pattern" --language python --format pretty

# Analyze workflow failures
python3 skills/gh-cli/scripts/gh_failed_run.py

# Deploy to GitHub Pages
python3 skills/gh-cli/scripts/gh_pages_deploy.py --help
```

**Documentation:** `skills/gh-cli/references/`

### github-actions-writer

Production-ready workflow generation with templates and validation.

**Templates** (`skills/github-actions-writer/assets/templates/`):

| Category | Templates |
|----------|-----------|
| CI | `node-ci.yml`, `python-ci.yml`, `docker-build-push.yml`, `multi-language-matrix.yml`, `monorepo-selective.yml` |
| CD | `aws-oidc-deploy.yml`, `kubernetes-gitops.yml`, `multi-environment.yml` |
| Security | `security-scan.yml` (CodeQL, Snyk, Trivy) |
| Advanced | `reusable-workflow.yml`, `composite-action/action.yml` |

**Validation scripts:**

```bash
# Validate workflow YAML
python3 skills/github-actions-writer/scripts/validate_workflow.py .github/workflows/ci.yml

# Security audit (use --fail-on for CI)
python3 skills/github-actions-writer/scripts/security_audit.py .github/workflows/*.yml --fail-on=high
```

**References:** `skills/github-actions-writer/references/` (syntax, security, troubleshooting)

## Usage Examples

**Create a new CI workflow:**
```
"Create a Node.js CI workflow that tests on PRs"
→ Generates workflow with caching, concurrency, minimal permissions
```

**Debug failing Actions:**
```
"Why did my last workflow fail?"
→ Uses gh_failed_run.py to extract error patterns
```

**Search for code patterns:**
```
"Search for all API key usage in my org"
→ Uses gh_code_search.py with filters
```

**Optimize existing workflow:**
```
"My CI takes 15 minutes, make it faster"
→ Analyzes for missing caching, parallelization, concurrency
```

## Dependencies

- **Python 3.9+**
- **gh CLI** - Authenticated via `gh auth login`
- **PyYAML** - For workflow validation scripts

## Security Best Practices

The github-actions-writer skill enforces:

| Practice | Description |
|----------|-------------|
| Minimal permissions | Explicit permissions, never `write-all` |
| OIDC authentication | No long-lived cloud credentials |
| Action pinning | Versions pinned to SHA for security |
| Input sanitization | Prevent command injection attacks |
| Timeouts | All jobs have appropriate limits |
| Concurrency control | Cancel stale runs automatically |

## Limitations

- Requires `gh auth login` before using gh-cli scripts
- OIDC setup requires cloud provider configuration
- Workflow templates may need customization for specific projects

## License

MIT
