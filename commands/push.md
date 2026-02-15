---
description: Push branch to remote and create a pull request
argument-hint: [base-branch]
allowed-tools: [Bash(git:*), Bash(gh:*)]
---

## Context

- Current branch: !`git branch --show-current`
- Base branch: $ARGUMENTS
- Remote tracking: !`git rev-parse --abbrev-ref --symbolic-full-name @{u} 2>&1 || echo "NO_REMOTE_TRACKING"`
- Uncommitted changes: !`git status --porcelain`
- Commits ahead of base: !`git log --oneline ${ARGUMENTS:-main}..HEAD 2>/dev/null`
- Existing PR: !`gh pr list --head "$(git branch --show-current)" --json number,url --jq '.[0] | "PR #\(.number): \(.url)"' 2>/dev/null || echo "NONE"`

## Your Task

Push the current branch to remote and create a pull request. Base branch defaults to `main` if no argument was provided.

### Pre-flight Checks

Run these checks FIRST. If any fail, STOP and report the issue — do not push or create a PR.

1. **Branch check**: If current branch is `main` or `master`, STOP. Tell the user: "Cannot push directly to main. Create a feature branch first."
2. **Uncommitted changes**: If there are uncommitted changes shown above, WARN the user but continue.
3. **No commits**: If there are no commits ahead of the base branch, STOP. Tell the user: "No commits ahead of base branch — nothing to push."
4. **Existing PR**: If a PR already exists for this branch, STOP. Show the existing PR URL to the user.

### Push

- If remote tracking shows `NO_REMOTE_TRACKING`, run: `git push -u origin <current-branch>`
- Otherwise, run: `git push`

### Create Pull Request

Generate the PR title and body from the commits ahead of base:

**Single commit:**
- Title: Use the commit subject line directly
- Body: Use the commit body if present

**Multiple commits:**
- Title: Summarize the changes in under 70 characters
- Body: List each commit as a bullet point

Format the PR body like this:

```
## Summary
<1-3 bullet points summarizing the changes>

## Commits
<bullet list of each commit message>

Generated with [Claude Code](https://claude.com/claude-code)
```

Create the PR:
```
gh pr create --title "TITLE" --body "BODY" --base <base-branch>
```

Use `--base $ARGUMENTS` if a base branch argument was provided, otherwise use `--base main`.

### Output

Report the results:
- Whether the push succeeded
- The PR URL

You MUST do all of the above using tool calls in a single message. Do not send any other text besides the tool calls and final confirmation.
