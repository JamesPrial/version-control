"""
Comprehensive test suite for security_audit.py GitHub Actions security auditor.

Tests cover:
- YAML file parsing and validation
- Excessive permissions detection (write-all, contents: write, etc.)
- Dangerous trigger detection (pull_request_target, workflow_run)
- Hardcoded secrets exposure (passwords, API keys, tokens, GitHub PATs, AWS keys)
- Action security issues (unpinned versions, mutable refs)
- Command injection vulnerabilities (9 dangerous contexts)
- Self-hosted runner security (public PR + self-hosted detection)
- Results formatting and list clearing
- Main function integration tests with --fail-on threshold
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any
from unittest.mock import patch

import pytest
import yaml

from security_audit import (
    SecurityAuditor,
    main,
)

if TYPE_CHECKING:
    from collections.abc import Generator


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def tmp_workflow(tmp_path: Path) -> Generator[Path, None, None]:
    """Create a temporary workflow file for testing."""
    workflow_path = tmp_path / ".github" / "workflows" / "test.yml"
    workflow_path.parent.mkdir(parents=True, exist_ok=True)
    yield workflow_path
    if workflow_path.exists():
        workflow_path.unlink()


@pytest.fixture
def auditor() -> SecurityAuditor:
    """Create a fresh SecurityAuditor instance for testing."""
    return SecurityAuditor()


@pytest.fixture
def minimal_workflow() -> dict[str, Any]:
    """Minimal valid GitHub Actions workflow."""
    return {
        "name": "Test Workflow",
        "on": "push",
        "jobs": {
            "build": {
                "runs-on": "ubuntu-latest",
                "steps": [{"uses": "actions/checkout@v3"}],
            }
        },
    }


def write_workflow_yaml(path: Path, workflow: dict[str, Any]) -> None:
    """Helper to write workflow dict to YAML file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.dump(workflow))


# =============================================================================
# TestAuditFile
# =============================================================================


class TestAuditFile:
    """Tests for SecurityAuditor.audit_file() method."""

    def test_audit_file_nonexistent_returns_error(
        self, auditor: SecurityAuditor, tmp_path: Path
    ) -> None:
        """Audit nonexistent file adds critical error and returns summary."""
        nonexistent = tmp_path / "nonexistent.yml"
        summary = auditor.audit_file(nonexistent)

        assert summary["critical"] == 1
        # Note: For file-not-found, audit_file calls get_summary() not print_results()
        # so the lists are NOT cleared (inconsistent with successful audits)
        assert len(auditor.critical) == 1
        assert "File not found" in auditor.critical[0]

    def test_audit_file_invalid_yaml_returns_error(
        self, auditor: SecurityAuditor, tmp_workflow: Path
    ) -> None:
        """Invalid YAML file adds critical error."""
        tmp_workflow.write_text("invalid: yaml: content:")
        summary = auditor.audit_file(tmp_workflow)

        # Summary captures the error count before lists are cleared
        assert summary["critical"] == 1
        # Lists are cleared after print_results, so we verify via summary only
        assert summary["high"] == 0
        assert summary["medium"] == 0
        assert summary["low"] == 0

    def test_audit_file_valid_workflow_returns_summary(
        self, auditor: SecurityAuditor, tmp_workflow: Path, minimal_workflow: dict[str, Any]
    ) -> None:
        """Valid workflow returns summary dict with severity counts."""
        write_workflow_yaml(tmp_workflow, minimal_workflow)
        summary = auditor.audit_file(tmp_workflow)

        assert isinstance(summary, dict)
        assert "critical" in summary
        assert "high" in summary
        assert "medium" in summary
        assert "low" in summary
        assert all(isinstance(v, int) for v in summary.values())

    def test_audit_file_not_dict_returns_empty_summary(
        self, auditor: SecurityAuditor, tmp_workflow: Path
    ) -> None:
        """YAML that's not a dict returns empty summary."""
        tmp_workflow.write_text("- item1\n- item2\n")
        summary = auditor.audit_file(tmp_workflow)

        # Should return empty summary (no issues)
        assert summary == {"critical": 0, "high": 0, "medium": 0, "low": 0}

    def test_audit_file_runs_all_checks(
        self, auditor: SecurityAuditor, tmp_workflow: Path
    ) -> None:
        """audit_file runs all security checks."""
        workflow = {
            "name": "Test",
            "on": "push",
            "permissions": "write-all",  # Critical: write-all
            "jobs": {
                "build": {
                    "runs-on": "ubuntu-latest",
                    "steps": [
                        {
                            "uses": "some-action",  # High: unpinned
                            "run": "echo {{ github.event.issue.title }}",
                        }
                    ],
                }
            },
        }
        write_workflow_yaml(tmp_workflow, workflow)
        summary = auditor.audit_file(tmp_workflow)

        assert summary["critical"] > 0 or summary["high"] > 0

    def test_audit_file_clears_lists_after_print_results(
        self, auditor: SecurityAuditor, tmp_workflow: Path, minimal_workflow: dict[str, Any]
    ) -> None:
        """Lists are cleared after audit_file returns."""
        write_workflow_yaml(tmp_workflow, minimal_workflow)
        auditor.audit_file(tmp_workflow)

        # All lists should be empty (cleared by print_results)
        assert auditor.critical == []
        assert auditor.high == []
        assert auditor.medium == []
        assert auditor.low == []

    def test_audit_file_empty_file_returns_empty_summary(
        self, auditor: SecurityAuditor, tmp_workflow: Path
    ) -> None:
        """Empty file (0 bytes) returns empty summary with no crashes."""
        tmp_workflow.write_text("")
        summary = auditor.audit_file(tmp_workflow)

        # Empty file parses as None, which is not a dict
        assert summary == {"critical": 0, "high": 0, "medium": 0, "low": 0}

    def test_audit_file_whitespace_only_returns_empty_summary(
        self, auditor: SecurityAuditor, tmp_workflow: Path
    ) -> None:
        """Whitespace-only file returns empty summary."""
        tmp_workflow.write_text("   \n\n   \n")
        summary = auditor.audit_file(tmp_workflow)

        assert summary == {"critical": 0, "high": 0, "medium": 0, "low": 0}


# =============================================================================
# TestCheckPermissions
# =============================================================================


class TestCheckPermissions:
    """Tests for SecurityAuditor.check_permissions() method."""

    def test_permissions_write_all_adds_critical(self, auditor: SecurityAuditor) -> None:
        """permissions: write-all adds critical issue."""
        workflow = {"permissions": "write-all"}
        auditor.check_permissions(workflow)

        assert len(auditor.critical) == 1
        assert "write-all" in auditor.critical[0]

    def test_permissions_none_adds_high(self, auditor: SecurityAuditor) -> None:
        """No permissions specified adds high severity issue."""
        workflow = {}
        auditor.check_permissions(workflow)

        assert len(auditor.high) == 1
        assert "No permissions specified" in auditor.high[0]

    def test_permissions_contents_write_adds_medium(self, auditor: SecurityAuditor) -> None:
        """contents: write adds medium severity issue."""
        workflow = {"permissions": {"contents": "write"}}
        auditor.check_permissions(workflow)

        assert len(auditor.medium) == 1
        assert "contents: write" in auditor.medium[0]

    def test_permissions_packages_write_adds_medium(self, auditor: SecurityAuditor) -> None:
        """packages: write adds medium severity issue."""
        workflow = {"permissions": {"packages": "write"}}
        auditor.check_permissions(workflow)

        assert len(auditor.medium) == 1
        assert "packages: write" in auditor.medium[0]

    def test_permissions_deployments_write_adds_medium(
        self, auditor: SecurityAuditor
    ) -> None:
        """deployments: write adds medium severity issue."""
        workflow = {"permissions": {"deployments": "write"}}
        auditor.check_permissions(workflow)

        assert len(auditor.medium) == 1
        assert "deployments: write" in auditor.medium[0]

    def test_permissions_multiple_write_perms_combined(self, auditor: SecurityAuditor) -> None:
        """Multiple write permissions combined in one issue."""
        workflow = {
            "permissions": {
                "contents": "write",
                "packages": "write",
            }
        }
        auditor.check_permissions(workflow)

        assert len(auditor.medium) == 1
        assert "contents: write" in auditor.medium[0]
        assert "packages: write" in auditor.medium[0]

    def test_permissions_contents_read_no_issue(self, auditor: SecurityAuditor) -> None:
        """contents: read does not trigger issue."""
        workflow = {"permissions": {"contents": "read"}}
        auditor.check_permissions(workflow)

        assert len(auditor.medium) == 0

    def test_job_permissions_write_all_adds_critical(self, auditor: SecurityAuditor) -> None:
        """Job-level permissions: write-all adds critical."""
        workflow = {
            "jobs": {
                "build": {
                    "permissions": "write-all",
                }
            }
        }
        auditor.check_permissions(workflow)

        assert len(auditor.critical) == 1
        assert "write-all" in auditor.critical[0]

    def test_job_permissions_read_no_issue(self, auditor: SecurityAuditor) -> None:
        """Job with read permissions doesn't trigger issue."""
        workflow = {
            "jobs": {
                "build": {
                    "permissions": {"contents": "read"},
                }
            }
        }
        auditor.check_permissions(workflow)

        assert len(auditor.medium) == 0


# =============================================================================
# TestCheckDangerousTriggers
# =============================================================================


class TestCheckDangerousTriggers:
    """Tests for SecurityAuditor.check_dangerous_triggers() method."""

    def test_pull_request_target_adds_high(self, auditor: SecurityAuditor) -> None:
        """pull_request_target trigger adds high severity issue."""
        workflow = {"on": {"pull_request_target": None}}
        auditor.check_dangerous_triggers(workflow)

        assert len(auditor.high) == 1
        assert "pull_request_target" in auditor.high[0]

    def test_workflow_run_adds_medium(self, auditor: SecurityAuditor) -> None:
        """workflow_run trigger adds medium severity issue."""
        workflow = {"on": {"workflow_run": None}}
        auditor.check_dangerous_triggers(workflow)

        assert len(auditor.medium) == 1
        assert "workflow_run" in auditor.medium[0]

    def test_push_trigger_no_issue(self, auditor: SecurityAuditor) -> None:
        """push trigger doesn't trigger issue."""
        workflow = {"on": {"push": None}}
        auditor.check_dangerous_triggers(workflow)

        assert len(auditor.high) == 0
        assert len(auditor.medium) == 0

    def test_triggers_as_string_normalized(self, auditor: SecurityAuditor) -> None:
        """Trigger as string is normalized to dict."""
        workflow = {"on": "pull_request_target"}
        auditor.check_dangerous_triggers(workflow)

        assert len(auditor.high) == 1

    def test_triggers_as_list_normalized(self, auditor: SecurityAuditor) -> None:
        """Trigger as list is normalized to dict."""
        workflow = {"on": ["push", "pull_request_target"]}
        auditor.check_dangerous_triggers(workflow)

        assert len(auditor.high) == 1

    def test_multiple_dangerous_triggers(self, auditor: SecurityAuditor) -> None:
        """Multiple dangerous triggers both detected."""
        workflow = {
            "on": {
                "pull_request_target": None,
                "workflow_run": None,
            }
        }
        auditor.check_dangerous_triggers(workflow)

        assert len(auditor.high) == 1
        assert len(auditor.medium) == 1


# =============================================================================
# TestCheckSecretsExposure
# =============================================================================


class TestCheckSecretsExposure:
    """Tests for SecurityAuditor.check_secrets_exposure() method."""

    def test_hardcoded_password_adds_critical(self, auditor: SecurityAuditor) -> None:
        """Hardcoded password pattern adds critical."""
        workflow = {}
        content = 'password = "secret123"'
        auditor.check_secrets_exposure(workflow, content)

        assert len(auditor.critical) == 1
        assert "password" in auditor.critical[0].lower()

    def test_hardcoded_api_key_adds_critical(self, auditor: SecurityAuditor) -> None:
        """Hardcoded API key pattern adds critical."""
        workflow = {}
        content = 'api_key = "sk-1234567890abcdef"'
        auditor.check_secrets_exposure(workflow, content)

        assert len(auditor.critical) == 1

    def test_hardcoded_token_adds_critical(self, auditor: SecurityAuditor) -> None:
        """Hardcoded token pattern adds critical."""
        workflow = {}
        content = 'token = "secret_token_value"'
        auditor.check_secrets_exposure(workflow, content)

        assert len(auditor.critical) == 1

    def test_github_pat_adds_critical(self, auditor: SecurityAuditor) -> None:
        """GitHub PAT pattern (ghp_) adds critical."""
        workflow = {}
        content = 'ghp_' + 'a' * 36  # Valid GitHub PAT length
        auditor.check_secrets_exposure(workflow, content)

        assert len(auditor.critical) == 1
        assert "GitHub" in auditor.critical[0]

    def test_aws_access_key_adds_critical(self, auditor: SecurityAuditor) -> None:
        """AWS Access Key pattern (AKIA) adds critical."""
        workflow = {}
        content = "AKIA" + "0" * 16
        auditor.check_secrets_exposure(workflow, content)

        assert len(auditor.critical) == 1
        assert "AWS" in auditor.critical[0]

    def test_hardcoded_secret_adds_critical(self, auditor: SecurityAuditor) -> None:
        """Hardcoded secret pattern adds critical."""
        workflow = {}
        content = 'secret = "hidden_value"'
        auditor.check_secrets_exposure(workflow, content)

        assert len(auditor.critical) == 1

    def test_secret_in_run_command_adds_medium(self, auditor: SecurityAuditor) -> None:
        """Secret used directly in run command adds medium."""
        workflow = {
            "jobs": {
                "build": {
                    "steps": [
                        {
                            "name": "test",
                            "run": "echo ${{ secrets.API_KEY }}",
                        }
                    ]
                }
            }
        }
        content = "valid content"
        auditor.check_secrets_exposure(workflow, content)

        assert len(auditor.medium) == 1
        assert "Secret" in auditor.medium[0]

    def test_secret_in_env_not_flagged(self, auditor: SecurityAuditor) -> None:
        """Secret in step env is not flagged (safe context)."""
        workflow = {
            "jobs": {
                "build": {
                    "steps": [
                        {
                            "name": "test",
                            "run": "command",
                            "env": {
                                "MY_KEY": "${{ secrets.API_KEY }}",
                            },
                        }
                    ]
                }
            }
        }
        content = "valid content"
        auditor.check_secrets_exposure(workflow, content)

        # Should not flag when secret is in env (safe for logging)
        assert len(auditor.medium) == 0

    def test_multiple_secret_patterns_detected(self, auditor: SecurityAuditor) -> None:
        """Multiple secret patterns in one file all detected."""
        workflow = {}
        content = '''
        password = "secret123"
        api_key = "key_value"
        ghp_abcdefghij1234567890abcdefghijklmnop
        '''
        auditor.check_secrets_exposure(workflow, content)

        assert len(auditor.critical) >= 3

    def test_no_secrets_no_issues(self, auditor: SecurityAuditor) -> None:
        """Valid content without secrets doesn't add issues."""
        workflow = {
            "jobs": {
                "build": {
                    "steps": [
                        {
                            "name": "test",
                            "run": "echo hello",
                        }
                    ]
                }
            }
        }
        content = "echo hello world"
        auditor.check_secrets_exposure(workflow, content)

        assert len(auditor.critical) == 0
        assert len(auditor.medium) == 0


# =============================================================================
# TestCheckActionSecurity
# =============================================================================


class TestCheckActionSecurity:
    """Tests for SecurityAuditor.check_action_security() method."""

    def test_unpinned_action_adds_high(self, auditor: SecurityAuditor) -> None:
        """Action without version adds high severity."""
        workflow = {
            "jobs": {
                "build": {
                    "steps": [
                        {"uses": "actions/checkout"},
                    ]
                }
            }
        }
        auditor.check_action_security(workflow)

        assert len(auditor.high) == 1
        assert "not pinned" in auditor.high[0]

    def test_action_pinned_to_version_no_issue(self, auditor: SecurityAuditor) -> None:
        """Action pinned to version doesn't trigger issue."""
        workflow = {
            "jobs": {
                "build": {
                    "steps": [
                        {"uses": "actions/checkout@v3"},
                    ]
                }
            }
        }
        auditor.check_action_security(workflow)

        assert len(auditor.high) == 0

    def test_action_pinned_to_sha_no_issue(self, auditor: SecurityAuditor) -> None:
        """Action pinned to SHA doesn't trigger issue."""
        workflow = {
            "jobs": {
                "build": {
                    "steps": [
                        {"uses": "actions/checkout@abc123def456"},
                    ]
                }
            }
        }
        auditor.check_action_security(workflow)

        assert len(auditor.high) == 0

    def test_mutable_ref_main_adds_high(self, auditor: SecurityAuditor) -> None:
        """Action using 'main' branch adds high severity."""
        workflow = {
            "jobs": {
                "build": {
                    "steps": [
                        {"uses": "actions/checkout@main"},
                    ]
                }
            }
        }
        auditor.check_action_security(workflow)

        assert len(auditor.high) == 1
        assert "mutable reference" in auditor.high[0]

    def test_mutable_ref_master_adds_high(self, auditor: SecurityAuditor) -> None:
        """Action using 'master' branch adds high severity."""
        workflow = {
            "jobs": {
                "build": {
                    "steps": [
                        {"uses": "actions/checkout@master"},
                    ]
                }
            }
        }
        auditor.check_action_security(workflow)

        assert len(auditor.high) == 1

    def test_mutable_ref_latest_adds_high(self, auditor: SecurityAuditor) -> None:
        """Action using 'latest' tag adds high severity."""
        workflow = {
            "jobs": {
                "build": {
                    "steps": [
                        {"uses": "actions/checkout@latest"},
                    ]
                }
            }
        }
        auditor.check_action_security(workflow)

        assert len(auditor.high) == 1

    def test_mutable_ref_develop_adds_high(self, auditor: SecurityAuditor) -> None:
        """Action using 'develop' branch adds high severity."""
        workflow = {
            "jobs": {
                "build": {
                    "steps": [
                        {"uses": "actions/checkout@develop"},
                    ]
                }
            }
        }
        auditor.check_action_security(workflow)

        assert len(auditor.high) == 1

    def test_major_version_only_adds_low(self, auditor: SecurityAuditor) -> None:
        """Action pinned to major version only adds low."""
        workflow = {
            "jobs": {
                "build": {
                    "steps": [
                        {"uses": "actions/checkout@v3"},
                    ]
                }
            }
        }
        auditor.check_action_security(workflow)

        assert len(auditor.low) == 1
        assert "major version only" in auditor.low[0]

    def test_third_party_action_adds_low(self, auditor: SecurityAuditor) -> None:
        """Third-party action adds low severity INFO."""
        workflow = {
            "jobs": {
                "build": {
                    "steps": [
                        {"uses": "some-org/some-action@v1.0.0"},
                    ]
                }
            }
        }
        auditor.check_action_security(workflow)

        assert len(auditor.low) == 1
        assert "third-party" in auditor.low[0].lower()

    def test_official_actions_no_third_party_warning(self, auditor: SecurityAuditor) -> None:
        """Official actions/ and github/ don't trigger third-party warning."""
        workflow = {
            "jobs": {
                "build": {
                    "steps": [
                        {"uses": "actions/checkout@v3"},
                        {"uses": "github/super-linter@v4"},
                    ]
                }
            }
        }
        auditor.check_action_security(workflow)

        # Should not warn about actions/ or github/
        for issue in auditor.low:
            assert "third-party" not in issue.lower()

    @pytest.mark.parametrize(
        "version",
        ["v1", "v2", "v3", "v10"],
    )
    def test_major_version_formats(self, auditor: SecurityAuditor, version: str) -> None:
        """Major version patterns like v1, v2, etc. trigger low warning."""
        workflow = {
            "jobs": {
                "build": {
                    "steps": [
                        {"uses": f"actions/checkout@{version}"},
                    ]
                }
            }
        }
        auditor.check_action_security(workflow)

        assert any("major version" in issue.lower() for issue in auditor.low)


# =============================================================================
# TestCheckCommandInjection
# =============================================================================


class TestCheckCommandInjection:
    """Tests for SecurityAuditor.check_command_injection() method."""

    @pytest.mark.parametrize(
        "context",
        [
            "github.event.issue.title",
            "github.event.issue.body",
            "github.event.pull_request.title",
            "github.event.pull_request.body",
            "github.event.comment.body",
            "github.event.review.body",
            "github.event.discussion.title",
            "github.event.discussion.body",
            "github.head_ref",
        ],
    )
    def test_dangerous_context_in_run_adds_critical(
        self, auditor: SecurityAuditor, context: str
    ) -> None:
        """Dangerous context directly in run command adds critical."""
        workflow = {
            "jobs": {
                "build": {
                    "steps": [
                        {
                            "name": "vulnerable",
                            "run": f"echo ${{{{{context}}}}}",
                        }
                    ]
                }
            }
        }
        auditor.check_command_injection(workflow)

        assert len(auditor.critical) == 1
        assert "injection" in auditor.critical[0].lower()

    def test_dangerous_context_in_env_not_flagged(self, auditor: SecurityAuditor) -> None:
        """Dangerous context in step env is safe (not flagged)."""
        workflow = {
            "jobs": {
                "build": {
                    "steps": [
                        {
                            "name": "safe",
                            "run": "echo hello",
                            "env": {
                                "CONTEXT": "${{ github.event.issue.title }}",
                            },
                        }
                    ]
                }
            }
        }
        auditor.check_command_injection(workflow)

        assert len(auditor.critical) == 0

    def test_safe_context_in_run_no_issue(self, auditor: SecurityAuditor) -> None:
        """Safe context in run command doesn't trigger issue."""
        workflow = {
            "jobs": {
                "build": {
                    "steps": [
                        {
                            "name": "safe",
                            "run": "echo ${{ github.ref }}",
                        }
                    ]
                }
            }
        }
        auditor.check_command_injection(workflow)

        assert len(auditor.critical) == 0

    def test_multiple_dangerous_contexts_all_detected(
        self, auditor: SecurityAuditor
    ) -> None:
        """Multiple dangerous contexts all detected."""
        # Note: Source code checks for "${{context" pattern (no space after ${{)
        workflow = {
            "jobs": {
                "build": {
                    "steps": [
                        {
                            "run": "${{github.event.issue.title}} ${{github.head_ref}}",
                        }
                    ]
                }
            }
        }
        auditor.check_command_injection(workflow)

        # Both dangerous contexts should be detected
        assert len(auditor.critical) == 2


# =============================================================================
# TestCheckSelfHostedRunners
# =============================================================================


class TestCheckSelfHostedRunners:
    """Tests for SecurityAuditor.check_self_hosted_runners() method."""

    def test_self_hosted_with_pull_request_trigger_adds_critical(
        self, auditor: SecurityAuditor
    ) -> None:
        """Self-hosted runner with PR trigger adds critical."""
        workflow = {
            "on": {"pull_request": None},
            "jobs": {
                "build": {
                    "runs-on": "self-hosted",
                }
            },
        }
        auditor.check_self_hosted_runners(workflow)

        assert len(auditor.critical) == 1
        assert "self-hosted" in auditor.critical[0].lower()

    def test_self_hosted_with_pull_request_target_adds_critical(
        self, auditor: SecurityAuditor
    ) -> None:
        """Self-hosted runner with pull_request_target trigger adds critical."""
        workflow = {
            "on": {"pull_request_target": None},
            "jobs": {
                "build": {
                    "runs-on": "self-hosted",
                }
            },
        }
        auditor.check_self_hosted_runners(workflow)

        assert len(auditor.critical) == 1

    def test_self_hosted_without_public_pr_adds_medium(
        self, auditor: SecurityAuditor
    ) -> None:
        """Self-hosted runner without public PR trigger adds medium."""
        workflow = {
            "on": {"push": None},
            "jobs": {
                "build": {
                    "runs-on": "self-hosted",
                }
            },
        }
        auditor.check_self_hosted_runners(workflow)

        assert len(auditor.medium) == 1

    def test_ubuntu_runner_no_issue(self, auditor: SecurityAuditor) -> None:
        """Standard ubuntu runner doesn't trigger issue."""
        workflow = {
            "on": {"pull_request": None},
            "jobs": {
                "build": {
                    "runs-on": "ubuntu-latest",
                }
            },
        }
        auditor.check_self_hosted_runners(workflow)

        assert len(auditor.critical) == 0

    def test_self_hosted_as_list_with_public_pr_adds_critical(
        self, auditor: SecurityAuditor
    ) -> None:
        """Self-hosted in runs-on list with PR trigger adds critical."""
        workflow = {
            "on": {"pull_request": None},
            "jobs": {
                "build": {
                    "runs-on": ["ubuntu-latest", "self-hosted"],
                }
            },
        }
        auditor.check_self_hosted_runners(workflow)

        assert len(auditor.critical) == 1

    def test_triggers_as_string_normalized(self, auditor: SecurityAuditor) -> None:
        """Trigger as string is properly detected."""
        workflow = {
            "on": "pull_request",
            "jobs": {
                "build": {
                    "runs-on": "self-hosted",
                }
            },
        }
        auditor.check_self_hosted_runners(workflow)

        assert len(auditor.critical) == 1


# =============================================================================
# TestPrintResults
# =============================================================================


class TestPrintResults:
    """Tests for SecurityAuditor.print_results() method."""

    def test_print_results_returns_summary(
        self, auditor: SecurityAuditor, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """print_results returns summary dict."""
        auditor.critical.append("critical issue")
        auditor.high.append("high issue")

        summary = auditor.print_results()

        assert summary["critical"] == 1
        assert summary["high"] == 1
        assert summary["medium"] == 0
        assert summary["low"] == 0

    def test_print_results_clears_lists(self, auditor: SecurityAuditor) -> None:
        """print_results clears all issue lists."""
        auditor.critical.append("critical issue")
        auditor.high.append("high issue")
        auditor.medium.append("medium issue")
        auditor.low.append("low issue")

        auditor.print_results()

        assert auditor.critical == []
        assert auditor.high == []
        assert auditor.medium == []
        assert auditor.low == []

    def test_print_results_prints_critical(
        self, auditor: SecurityAuditor, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Prints critical issues with emoji."""
        auditor.critical.append("critical issue")

        auditor.print_results()
        captured = capsys.readouterr()

        assert "🔴" in captured.out
        assert "CRITICAL" in captured.out

    def test_print_results_prints_high(
        self, auditor: SecurityAuditor, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Prints high severity issues with emoji."""
        auditor.high.append("high issue")

        auditor.print_results()
        captured = capsys.readouterr()

        assert "🟠" in captured.out
        assert "HIGH" in captured.out

    def test_print_results_prints_medium(
        self, auditor: SecurityAuditor, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Prints medium severity issues with emoji."""
        auditor.medium.append("medium issue")

        auditor.print_results()
        captured = capsys.readouterr()

        assert "🟡" in captured.out
        assert "MEDIUM" in captured.out

    def test_print_results_prints_low(
        self, auditor: SecurityAuditor, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Prints low severity / INFO issues with emoji."""
        auditor.low.append("low issue")

        auditor.print_results()
        captured = capsys.readouterr()

        assert "🟢" in captured.out
        assert "INFO" in captured.out or "LOW" in captured.out

    def test_print_results_no_issues_prints_ok(
        self, auditor: SecurityAuditor, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """No issues prints success message."""
        auditor.print_results()
        captured = capsys.readouterr()

        assert "✅" in captured.out
        assert "No security issues" in captured.out

    def test_print_results_multiple_issues_formatting(
        self, auditor: SecurityAuditor, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Multiple issues formatted correctly with bullets."""
        auditor.critical.append("issue1")
        auditor.critical.append("issue2")

        auditor.print_results()
        captured = capsys.readouterr()

        assert captured.out.count("- issue") == 2


# =============================================================================
# TestGetSummary
# =============================================================================


class TestGetSummary:
    """Tests for SecurityAuditor.get_summary() method."""

    def test_get_summary_empty_lists(self, auditor: SecurityAuditor) -> None:
        """Empty lists return all zeros."""
        summary = auditor.get_summary()

        assert summary == {"critical": 0, "high": 0, "medium": 0, "low": 0}

    def test_get_summary_with_issues(self, auditor: SecurityAuditor) -> None:
        """Summary counts issues correctly."""
        auditor.critical.extend(["c1", "c2"])
        auditor.high.append("h1")
        auditor.medium.extend(["m1", "m2", "m3"])
        auditor.low.append("l1")

        summary = auditor.get_summary()

        assert summary["critical"] == 2
        assert summary["high"] == 1
        assert summary["medium"] == 3
        assert summary["low"] == 1

    def test_get_summary_doesnt_clear_lists(self, auditor: SecurityAuditor) -> None:
        """get_summary doesn't modify lists (unlike print_results)."""
        auditor.critical.append("issue")
        original_critical = auditor.critical.copy()

        auditor.get_summary()

        assert auditor.critical == original_critical


# =============================================================================
# TestMain
# =============================================================================


class TestMain:
    """Tests for main() function integration."""

    def test_main_single_file_no_issues_exit_0(
        self, tmp_workflow: Path, minimal_workflow: dict[str, Any]
    ) -> None:
        """Main exits 0 when no issues found with default fail-on=critical."""
        write_workflow_yaml(tmp_workflow, minimal_workflow)

        with patch.object(sys, "argv", ["security_audit.py", str(tmp_workflow)]):
            with pytest.raises(SystemExit) as exc_info:
                main()

        assert exc_info.value.code == 0

    def test_main_single_file_with_critical_exit_1(
        self, tmp_workflow: Path
    ) -> None:
        """Main exits 1 when critical issue found."""
        workflow = {"permissions": "write-all"}
        write_workflow_yaml(tmp_workflow, workflow)

        with patch.object(sys, "argv", ["security_audit.py", str(tmp_workflow)]):
            with pytest.raises(SystemExit) as exc_info:
                main()

        assert exc_info.value.code == 1

    def test_main_multiple_files_aggregates_issues(
        self, tmp_path: Path
    ) -> None:
        """Multiple files aggregate issue counts."""
        file1 = tmp_path / "workflow1.yml"
        file2 = tmp_path / "workflow2.yml"

        workflow_with_issue = {"permissions": "write-all"}
        write_workflow_yaml(file1, workflow_with_issue)
        write_workflow_yaml(file2, workflow_with_issue)

        with patch.object(
            sys, "argv", ["security_audit.py", str(file1), str(file2)]
        ):
            with pytest.raises(SystemExit) as exc_info:
                main()

        assert exc_info.value.code == 1

    def test_main_fail_on_high(self, tmp_workflow: Path) -> None:
        """--fail-on=high fails on high severity issues."""
        workflow = {
            "on": {"pull_request_target": None},
            "jobs": {"build": {"runs-on": "ubuntu-latest"}},
        }
        write_workflow_yaml(tmp_workflow, workflow)

        with patch.object(
            sys, "argv", ["security_audit.py", str(tmp_workflow), "--fail-on=high"]
        ):
            with pytest.raises(SystemExit) as exc_info:
                main()

        assert exc_info.value.code == 1

    def test_main_fail_on_medium(self, tmp_workflow: Path) -> None:
        """--fail-on=medium fails on medium severity issues."""
        workflow = {
            "on": {"workflow_run": None},
            "jobs": {"build": {"runs-on": "ubuntu-latest"}},
        }
        write_workflow_yaml(tmp_workflow, workflow)

        with patch.object(
            sys, "argv", ["security_audit.py", str(tmp_workflow), "--fail-on=medium"]
        ):
            with pytest.raises(SystemExit) as exc_info:
                main()

        assert exc_info.value.code == 1

    def test_main_fail_on_low(self, tmp_workflow: Path) -> None:
        """--fail-on=low fails on low severity issues."""
        workflow = {
            "jobs": {
                "build": {
                    "runs-on": "ubuntu-latest",
                    "steps": [{"uses": "org/action@v1"}],
                }
            }
        }
        write_workflow_yaml(tmp_workflow, workflow)

        with patch.object(
            sys, "argv", ["security_audit.py", str(tmp_workflow), "--fail-on=low"]
        ):
            with pytest.raises(SystemExit) as exc_info:
                main()

        assert exc_info.value.code == 1

    def test_main_fail_on_critical_passes_on_high(
        self, tmp_workflow: Path
    ) -> None:
        """--fail-on=critical passes when only high severity found."""
        workflow = {
            "on": {"pull_request_target": None},
            "jobs": {"build": {"runs-on": "ubuntu-latest"}},
        }
        write_workflow_yaml(tmp_workflow, workflow)

        with patch.object(
            sys, "argv", ["security_audit.py", str(tmp_workflow), "--fail-on=critical"]
        ):
            with pytest.raises(SystemExit) as exc_info:
                main()

        assert exc_info.value.code == 0

    def test_main_prints_summary_header(
        self, tmp_workflow: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Main prints summary header."""
        write_workflow_yaml(tmp_workflow, {"permissions": "write-all"})

        with patch.object(sys, "argv", ["security_audit.py", str(tmp_workflow)]):
            with pytest.raises(SystemExit):
                main()

        captured = capsys.readouterr()
        assert "SECURITY AUDIT SUMMARY" in captured.out

    def test_main_prints_issue_counts(
        self, tmp_workflow: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Main prints counts for each severity level."""
        write_workflow_yaml(tmp_workflow, {"permissions": "write-all"})

        with patch.object(sys, "argv", ["security_audit.py", str(tmp_workflow)]):
            with pytest.raises(SystemExit):
                main()

        captured = capsys.readouterr()
        assert "Critical:" in captured.out
        assert "High:" in captured.out
        assert "Medium:" in captured.out
        assert "Low:" in captured.out

    def test_main_prints_threshold_message_on_failure(
        self, tmp_workflow: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Main prints threshold message when failing."""
        write_workflow_yaml(tmp_workflow, {"permissions": "write-all"})

        with patch.object(sys, "argv", ["security_audit.py", str(tmp_workflow)]):
            with pytest.raises(SystemExit):
                main()

        captured = capsys.readouterr()
        assert "failed" in captured.out.lower()

    def test_main_prints_success_message_on_pass(
        self, tmp_workflow: Path, minimal_workflow: dict[str, Any], capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Main prints success message when passing."""
        write_workflow_yaml(tmp_workflow, minimal_workflow)

        with patch.object(sys, "argv", ["security_audit.py", str(tmp_workflow)]):
            with pytest.raises(SystemExit):
                main()

        captured = capsys.readouterr()
        assert "✅" in captured.out or "passed" in captured.out.lower()

    def test_main_no_files_exits_with_error(self) -> None:
        """Main with no file arguments exits with error."""
        with patch.object(sys, "argv", ["security_audit.py"]):
            with pytest.raises(SystemExit) as exc_info:
                main()

        # argparse exits with code 2 for missing required arguments
        assert exc_info.value.code == 2

    def test_main_invalid_fail_on_value_exits_with_error(
        self, tmp_workflow: Path, minimal_workflow: dict[str, Any]
    ) -> None:
        """Main with invalid --fail-on value exits with error."""
        write_workflow_yaml(tmp_workflow, minimal_workflow)

        with patch.object(
            sys, "argv", ["security_audit.py", str(tmp_workflow), "--fail-on=invalid"]
        ):
            with pytest.raises(SystemExit) as exc_info:
                main()

        # argparse exits with code 2 for invalid choice
        assert exc_info.value.code == 2


# =============================================================================
# TestIntegration
# =============================================================================


class TestIntegration:
    """Integration tests combining multiple security checks."""

    def test_comprehensive_insecure_workflow(
        self, auditor: SecurityAuditor, tmp_workflow: Path
    ) -> None:
        """Comprehensive workflow with multiple security issues."""
        workflow = {
            "name": "Insecure Workflow",
            "on": {"pull_request_target": None},
            "permissions": "write-all",
            "jobs": {
                "build": {
                    "permissions": "write-all",
                    "runs-on": "self-hosted",
                    "steps": [
                        {
                            "uses": "some-action",  # Unpinned
                            "run": "echo ${{ github.event.issue.title }}",
                        },
                        {
                            "run": "password = 'secret123'",
                        },
                    ],
                }
            },
        }
        content = yaml.dump(workflow)
        write_workflow_yaml(tmp_workflow, workflow)

        summary = auditor.audit_file(tmp_workflow)

        # Should detect multiple issues
        assert summary["critical"] > 0
        assert summary["high"] > 0

    def test_secure_workflow_minimal_issues(
        self, auditor: SecurityAuditor, tmp_workflow: Path
    ) -> None:
        """Secure workflow has minimal to no issues."""
        workflow = {
            "name": "Secure Workflow",
            "on": "push",
            "permissions": {"contents": "read"},
            "jobs": {
                "build": {
                    "runs-on": "ubuntu-latest",
                    "permissions": {"contents": "read"},
                    "steps": [
                        {
                            "uses": "actions/checkout@v3",
                            "with": {"fetch-depth": 0},
                        },
                        {
                            "uses": "actions/setup-python@abc123def456",
                            "with": {"python-version": "3.9"},
                        },
                        {
                            "name": "Run tests",
                            "run": "python -m pytest",
                        },
                    ],
                }
            },
        }
        write_workflow_yaml(tmp_workflow, workflow)

        summary = auditor.audit_file(tmp_workflow)

        # Should have few or no critical issues
        assert summary["critical"] == 0

    def test_all_dangerous_contexts_detected(
        self, auditor: SecurityAuditor, tmp_workflow: Path
    ) -> None:
        """All 9 dangerous contexts are detected."""
        dangerous_contexts = [
            "github.event.issue.title",
            "github.event.issue.body",
            "github.event.pull_request.title",
            "github.event.pull_request.body",
            "github.event.comment.body",
            "github.event.review.body",
            "github.event.discussion.title",
            "github.event.discussion.body",
            "github.head_ref",
        ]

        for context in dangerous_contexts:
            auditor = SecurityAuditor()
            workflow = {
                "jobs": {
                    "build": {
                        "steps": [
                            {"run": f"echo ${{{{{context}}}}}"},
                        ]
                    }
                }
            }
            write_workflow_yaml(tmp_workflow, workflow)
            summary = auditor.audit_file(tmp_workflow)

            assert summary["critical"] == 1

    def test_all_secret_patterns_detected(
        self, auditor: SecurityAuditor, tmp_workflow: Path
    ) -> None:
        """All secret patterns are detected."""
        secrets = [
            ('password = "secret"', "password"),
            ('api_key = "key123456"', "API key"),
            ('token = "token123456"', "token"),
            ('ghp_' + 'a' * 36, "GitHub PAT"),
            ('AKIA' + '0' * 16, "AWS key"),
            ('secret = "value"', "secret"),
        ]

        for content, pattern_type in secrets:
            auditor = SecurityAuditor()
            workflow = {}
            write_workflow_yaml(tmp_workflow, workflow)

            auditor.check_secrets_exposure(workflow, content)
            assert len(auditor.critical) > 0

    @pytest.mark.parametrize(
        "perm_type,level",
        [
            ("write-all", "critical"),
            ({"contents": "write"}, "medium"),
            ({"packages": "write"}, "medium"),
        ],
    )
    def test_all_permission_levels(
        self, auditor: SecurityAuditor, perm_type: str | dict[str, str], level: str
    ) -> None:
        """All permission issue levels are correctly assigned."""
        workflow = {"permissions": perm_type}
        auditor.check_permissions(workflow)

        if level == "critical":
            assert len(auditor.critical) > 0
        elif level == "medium":
            assert len(auditor.medium) > 0


# =============================================================================
# Main Entry Point
# =============================================================================


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
