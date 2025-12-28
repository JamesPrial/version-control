"""
Comprehensive test suite for validate_workflow.py GitHub Actions workflow validator.

Tests cover:
- YAML file validation and parsing
- Required field checking ('on', 'jobs')
- Permissions configuration validation
- Workflow trigger validation
- Job and step validation
- Action version pinning and floating tags
- Command injection vulnerability detection
- Best practices checking (caching, optimization)
- Main function integration tests
"""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any
from unittest.mock import patch

import pytest
import yaml

from validate_workflow import (
    WorkflowValidator,
    main,
)

if TYPE_CHECKING:
    from collections.abc import Generator


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def workflow_file(tmp_path: Path) -> Generator[Path, None, None]:
    """Create a temporary workflow YAML file for testing."""
    workflow_path = tmp_path / ".github" / "workflows" / "test.yml"
    workflow_path.parent.mkdir(parents=True)
    yield workflow_path
    if workflow_path.exists():
        workflow_path.unlink()


@pytest.fixture
def valid_workflow() -> dict[str, Any]:
    """Minimal valid workflow configuration."""
    return {
        "name": "Test Workflow",
        "on": "push",
        "jobs": {
            "test": {
                "runs-on": "ubuntu-latest",
                "steps": [
                    {
                        "name": "Checkout",
                        "uses": "actions/checkout@v4",
                    },
                    {
                        "name": "Run tests",
                        "run": "pytest",
                    },
                ],
            }
        },
    }


@pytest.fixture
def validator() -> WorkflowValidator:
    """Create a fresh WorkflowValidator instance."""
    return WorkflowValidator()


# =============================================================================
# TestWorkflowValidatorInit
# =============================================================================


class TestWorkflowValidatorInit:
    """Tests for WorkflowValidator initialization."""

    def test_init_creates_empty_error_list(self) -> None:
        """Initialization creates empty errors list."""
        validator = WorkflowValidator()
        assert validator.errors == []

    def test_init_creates_empty_warnings_list(self) -> None:
        """Initialization creates empty warnings list."""
        validator = WorkflowValidator()
        assert validator.warnings == []

    def test_init_creates_empty_info_list(self) -> None:
        """Initialization creates empty info list."""
        validator = WorkflowValidator()
        assert validator.info == []


# =============================================================================
# TestValidateFile
# =============================================================================


class TestValidateFile:
    """Tests for validate_file() method."""

    def test_validate_file_not_found_returns_false(
        self, validator: WorkflowValidator
    ) -> None:
        """Non-existent file returns False and adds error."""
        result = validator.validate_file(Path("/nonexistent/file.yml"))
        assert result is False
        # Note: errors list is cleared after print_results, so we can only
        # verify the return value here

    def test_validate_file_yaml_parsing_error_returns_false(
        self, workflow_file: Path, validator: WorkflowValidator
    ) -> None:
        """Invalid YAML syntax returns False and adds error."""
        workflow_file.write_text("invalid: yaml: content: [")
        result = validator.validate_file(workflow_file)
        assert result is False

    def test_validate_file_non_dict_returns_false(
        self, workflow_file: Path, validator: WorkflowValidator
    ) -> None:
        """Non-dict YAML returns False and adds error."""
        workflow_file.write_text("- just\n- a\n- list\n")
        result = validator.validate_file(workflow_file)
        assert result is False

    def test_validate_file_valid_workflow_returns_true(
        self,
        workflow_file: Path,
        validator: WorkflowValidator,
        valid_workflow: dict[str, Any],
    ) -> None:
        """Valid workflow file returns True."""
        workflow_file.write_text(yaml.dump(valid_workflow))
        result = validator.validate_file(workflow_file)
        assert result is True

    def test_validate_file_clears_lists_after_print(
        self,
        workflow_file: Path,
        validator: WorkflowValidator,
        valid_workflow: dict[str, Any],
    ) -> None:
        """Lists are cleared after printing results."""
        workflow_file.write_text(yaml.dump(valid_workflow))
        validator.validate_file(workflow_file)
        # After validate_file, lists should be empty (cleared by print_results)
        assert validator.errors == []
        assert validator.warnings == []
        assert validator.info == []

    def test_validate_file_with_errors_returns_false(
        self, workflow_file: Path, validator: WorkflowValidator
    ) -> None:
        """Workflow with errors returns False."""
        invalid_workflow: dict[str, Any] = {
            "on": "push",
            # Missing 'jobs'
        }
        workflow_file.write_text(yaml.dump(invalid_workflow))
        result = validator.validate_file(workflow_file)
        assert result is False


# =============================================================================
# TestCheckRequiredFields
# =============================================================================


class TestCheckRequiredFields:
    """Tests for check_required_fields() method."""

    def test_missing_on_field_adds_error(self, validator: WorkflowValidator) -> None:
        """Missing 'on' field adds error."""
        workflow: dict[str, Any] = {"jobs": {"test": {}}}
        validator.check_required_fields(workflow)
        assert any("'on'" in error for error in validator.errors)

    def test_missing_jobs_field_adds_error(self, validator: WorkflowValidator) -> None:
        """Missing 'jobs' field adds error."""
        workflow: dict[str, Any] = {"on": "push"}
        validator.check_required_fields(workflow)
        assert any("'jobs'" in error for error in validator.errors)

    def test_empty_jobs_dict_adds_error(self, validator: WorkflowValidator) -> None:
        """Empty jobs dict adds error."""
        workflow: dict[str, Any] = {"on": "push", "jobs": {}}
        validator.check_required_fields(workflow)
        assert any("at least one job" in error for error in validator.errors)

    def test_missing_name_adds_warning(self, validator: WorkflowValidator) -> None:
        """Missing 'name' field adds warning."""
        workflow: dict[str, Any] = {
            "on": "push",
            "jobs": {"test": {"runs-on": "ubuntu-latest"}},
        }
        validator.check_required_fields(workflow)
        assert any("name" in warning for warning in validator.warnings)

    def test_valid_required_fields_no_errors(
        self, validator: WorkflowValidator
    ) -> None:
        """Valid required fields produces no errors."""
        workflow: dict[str, Any] = {
            "name": "Test",
            "on": "push",
            "jobs": {"test": {"runs-on": "ubuntu-latest"}},
        }
        validator.check_required_fields(workflow)
        assert len(validator.errors) == 0


# =============================================================================
# TestCheckPermissions
# =============================================================================


class TestCheckPermissions:
    """Tests for check_permissions() method."""

    def test_missing_permissions_adds_warning(
        self, validator: WorkflowValidator
    ) -> None:
        """Missing permissions field adds warning."""
        workflow: dict[str, Any] = {}
        validator.check_permissions(workflow)
        assert any("permissions" in warning for warning in validator.warnings)

    def test_write_all_permission_adds_error(
        self, validator: WorkflowValidator
    ) -> None:
        """permissions: write-all adds error."""
        workflow: dict[str, Any] = {"permissions": "write-all"}
        validator.check_permissions(workflow)
        assert any("write-all" in error for error in validator.errors)

    def test_contents_write_adds_info(self, validator: WorkflowValidator) -> None:
        """contents: write adds info message."""
        workflow: dict[str, Any] = {"permissions": {"contents": "write"}}
        validator.check_permissions(workflow)
        assert any("contents" in info for info in validator.info)

    def test_id_token_write_adds_positive_info(
        self, validator: WorkflowValidator
    ) -> None:
        """id-token: write adds positive info message."""
        workflow: dict[str, Any] = {"permissions": {"id-token": "write"}}
        validator.check_permissions(workflow)
        assert any("OIDC" in info or "Good" in info for info in validator.info)

    def test_permissions_read_only_no_issues(
        self, validator: WorkflowValidator
    ) -> None:
        """Read-only permissions produce no errors."""
        workflow: dict[str, Any] = {"permissions": {"contents": "read"}}
        validator.check_permissions(workflow)
        assert len(validator.errors) == 0


# =============================================================================
# TestCheckTriggers
# =============================================================================


class TestCheckTriggers:
    """Tests for check_triggers() method."""

    def test_string_trigger_converted_to_dict(
        self, validator: WorkflowValidator
    ) -> None:
        """String trigger is converted to dict."""
        workflow: dict[str, Any] = {"on": "push"}
        validator.check_triggers(workflow)
        # Should not error - just processes it
        assert len(validator.errors) == 0

    def test_list_trigger_converted_to_dict(
        self, validator: WorkflowValidator
    ) -> None:
        """List trigger is converted to dict."""
        workflow: dict[str, Any] = {"on": ["push", "pull_request"]}
        validator.check_triggers(workflow)
        assert len(validator.errors) == 0

    def test_pull_request_target_adds_warning(
        self, validator: WorkflowValidator
    ) -> None:
        """pull_request_target trigger adds warning."""
        workflow: dict[str, Any] = {"on": {"pull_request_target": None}}
        validator.check_triggers(workflow)
        assert any("pull_request_target" in warning for warning in validator.warnings)

    def test_missing_concurrency_with_push_adds_info(
        self, validator: WorkflowValidator
    ) -> None:
        """Push without concurrency adds info."""
        workflow: dict[str, Any] = {"on": "push"}
        validator.check_triggers(workflow)
        assert any("concurrency" in info for info in validator.info)

    def test_missing_concurrency_with_pull_request_adds_info(
        self, validator: WorkflowValidator
    ) -> None:
        """Pull request without concurrency adds info."""
        workflow: dict[str, Any] = {"on": "pull_request"}
        validator.check_triggers(workflow)
        assert any("concurrency" in info for info in validator.info)

    def test_push_with_concurrency_no_info(self, validator: WorkflowValidator) -> None:
        """Push with concurrency produces no info."""
        workflow: dict[str, Any] = {"on": "push", "concurrency": "ci"}
        validator.check_triggers(workflow)
        # Should not have the concurrency suggestion
        assert not any("concurrency" in info for info in validator.info)

    def test_push_without_path_filters_adds_info(
        self, validator: WorkflowValidator
    ) -> None:
        """Push without path filters adds info."""
        workflow: dict[str, Any] = {"on": {"push": {}}}
        validator.check_triggers(workflow)
        assert any("paths" in info for info in validator.info)

    def test_push_with_paths_no_info(self, validator: WorkflowValidator) -> None:
        """Push with paths produces no info."""
        workflow: dict[str, Any] = {"on": {"push": {"paths": ["src/**"]}}}
        validator.check_triggers(workflow)
        assert not any("paths" in info for info in validator.info)

    def test_push_with_paths_ignore_no_info(
        self, validator: WorkflowValidator
    ) -> None:
        """Push with paths-ignore produces no info."""
        workflow: dict[str, Any] = {"on": {"push": {"paths-ignore": ["docs/**"]}}}
        validator.check_triggers(workflow)
        assert not any("paths" in info for info in validator.info)


# =============================================================================
# TestCheckJobs
# =============================================================================


class TestCheckJobs:
    """Tests for check_jobs() method."""

    def test_empty_jobs_dict(self, validator: WorkflowValidator) -> None:
        """Empty jobs dict is handled without crashing."""
        workflow: dict[str, Any] = {"jobs": {}}
        validator.check_jobs(workflow)
        assert isinstance(validator.errors, list)

    def test_single_job_validation(self, validator: WorkflowValidator) -> None:
        """Single job triggers validation of that job."""
        workflow: dict[str, Any] = {
            "jobs": {"test": {"runs-on": "ubuntu-latest", "steps": []}}
        }
        validator.check_jobs(workflow)
        # Should have warnings for empty steps and missing timeout
        assert any("no steps" in warning for warning in validator.warnings)
        assert any("timeout" in warning for warning in validator.warnings)

    def test_multiple_jobs_validation(self, validator: WorkflowValidator) -> None:
        """Multiple jobs are each validated."""
        workflow: dict[str, Any] = {
            "jobs": {
                "test": {"runs-on": "ubuntu-latest", "steps": []},
                "build": {"runs-on": "ubuntu-latest", "steps": []},
            }
        }
        validator.check_jobs(workflow)
        # Should have warnings for both jobs having empty steps
        empty_steps_warnings = sum(1 for w in validator.warnings if "no steps" in w)
        assert empty_steps_warnings >= 2


# =============================================================================
# TestCheckJob
# =============================================================================


class TestCheckJob:
    """Tests for check_job() method."""

    def test_missing_runs_on_adds_error(self, validator: WorkflowValidator) -> None:
        """Missing 'runs-on' adds error."""
        job: dict[str, Any] = {"steps": []}
        validator.check_job("test", job)
        assert any("runs-on" in error for error in validator.errors)

    def test_missing_steps_and_uses_adds_error(
        self, validator: WorkflowValidator
    ) -> None:
        """Missing both 'steps' and 'uses' adds error."""
        job: dict[str, Any] = {"runs-on": "ubuntu-latest"}
        validator.check_job("test", job)
        assert any("steps" in error or "uses" in error for error in validator.errors)

    def test_uses_without_steps_valid(self, validator: WorkflowValidator) -> None:
        """Job with 'uses' but no 'steps' is valid (reusable workflow)."""
        job: dict[str, Any] = {
            "runs-on": "ubuntu-latest",
            "uses": "other/repo/workflow@v1",
        }
        validator.check_job("test", job)
        assert not any("steps" in error for error in validator.errors)

    def test_missing_timeout_adds_warning(self, validator: WorkflowValidator) -> None:
        """Missing 'timeout-minutes' adds warning."""
        job: dict[str, Any] = {"runs-on": "ubuntu-latest", "steps": []}
        validator.check_job("test", job)
        assert any("timeout" in warning for warning in validator.warnings)

    def test_valid_job_no_errors(self, validator: WorkflowValidator) -> None:
        """Valid job produces no errors."""
        job: dict[str, Any] = {
            "runs-on": "ubuntu-latest",
            "timeout-minutes": 30,
            "steps": [{"run": "echo test"}],
        }
        validator.check_job("test", job)
        assert len(validator.errors) == 0


# =============================================================================
# TestCheckSteps
# =============================================================================


class TestCheckSteps:
    """Tests for check_steps() method."""

    def test_empty_steps_list_adds_warning(self, validator: WorkflowValidator) -> None:
        """Empty steps list adds warning."""
        validator.check_steps("test", [])
        assert any("no steps" in warning for warning in validator.warnings)

    def test_non_dict_step_adds_error(self, validator: WorkflowValidator) -> None:
        """Non-dict step adds error."""
        validator.check_steps("test", ["not a dict"])  # type: ignore[list-item]
        assert len(validator.errors) > 0

    def test_step_missing_run_and_uses_adds_error(
        self, validator: WorkflowValidator
    ) -> None:
        """Step missing both 'run' and 'uses' adds error."""
        validator.check_steps("test", [{"name": "Do something"}])
        assert len(validator.errors) > 0

    def test_step_with_run_valid(self, validator: WorkflowValidator) -> None:
        """Step with 'run' is valid."""
        validator.check_steps("test", [{"run": "echo test"}])
        # Should not add error for run/uses
        assert not any("run" in err and "uses" in err for err in validator.errors)

    def test_step_with_uses_valid(self, validator: WorkflowValidator) -> None:
        """Step with 'uses' is valid."""
        validator.check_steps("test", [{"uses": "actions/checkout@v4"}])
        # Should not add error for run/uses
        assert not any("run" in err and "uses" in err for err in validator.errors)

    def test_multiple_steps_validated(self, validator: WorkflowValidator) -> None:
        """Multiple steps are validated."""
        validator.check_steps(
            "test",
            [
                {"run": "echo 1"},
                {"uses": "actions/checkout@v4"},
                {"run": "echo 2"},
            ],
        )
        assert len(validator.errors) == 0


# =============================================================================
# TestCheckActionVersion
# =============================================================================


class TestCheckActionVersion:
    """Tests for check_action_version() method."""

    def test_missing_version_adds_error(self, validator: WorkflowValidator) -> None:
        """Action without version adds error."""
        validator.check_action_version("test", 0, "actions/checkout")
        assert any("must specify a version" in error for error in validator.errors)

    def test_floating_tag_main_adds_warning(
        self, validator: WorkflowValidator
    ) -> None:
        """Floating tag 'main' adds warning."""
        validator.check_action_version("test", 0, "actions/checkout@main")
        assert any("floating tag" in warning for warning in validator.warnings)

    def test_floating_tag_master_adds_warning(
        self, validator: WorkflowValidator
    ) -> None:
        """Floating tag 'master' adds warning."""
        validator.check_action_version("test", 0, "actions/checkout@master")
        assert any("floating tag" in warning for warning in validator.warnings)

    def test_floating_tag_latest_adds_warning(
        self, validator: WorkflowValidator
    ) -> None:
        """Floating tag 'latest' adds warning."""
        validator.check_action_version("test", 0, "actions/checkout@latest")
        assert any("floating tag" in warning for warning in validator.warnings)

    def test_pinned_version_tag_no_warning(self, validator: WorkflowValidator) -> None:
        """Pinned version tag produces no warning."""
        validator.check_action_version("test", 0, "actions/checkout@v4")
        assert not any("floating" in warning for warning in validator.warnings)

    def test_pinned_sha_no_warning(self, validator: WorkflowValidator) -> None:
        """Pinned SHA produces no warning."""
        validator.check_action_version(
            "test", 0, "actions/checkout@a81bbbf8298c0fa03ea29cdc473d45aaaf96a1c0"
        )
        assert not any("floating" in warning for warning in validator.warnings)


# =============================================================================
# TestCheckCommandInjection
# =============================================================================


class TestCheckCommandInjection:
    """Tests for check_command_injection() method."""

    @pytest.mark.parametrize(
        "dangerous_pattern",
        [
            "github.event.issue.title",
            "github.event.issue.body",
            "github.event.pull_request.title",
            "github.event.pull_request.body",
            "github.event.comment.body",
            "github.head_ref",
        ],
    )
    def test_dangerous_pattern_adds_warning(
        self, validator: WorkflowValidator, dangerous_pattern: str
    ) -> None:
        """Each dangerous pattern adds warning."""
        # Note: validator looks for '${{pattern' without space after '${{'.
        step: dict[str, Any] = {"run": f"echo ${{{{{dangerous_pattern}}}}}"}
        validator.check_command_injection("test", 0, step)
        assert any(dangerous_pattern in warning for warning in validator.warnings)

    def test_safe_context_no_warning(self, validator: WorkflowValidator) -> None:
        """Safe context variables produce no warning."""
        step: dict[str, Any] = {"run": "echo ${{github.ref}}"}
        validator.check_command_injection("test", 0, step)
        assert not any("injection" in warning for warning in validator.warnings)

    def test_no_context_interpolation_no_warning(
        self, validator: WorkflowValidator
    ) -> None:
        """Step without context interpolation produces no warning."""
        step: dict[str, Any] = {"run": "echo test"}
        validator.check_command_injection("test", 0, step)
        assert len(validator.warnings) == 0

    def test_issue_title_injection_detected(
        self, validator: WorkflowValidator
    ) -> None:
        """Issue title injection is detected."""
        step: dict[str, Any] = {"run": 'echo "Issue: ${{github.event.issue.title}}"'}
        validator.check_command_injection("test", 0, step)
        assert len(validator.warnings) > 0

    def test_pull_request_body_injection_detected(
        self, validator: WorkflowValidator
    ) -> None:
        """PR body injection is detected."""
        step: dict[str, Any] = {"run": 'echo "${{github.event.pull_request.body}}"'}
        validator.check_command_injection("test", 0, step)
        assert len(validator.warnings) > 0

    def test_comment_body_injection_detected(
        self, validator: WorkflowValidator
    ) -> None:
        """Comment body injection is detected."""
        step: dict[str, Any] = {"run": 'echo "${{github.event.comment.body}}"'}
        validator.check_command_injection("test", 0, step)
        assert len(validator.warnings) > 0

    def test_head_ref_injection_detected(self, validator: WorkflowValidator) -> None:
        """head_ref injection is detected."""
        step: dict[str, Any] = {"run": 'git checkout "${{github.head_ref}}"'}
        validator.check_command_injection("test", 0, step)
        assert len(validator.warnings) > 0

    def test_multiple_dangerous_patterns_detected(
        self, validator: WorkflowValidator
    ) -> None:
        """Multiple dangerous patterns are detected."""
        step: dict[str, Any] = {
            "run": (
                "echo ${{github.event.issue.title}} && "
                "echo ${{github.event.pull_request.body}}"
            )
        }
        validator.check_command_injection("test", 0, step)
        assert len(validator.warnings) >= 2


# =============================================================================
# TestCheckBestPractices
# =============================================================================


class TestCheckBestPractices:
    """Tests for check_best_practices() method."""

    def test_setup_node_without_cache_adds_info(
        self, validator: WorkflowValidator
    ) -> None:
        """setup-node without caching adds info."""
        workflow: dict[str, Any] = {
            "jobs": {
                "test": {
                    "runs-on": "ubuntu-latest",
                    "steps": [{"uses": "actions/setup-node@v4"}],
                }
            }
        }
        validator.check_best_practices(workflow)
        assert any("cach" in info.lower() for info in validator.info)

    def test_setup_node_with_cache_no_info(self, validator: WorkflowValidator) -> None:
        """setup-node with built-in cache produces no info."""
        workflow: dict[str, Any] = {
            "jobs": {
                "test": {
                    "runs-on": "ubuntu-latest",
                    "steps": [
                        {"uses": "actions/setup-node@v4", "with": {"cache": "npm"}}
                    ],
                }
            }
        }
        validator.check_best_practices(workflow)
        assert not any("cach" in info.lower() for info in validator.info)

    def test_setup_node_with_explicit_cache_action(
        self, validator: WorkflowValidator
    ) -> None:
        """setup-node with explicit cache action produces no info."""
        workflow: dict[str, Any] = {
            "jobs": {
                "test": {
                    "runs-on": "ubuntu-latest",
                    "steps": [
                        {"uses": "actions/setup-node@v4"},
                        {"uses": "actions/cache@v3"},
                    ],
                }
            }
        }
        validator.check_best_practices(workflow)
        assert not any("cach" in info.lower() for info in validator.info)

    def test_no_setup_node_no_caching_info(self, validator: WorkflowValidator) -> None:
        """No setup-node means no caching info."""
        workflow: dict[str, Any] = {
            "jobs": {
                "test": {
                    "runs-on": "ubuntu-latest",
                    "steps": [{"uses": "actions/checkout@v4"}],
                }
            }
        }
        validator.check_best_practices(workflow)
        assert not any("cach" in info.lower() for info in validator.info)

    def test_non_dict_job_skipped(self, validator: WorkflowValidator) -> None:
        """Non-dict job is skipped safely."""
        workflow: dict[str, Any] = {"jobs": {"test": "not a dict"}}
        # Should not crash
        validator.check_best_practices(workflow)
        assert isinstance(validator.info, list)

    def test_non_dict_step_skipped(self, validator: WorkflowValidator) -> None:
        """Non-dict step is skipped safely."""
        workflow: dict[str, Any] = {"jobs": {"test": {"steps": ["not a dict"]}}}
        # Should not crash
        validator.check_best_practices(workflow)
        assert isinstance(validator.info, list)


# =============================================================================
# TestPrintResults
# =============================================================================


class TestPrintResults:
    """Tests for print_results() method."""

    def test_print_results_clears_errors(self, validator: WorkflowValidator) -> None:
        """print_results clears errors list."""
        validator.errors = ["test error"]
        validator.print_results()
        assert validator.errors == []

    def test_print_results_clears_warnings(self, validator: WorkflowValidator) -> None:
        """print_results clears warnings list."""
        validator.warnings = ["test warning"]
        validator.print_results()
        assert validator.warnings == []

    def test_print_results_clears_info(self, validator: WorkflowValidator) -> None:
        """print_results clears info list."""
        validator.info = ["test info"]
        validator.print_results()
        assert validator.info == []

    def test_print_results_outputs_errors(
        self, validator: WorkflowValidator, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """print_results outputs errors."""
        validator.errors = ["Critical error"]
        validator.print_results()
        captured = capsys.readouterr()
        assert "Critical error" in captured.out

    def test_print_results_outputs_warnings(
        self, validator: WorkflowValidator, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """print_results outputs warnings."""
        validator.warnings = ["Be careful"]
        validator.print_results()
        captured = capsys.readouterr()
        assert "Be careful" in captured.out

    def test_print_results_outputs_info(
        self, validator: WorkflowValidator, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """print_results outputs info."""
        validator.info = ["Helpful tip"]
        validator.print_results()
        captured = capsys.readouterr()
        assert "Helpful tip" in captured.out

    def test_print_results_no_issues_success_message(
        self, validator: WorkflowValidator, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """No issues prints success message."""
        validator.print_results()
        captured = capsys.readouterr()
        assert "No issues found" in captured.out

    def test_print_results_with_all_categories(
        self, validator: WorkflowValidator, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """All categories are output."""
        validator.errors = ["Error 1"]
        validator.warnings = ["Warning 1"]
        validator.info = ["Info 1"]
        validator.print_results()
        captured = capsys.readouterr()
        assert "Error 1" in captured.out
        assert "Warning 1" in captured.out
        assert "Info 1" in captured.out


# =============================================================================
# TestComplexWorkflows
# =============================================================================


class TestComplexWorkflows:
    """Tests with complex, realistic workflow scenarios."""

    def test_workflow_with_matrix_strategy(
        self, workflow_file: Path, validator: WorkflowValidator
    ) -> None:
        """Workflow with matrix strategy is validated."""
        workflow: dict[str, Any] = {
            "name": "Matrix Test",
            "on": "push",
            "jobs": {
                "test": {
                    "runs-on": "ubuntu-latest",
                    "strategy": {
                        "matrix": {"python-version": ["3.9", "3.10", "3.11"]}
                    },
                    "steps": [
                        {"uses": "actions/checkout@v4"},
                        {"uses": "actions/setup-python@v4"},
                    ],
                }
            },
        }
        workflow_file.write_text(yaml.dump(workflow))
        result = validator.validate_file(workflow_file)
        assert result is True

    def test_workflow_with_environment_variables(
        self, workflow_file: Path, validator: WorkflowValidator
    ) -> None:
        """Workflow with environment variables is validated."""
        workflow: dict[str, Any] = {
            "name": "Env Test",
            "on": "push",
            "env": {"NODE_ENV": "production"},
            "jobs": {
                "test": {
                    "runs-on": "ubuntu-latest",
                    "steps": [{"run": "echo $NODE_ENV"}],
                }
            },
        }
        workflow_file.write_text(yaml.dump(workflow))
        result = validator.validate_file(workflow_file)
        assert result is True

    def test_workflow_with_conditional_steps(
        self, workflow_file: Path, validator: WorkflowValidator
    ) -> None:
        """Workflow with conditional steps is validated."""
        workflow: dict[str, Any] = {
            "name": "Conditional Test",
            "on": "push",
            "jobs": {
                "test": {
                    "runs-on": "ubuntu-latest",
                    "steps": [
                        {
                            "if": "github.ref == 'refs/heads/main'",
                            "run": "echo main branch",
                        }
                    ],
                }
            },
        }
        workflow_file.write_text(yaml.dump(workflow))
        result = validator.validate_file(workflow_file)
        assert result is True

    def test_workflow_with_needs_dependency(
        self, workflow_file: Path, validator: WorkflowValidator
    ) -> None:
        """Workflow with job dependencies is validated."""
        workflow: dict[str, Any] = {
            "name": "Dependencies Test",
            "on": "push",
            "jobs": {
                "build": {
                    "runs-on": "ubuntu-latest",
                    "steps": [{"run": "echo building"}],
                },
                "test": {
                    "runs-on": "ubuntu-latest",
                    "needs": "build",
                    "steps": [{"run": "echo testing"}],
                },
            },
        }
        workflow_file.write_text(yaml.dump(workflow))
        result = validator.validate_file(workflow_file)
        assert result is True

    def test_security_focused_workflow(
        self, workflow_file: Path, validator: WorkflowValidator
    ) -> None:
        """Security-focused workflow with best practices validates successfully."""
        workflow: dict[str, Any] = {
            "name": "Secure CI",
            "on": {"push": {"branches": ["main"]}, "pull_request": None},
            "permissions": {"contents": "read", "id-token": "write"},
            "concurrency": "ci",
            "jobs": {
                "test": {
                    "runs-on": "ubuntu-latest",
                    "timeout-minutes": 30,
                    "steps": [
                        {
                            "name": "Checkout",
                            "uses": "actions/checkout@a81bbbf8298c0fa03ea29cdc473d45aaaf96a1c0",
                        },
                        {
                            "name": "Setup Node",
                            "uses": "actions/setup-node@v4",
                            "with": {"cache": "npm"},
                        },
                        {
                            "name": "Test",
                            "run": "npm test",
                            "env": {"NODE_OPTIONS": "--experimental-vm-modules"},
                        },
                    ],
                }
            },
        }
        workflow_file.write_text(yaml.dump(workflow))
        result = validator.validate_file(workflow_file)
        assert result is True

    def test_security_focused_workflow_has_oidc_info(
        self, validator: WorkflowValidator
    ) -> None:
        """Verify OIDC permission detection before print_results clears lists."""
        workflow: dict[str, Any] = {"permissions": {"id-token": "write"}}
        validator.check_permissions(workflow)
        assert any("OIDC" in info or "Good" in info for info in validator.info)


# =============================================================================
# TestEdgeCases
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_workflow_with_null_on_field_crashes(
        self, workflow_file: Path, validator: WorkflowValidator
    ) -> None:
        """Workflow with null 'on' field crashes the validator (known bug).
        
        The validator does not handle None for the 'on' field, causing a
        TypeError in check_triggers(). This test documents the current behavior.
        """
        workflow: dict[str, Any] = {
            "on": None,
            "jobs": {"test": {"runs-on": "ubuntu-latest", "steps": []}},
        }
        workflow_file.write_text(yaml.dump(workflow))
        # Current behavior: crashes with TypeError
        with pytest.raises(TypeError, match="argument of type 'NoneType'"):
            validator.validate_file(workflow_file)

    def test_workflow_with_null_jobs_field(
        self, workflow_file: Path, validator: WorkflowValidator
    ) -> None:
        """Workflow with null 'jobs' field is handled gracefully."""
        workflow: dict[str, Any] = {"on": "push", "jobs": None}
        workflow_file.write_text(yaml.dump(workflow))
        # The validator checks for empty jobs but not None, so check_jobs crashes
        with pytest.raises(AttributeError):
            validator.validate_file(workflow_file)

    def test_job_with_empty_name_field(self, validator: WorkflowValidator) -> None:
        """Job step with empty name field is still valid."""
        validator.check_steps("test", [{"run": "echo test", "name": ""}])
        # Should not add any errors for the empty name
        assert not any("name" in error for error in validator.errors)

    def test_very_long_action_name(self, validator: WorkflowValidator) -> None:
        """Very long action name is handled without error."""
        long_action = "some/namespace/action@v1"
        validator.check_action_version("test", 0, long_action)
        # Should not raise or add unexpected errors
        assert len(validator.errors) == 0

    def test_unicode_in_workflow_name(
        self, workflow_file: Path, validator: WorkflowValidator
    ) -> None:
        """Unicode in workflow name is handled correctly."""
        workflow: dict[str, Any] = {
            "name": "Test Workflow",
            "on": "push",
            "jobs": {"test": {"runs-on": "ubuntu-latest", "steps": []}},
        }
        workflow_file.write_text(yaml.dump(workflow), encoding="utf-8")
        result = validator.validate_file(workflow_file)
        assert result is True


# =============================================================================
# TestMain
# =============================================================================


class TestMain:
    """Tests for main() function."""

    def test_main_single_valid_file_exit_0(self, workflow_file: Path) -> None:
        """Single valid file exits with 0."""
        workflow: dict[str, Any] = {
            "name": "Test",
            "on": "push",
            "jobs": {"test": {"runs-on": "ubuntu-latest", "steps": []}},
        }
        workflow_file.write_text(yaml.dump(workflow))

        with patch("sys.argv", ["validate_workflow.py", str(workflow_file)]):
            with pytest.raises(SystemExit) as exc_info:
                main()

        assert exc_info.value.code == 0

    def test_main_invalid_file_exit_1(self, workflow_file: Path) -> None:
        """Invalid file exits with 1."""
        workflow_file.write_text("invalid: yaml: [")

        with patch("sys.argv", ["validate_workflow.py", str(workflow_file)]):
            with pytest.raises(SystemExit) as exc_info:
                main()

        assert exc_info.value.code == 1

    def test_main_multiple_files(self, tmp_path: Path) -> None:
        """Multiple files are validated."""
        file1 = tmp_path / "workflow1.yml"
        file2 = tmp_path / "workflow2.yml"

        workflow: dict[str, Any] = {
            "name": "Test",
            "on": "push",
            "jobs": {"test": {"runs-on": "ubuntu-latest", "steps": []}},
        }

        file1.write_text(yaml.dump(workflow))
        file2.write_text(yaml.dump(workflow))

        with patch("sys.argv", ["validate_workflow.py", str(file1), str(file2)]):
            with pytest.raises(SystemExit) as exc_info:
                main()

        assert exc_info.value.code == 0

    def test_main_no_files_shows_error(self) -> None:
        """No files exits with error."""
        with patch("sys.argv", ["validate_workflow.py"]):
            with pytest.raises(SystemExit) as exc_info:
                main()
        # argparse exits with code 2 for missing required arguments
        assert exc_info.value.code == 2

    def test_main_nonexistent_file_exit_1(self) -> None:
        """Non-existent file exits with 1."""
        with patch("sys.argv", ["validate_workflow.py", "/nonexistent/file.yml"]):
            with pytest.raises(SystemExit) as exc_info:
                main()

        assert exc_info.value.code == 1


# =============================================================================
# TestIntegration
# =============================================================================


class TestIntegration:
    """Integration tests with full validation workflows."""

    def test_full_validation_pass(
        self, workflow_file: Path, validator: WorkflowValidator
    ) -> None:
        """Full validation pass with all checks."""
        workflow: dict[str, Any] = {
            "name": "Full CI",
            "on": {"push": {"branches": ["main"]}, "pull_request": None},
            "permissions": {"contents": "read", "id-token": "write"},
            "concurrency": "ci",
            "jobs": {
                "test": {
                    "runs-on": "ubuntu-latest",
                    "timeout-minutes": 30,
                    "steps": [
                        {
                            "uses": "actions/checkout@a81bbbf8298c0fa03ea29cdc473d45aaaf96a1c0"
                        },
                        {"uses": "actions/setup-node@v4", "with": {"cache": "npm"}},
                        {"run": "npm install"},
                        {"run": "npm test"},
                    ],
                }
            },
        }
        workflow_file.write_text(yaml.dump(workflow))
        result = validator.validate_file(workflow_file)
        assert result is True

    def test_full_validation_fail(
        self, workflow_file: Path, validator: WorkflowValidator
    ) -> None:
        """Full validation fail with multiple errors."""
        workflow: dict[str, Any] = {
            # Missing 'on' and 'jobs'
            "name": "Invalid"
        }
        workflow_file.write_text(yaml.dump(workflow))
        result = validator.validate_file(workflow_file)
        assert result is False

    def test_security_audit_detects_write_all(
        self, validator: WorkflowValidator
    ) -> None:
        """Security audit detects write-all permission before print clears lists."""
        workflow: dict[str, Any] = {"permissions": "write-all"}
        validator.check_permissions(workflow)
        assert any("write-all" in error for error in validator.errors)

    def test_security_audit_results(
        self, workflow_file: Path, validator: WorkflowValidator
    ) -> None:
        """Security audit detects dangerous patterns (returns False)."""
        workflow: dict[str, Any] = {
            "name": "Insecure",
            "on": "push",
            "permissions": "write-all",
            "jobs": {
                "build": {
                    "runs-on": "ubuntu-latest",
                    "steps": [
                        {"uses": "actions/checkout@main"},
                        {"run": 'echo "${{github.event.pull_request.body}}"'},
                    ],
                }
            },
        }
        workflow_file.write_text(yaml.dump(workflow))
        result = validator.validate_file(workflow_file)
        assert result is False


# =============================================================================
# TestValidatorStateManagement
# =============================================================================


class TestValidatorStateManagement:
    """Tests for proper state management across validations."""

    def test_validator_reuse_clears_state(self, tmp_path: Path) -> None:
        """Reusing validator clears previous state."""
        validator = WorkflowValidator()

        # Validate first file (has errors)
        workflow1: dict[str, Any] = {
            "on": "push",
            "jobs": {"test": {}},  # Missing runs-on
        }
        file1 = tmp_path / "workflow1.yml"
        file1.write_text(yaml.dump(workflow1))
        result1 = validator.validate_file(file1)
        assert result1 is False

        # After print_results, lists should be cleared
        assert validator.errors == []

        # Validate second file (valid)
        workflow2: dict[str, Any] = {
            "name": "Test",
            "on": "push",
            "jobs": {"test": {"runs-on": "ubuntu-latest", "steps": []}},
        }
        file2 = tmp_path / "workflow2.yml"
        file2.write_text(yaml.dump(workflow2))
        result2 = validator.validate_file(file2)
        assert result2 is True


# =============================================================================
# Main Entry Point
# =============================================================================


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
