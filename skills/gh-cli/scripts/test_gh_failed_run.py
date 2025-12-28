"""
Comprehensive test suite for gh_failed_run.py GitHub Actions failed run analyzer.

Tests cover:
- gh CLI command execution and JSON parsing
- Failed run retrieval and filtering
- Failed job extraction and filtering
- Error pattern matching and ANSI code stripping
- Log retrieval with fallback logic
- Full analysis orchestration
- Main function integration tests
- Edge cases and error handling
"""
from __future__ import annotations

import json
import subprocess
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from gh_failed_run import (
    run_gh_command,
    get_most_recent_failed_run,
    get_failed_jobs,
    extract_error_excerpts,
    get_job_logs,
    analyze_failed_run,
    main,
)


# =============================================================================
# Test Data Fixtures
# =============================================================================


@pytest.fixture
def sample_failed_run() -> dict:
    """Sample failed workflow run from gh CLI."""
    return {
        "databaseId": 12345,
        "number": 42,
        "conclusion": "failure",
        "status": "completed",
        "createdAt": "2025-01-15T10:30:00Z",
        "displayTitle": "CI/CD Pipeline",
        "url": "https://github.com/owner/repo/actions/runs/12345",
        "headBranch": "main",
        "headSha": "abc1234567890def",
        "event": "push",
    }


@pytest.fixture
def sample_jobs() -> dict[str, Any]:
    """Sample jobs from a failed run."""
    return {
        "jobs": [
            {
                "databaseId": 1001,
                "name": "Unit Tests",
                "conclusion": "failure",
                "status": "completed",
                "startedAt": "2025-01-15T10:30:10Z",
                "completedAt": "2025-01-15T10:31:00Z",
            },
            {
                "databaseId": 1002,
                "name": "Build",
                "conclusion": "success",
                "status": "completed",
                "startedAt": "2025-01-15T10:31:00Z",
                "completedAt": "2025-01-15T10:35:00Z",
            },
            {
                "databaseId": 1003,
                "name": "Integration Tests",
                "conclusion": "timed_out",
                "status": "completed",
                "startedAt": "2025-01-15T10:35:00Z",
                "completedAt": "2025-01-15T10:40:00Z",
            },
        ]
    }


@pytest.fixture
def sample_job_logs() -> str:
    """Sample logs from a failed job with various error patterns."""
    return (
        "Starting job...\n"
        "2025-01-15T10:30:10.000Z Running tests\n"
        "\x1b[31mERROR: Connection failed\x1b[0m\n"
        "2025-01-15T10:30:15.000Z Test execution error detected\n"
        "Failed to connect to database\n"
        "\x1b[33mWARNING: Exception in test handler\x1b[0m\n"
        "Cannot establish connection\n"
        "2025-01-15T10:30:20.000Z Process completed with exit code 1\n"
        "Timeout waiting for response\n"
        "panic: out of memory\n"
        "Completing job...\n"
    )


@pytest.fixture
def sample_timeout_logs() -> str:
    """Sample logs from a timed-out job."""
    return (
        "Starting long-running test...\n"
        "Still running...\n"
        "Waiting for response...\n"
        "\x1b[31mERROR: Job timeout\x1b[0m\n"
        "The job was cancelled after 360 minutes\n"
    )


@pytest.fixture
def sample_successful_logs() -> str:
    """Sample logs from a successful job."""
    return "All tests passed\nBuild completed successfully\nDeployment successful\n"


# =============================================================================
# TestRunGhCommand
# =============================================================================


class TestRunGhCommand:
    """Tests for run_gh_command() function."""

    def test_run_gh_command_success_with_json(self) -> None:
        """Successfully run gh command and parse JSON output."""
        expected_output = [
            {"id": 1, "name": "test1"},
            {"id": 2, "name": "test2"},
        ]
        mock_result = MagicMock()
        mock_result.stdout = json.dumps(expected_output)

        with patch("subprocess.run", return_value=mock_result):
            result = run_gh_command(["gh", "run", "list"])

        assert result == expected_output

    def test_run_gh_command_empty_json_array(self) -> None:
        """Handle empty JSON array response."""
        mock_result = MagicMock()
        mock_result.stdout = "[]"

        with patch("subprocess.run", return_value=mock_result):
            result = run_gh_command(["gh", "run", "list"])

        assert result == []

    def test_run_gh_command_empty_stdout_returns_empty_dict(self) -> None:
        """Empty stdout returns empty dict."""
        mock_result = MagicMock()
        mock_result.stdout = ""

        with patch("subprocess.run", return_value=mock_result):
            result = run_gh_command(["gh", "run", "list"])

        assert result == {}

    def test_run_gh_command_subprocess_error_exits(self, capsys: pytest.CaptureFixture[str]) -> None:
        """CalledProcessError exits with status 1."""
        with patch(
            "subprocess.run",
            side_effect=subprocess.CalledProcessError(
                1, "gh", stderr="Error: not authenticated"
            ),
        ):
            with pytest.raises(SystemExit) as exc_info:
                run_gh_command(["gh", "run", "list"])

        assert exc_info.value.code == 1
        captured = capsys.readouterr()
        assert "Error running gh command" in captured.err
        assert "not authenticated" in captured.err

    def test_run_gh_command_json_decode_error_exits(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """JSONDecodeError exits with status 1."""
        mock_result = MagicMock()
        mock_result.stdout = "not valid json"

        with patch("subprocess.run", return_value=mock_result):
            with pytest.raises(SystemExit) as exc_info:
                run_gh_command(["gh", "run", "list"])

        assert exc_info.value.code == 1
        captured = capsys.readouterr()
        assert "Error parsing JSON" in captured.err

    def test_run_gh_command_passes_correct_arguments(self) -> None:
        """Verify correct arguments are passed to subprocess.run."""
        mock_result = MagicMock()
        mock_result.stdout = "[]"

        with patch("subprocess.run", return_value=mock_result) as mock_run:
            cmd = ["gh", "run", "list", "--status", "failure"]
            run_gh_command(cmd)

        mock_run.assert_called_once_with(cmd, capture_output=True, text=True, check=True)

    def test_run_gh_command_with_complex_json_object(self) -> None:
        """Parse complex nested JSON objects."""
        expected_output = {
            "run": {
                "id": 123,
                "jobs": [
                    {"id": 1, "name": "test", "logs": "content"},
                    {"id": 2, "name": "build", "logs": "more"},
                ],
            }
        }
        mock_result = MagicMock()
        mock_result.stdout = json.dumps(expected_output)

        with patch("subprocess.run", return_value=mock_result):
            result = run_gh_command(["gh", "run", "view", "123"])

        assert result == expected_output
        assert result["run"]["jobs"][0]["name"] == "test"


# =============================================================================
# TestGetMostRecentFailedRun
# =============================================================================


class TestGetMostRecentFailedRun:
    """Tests for get_most_recent_failed_run() function."""

    def test_get_most_recent_failed_run_success(
        self, sample_failed_run: dict
    ) -> None:
        """Retrieve most recent failed run successfully."""
        mock_result = MagicMock()
        mock_result.stdout = json.dumps([sample_failed_run])

        with patch("subprocess.run", return_value=mock_result):
            result = get_most_recent_failed_run()

        assert result == sample_failed_run
        assert result["conclusion"] == "failure"
        assert result["number"] == 42

    def test_get_most_recent_failed_run_with_repo(
        self, sample_failed_run: dict
    ) -> None:
        """Specify repository when getting failed run."""
        mock_result = MagicMock()
        mock_result.stdout = json.dumps([sample_failed_run])

        with patch("subprocess.run", return_value=mock_result) as mock_run:
            result = get_most_recent_failed_run(repo="owner/repo")

        assert result == sample_failed_run
        # Verify --repo flag was passed
        args = mock_run.call_args[0][0]
        assert "--repo" in args
        assert "owner/repo" in args

    def test_get_most_recent_failed_run_no_failed_runs(self) -> None:
        """Return None when no failed runs found."""
        mock_result = MagicMock()
        mock_result.stdout = "[]"

        with patch("subprocess.run", return_value=mock_result):
            result = get_most_recent_failed_run()

        assert result is None

    def test_get_most_recent_failed_run_returns_first(
        self, sample_failed_run: dict
    ) -> None:
        """Return first (most recent) run from list."""
        second_run = dict(sample_failed_run)
        second_run["number"] = 41
        second_run["createdAt"] = "2025-01-14T10:30:00Z"

        mock_result = MagicMock()
        mock_result.stdout = json.dumps([sample_failed_run, second_run])

        with patch("subprocess.run", return_value=mock_result):
            result = get_most_recent_failed_run()

        assert result["number"] == 42  # First one (most recent)

    def test_get_most_recent_failed_run_includes_all_fields(
        self, sample_failed_run: dict
    ) -> None:
        """Verify all required fields are present."""
        mock_result = MagicMock()
        mock_result.stdout = json.dumps([sample_failed_run])

        with patch("subprocess.run", return_value=mock_result):
            result = get_most_recent_failed_run()

        required_fields = [
            "databaseId",
            "number",
            "conclusion",
            "status",
            "createdAt",
            "displayTitle",
            "url",
            "headBranch",
            "headSha",
            "event",
        ]
        for field in required_fields:
            assert field in result


# =============================================================================
# TestGetFailedJobs
# =============================================================================


class TestGetFailedJobs:
    """Tests for get_failed_jobs() function."""

    def test_get_failed_jobs_filters_successfully(
        self, sample_jobs: dict[str, Any]
    ) -> None:
        """Filter jobs and return only failed ones."""
        mock_result = MagicMock()
        mock_result.stdout = json.dumps(sample_jobs)

        with patch("subprocess.run", return_value=mock_result):
            result = get_failed_jobs(12345)

        # Should have 2 failed jobs (failure and timed_out)
        assert len(result) == 2
        assert all(job["conclusion"] != "success" for job in result)
        assert result[0]["name"] == "Unit Tests"
        assert result[1]["name"] == "Integration Tests"

    def test_get_failed_jobs_excludes_success(
        self, sample_jobs: dict[str, Any]
    ) -> None:
        """Exclude jobs with success conclusion."""
        mock_result = MagicMock()
        mock_result.stdout = json.dumps(sample_jobs)

        with patch("subprocess.run", return_value=mock_result):
            result = get_failed_jobs(12345)

        assert not any(job["conclusion"] == "success" for job in result)

    def test_get_failed_jobs_excludes_skipped(self) -> None:
        """Exclude skipped jobs."""
        jobs = {
            "jobs": [
                {
                    "name": "Skipped job",
                    "conclusion": "skipped",
                    "status": "skipped",
                },
                {
                    "name": "Failed job",
                    "conclusion": "failure",
                    "status": "completed",
                },
            ]
        }
        mock_result = MagicMock()
        mock_result.stdout = json.dumps(jobs)

        with patch("subprocess.run", return_value=mock_result):
            result = get_failed_jobs(12345)

        assert len(result) == 1
        assert result[0]["name"] == "Failed job"

    def test_get_failed_jobs_includes_timed_out(self) -> None:
        """Include jobs with timed_out conclusion."""
        jobs = {
            "jobs": [
                {
                    "name": "Timeout job",
                    "conclusion": "timed_out",
                    "status": "completed",
                }
            ]
        }
        mock_result = MagicMock()
        mock_result.stdout = json.dumps(jobs)

        with patch("subprocess.run", return_value=mock_result):
            result = get_failed_jobs(12345)

        assert len(result) == 1
        assert result[0]["conclusion"] == "timed_out"

    def test_get_failed_jobs_includes_cancelled(self) -> None:
        """Include jobs with cancelled conclusion."""
        jobs = {
            "jobs": [
                {
                    "name": "Cancelled job",
                    "conclusion": "cancelled",
                    "status": "completed",
                }
            ]
        }
        mock_result = MagicMock()
        mock_result.stdout = json.dumps(jobs)

        with patch("subprocess.run", return_value=mock_result):
            result = get_failed_jobs(12345)

        assert len(result) == 1
        assert result[0]["conclusion"] == "cancelled"


    def test_get_failed_jobs_excludes_none_conclusion(self) -> None:
        """Exclude jobs with None conclusion (in-progress jobs)."""
        jobs = {
            "jobs": [
                {
                    "name": "In progress job",
                    "conclusion": None,
                    "status": "in_progress",
                }
            ]
        }
        mock_result = MagicMock()
        mock_result.stdout = json.dumps(jobs)

        with patch("subprocess.run", return_value=mock_result):
            result = get_failed_jobs(12345)

        assert result == []

    def test_get_failed_jobs_empty_list(self) -> None:
        """Return empty list when no jobs present."""
        jobs = {"jobs": []}
        mock_result = MagicMock()
        mock_result.stdout = json.dumps(jobs)

        with patch("subprocess.run", return_value=mock_result):
            result = get_failed_jobs(12345)

        assert result == []

    def test_get_failed_jobs_with_repo(self) -> None:
        """Specify repository when getting failed jobs."""
        jobs = {
            "jobs": [
                {"name": "Test", "conclusion": "failure", "status": "completed"}
            ]
        }
        mock_result = MagicMock()
        mock_result.stdout = json.dumps(jobs)

        with patch("subprocess.run", return_value=mock_result) as mock_run:
            get_failed_jobs(12345, repo="owner/repo")

        args = mock_run.call_args[0][0]
        assert "--repo" in args
        assert "owner/repo" in args


# =============================================================================
# TestExtractErrorExcerpts
# =============================================================================


class TestExtractErrorExcerpts:
    """Tests for extract_error_excerpts() function."""

    def test_extract_error_excerpts_finds_error_pattern(
        self, sample_job_logs: str
    ) -> None:
        """Extract lines matching 'error' pattern."""
        result = extract_error_excerpts(sample_job_logs)

        assert len(result) > 0
        assert any("error" in line.lower() for line in result)

    def test_extract_error_excerpts_finds_failed_pattern(
        self, sample_job_logs: str
    ) -> None:
        """Extract lines matching 'failed' pattern."""
        result = extract_error_excerpts(sample_job_logs)

        assert any("failed" in line.lower() for line in result)

    def test_extract_error_excerpts_finds_failure_pattern(
        self, sample_job_logs: str
    ) -> None:
        """Extract lines matching 'failure' pattern."""
        result = extract_error_excerpts(sample_job_logs)

        # Sample has "Connection failed"
        assert any("failed" in line.lower() for line in result)

    def test_extract_error_excerpts_finds_exception_pattern(
        self, sample_job_logs: str
    ) -> None:
        """Extract lines matching 'exception' pattern."""
        result = extract_error_excerpts(sample_job_logs)

        assert any("exception" in line.lower() for line in result)

    def test_extract_error_excerpts_finds_cannot_pattern(
        self, sample_job_logs: str
    ) -> None:
        """Extract lines matching 'cannot' pattern."""
        result = extract_error_excerpts(sample_job_logs)

        assert any("cannot" in line.lower() for line in result)

    def test_extract_error_excerpts_finds_panic_pattern(
        self, sample_job_logs: str
    ) -> None:
        """Extract lines matching 'panic' pattern."""
        result = extract_error_excerpts(sample_job_logs)

        assert any("panic" in line.lower() for line in result)

    def test_extract_error_excerpts_finds_exit_code_pattern(
        self, sample_job_logs: str
    ) -> None:
        """Extract lines matching 'exit code' pattern."""
        result = extract_error_excerpts(sample_job_logs)

        assert any("exit code" in line.lower() for line in result)

    def test_extract_error_excerpts_finds_timeout_pattern(
        self, sample_job_logs: str
    ) -> None:
        """Extract lines matching 'timeout' pattern."""
        result = extract_error_excerpts(sample_job_logs)

        assert any("timeout" in line.lower() for line in result)

    def test_extract_error_excerpts_removes_ansi_codes(
        self, sample_job_logs: str
    ) -> None:
        """Remove ANSI color codes from extracted lines."""
        result = extract_error_excerpts(sample_job_logs)

        # Should not contain ANSI codes
        assert not any("\x1b[" in line for line in result)

    def test_extract_error_excerpts_removes_timestamps(
        self, sample_job_logs: str
    ) -> None:
        """Remove ISO 8601 timestamps from extracted lines."""
        result = extract_error_excerpts(sample_job_logs)

        # Should not contain timestamps
        assert not any("2025-01-15T" in line for line in result)

    def test_extract_error_excerpts_deduplicates(self) -> None:
        """Remove duplicate error lines."""
        logs = (
            "ERROR: Connection failed\n"
            "ERROR: Connection failed\n"
            "ERROR: Connection failed\n"
        )

        result = extract_error_excerpts(logs)

        assert len(result) == 1

    def test_extract_error_excerpts_respects_max_lines(
        self, sample_job_logs: str
    ) -> None:
        """Respect max_lines limit."""
        result = extract_error_excerpts(sample_job_logs, max_lines=2)

        assert len(result) <= 2

    def test_extract_error_excerpts_empty_logs(self) -> None:
        """Handle empty log text."""
        result = extract_error_excerpts("")

        assert result == []

    def test_extract_error_excerpts_no_errors(self) -> None:
        """Return empty list when no errors found."""
        logs = "All tests passed\nBuild successful\nDeployment complete\n"

        result = extract_error_excerpts(logs)

        assert result == []

    def test_extract_error_excerpts_skips_empty_lines(self) -> None:
        """Skip empty and whitespace-only lines."""
        logs = (
            "ERROR: Something\n"
            "\n"
            "  \n"
            "FAILED: Something else\n"
        )

        result = extract_error_excerpts(logs)

        assert all(line.strip() for line in result)  # No empty lines

    def test_extract_error_excerpts_case_insensitive_patterns(self) -> None:
        """Patterns are case-insensitive."""
        logs = (
            "ERROR: uppercase\n"
            "error: lowercase\n"
            "Error: titlecase\n"
        )

        result = extract_error_excerpts(logs)

        assert len(result) == 3

    @pytest.mark.parametrize(
        "error_keyword",
        [
            "error",
            "failed",
            "failure",
            "exception",
            "cannot",
            "panic",
            "timeout",
        ],
    )
    def test_extract_error_excerpts_detects_all_keywords(
        self, error_keyword: str
    ) -> None:
        """Detect all documented error keywords."""
        logs = f"Process {error_keyword} occurred\n"

        result = extract_error_excerpts(logs)

        assert len(result) > 0


# =============================================================================
# TestGetJobLogs
# =============================================================================


class TestGetJobLogs:
    """Tests for get_job_logs() function."""

    def test_get_job_logs_success_with_specific_job(
        self, sample_job_logs: str
    ) -> None:
        """Successfully retrieve logs for specific job."""
        mock_result = MagicMock()
        mock_result.stdout = sample_job_logs

        with patch("subprocess.run", return_value=mock_result):
            result = get_job_logs(12345, "Unit Tests")

        assert result == sample_job_logs

    def test_get_job_logs_specific_job_failure_fallback_to_all(
        self, sample_job_logs: str
    ) -> None:
        """Fall back to all logs when specific job logs fail."""
        mock_result = MagicMock()
        mock_result.stdout = sample_job_logs

        def mock_subprocess_run(cmd, *args, **kwargs):
            if "--job" in cmd:
                raise subprocess.CalledProcessError(1, "gh")
            return mock_result

        with patch("subprocess.run", side_effect=mock_subprocess_run):
            result = get_job_logs(12345, "Unit Tests")

        assert result == sample_job_logs

    def test_get_job_logs_all_logs_failure_returns_empty(self) -> None:
        """Return empty string when both specific and all logs fail."""

        def mock_subprocess_run(cmd, *args, **kwargs):
            raise subprocess.CalledProcessError(1, "gh")

        with patch("subprocess.run", side_effect=mock_subprocess_run):
            result = get_job_logs(12345, "Unit Tests")

        assert result == ""

    def test_get_job_logs_with_repo(self, sample_job_logs: str) -> None:
        """Specify repository when getting job logs."""
        mock_result = MagicMock()
        mock_result.stdout = sample_job_logs

        with patch("subprocess.run", return_value=mock_result) as mock_run:
            get_job_logs(12345, "Build", repo="owner/repo")

        # Check that --repo flag was passed
        args = mock_run.call_args_list[0][0][0]  # First call
        assert "--repo" in args
        assert "owner/repo" in args

    def test_get_job_logs_uses_log_failed_flag(
        self, sample_job_logs: str
    ) -> None:
        """Verify --log-failed flag is used."""
        mock_result = MagicMock()
        mock_result.stdout = sample_job_logs

        with patch("subprocess.run", return_value=mock_result) as mock_run:
            get_job_logs(12345, "Build")

        args = mock_run.call_args[0][0]
        assert "--log-failed" in args

    def test_get_job_logs_includes_job_name(
        self, sample_job_logs: str
    ) -> None:
        """Verify job name is passed to command."""
        mock_result = MagicMock()
        mock_result.stdout = sample_job_logs

        with patch("subprocess.run", return_value=mock_result) as mock_run:
            get_job_logs(12345, "Build")

        args = mock_run.call_args[0][0]
        assert "--job" in args
        assert "Build" in args


# =============================================================================
# TestAnalyzeFailedRun
# =============================================================================


class TestAnalyzeFailedRun:
    """Tests for analyze_failed_run() function."""

    def test_analyze_failed_run_no_failed_runs(self) -> None:
        """Return error structure when no failed runs found."""
        mock_result = MagicMock()
        mock_result.stdout = "[]"

        with patch("subprocess.run", return_value=mock_result):
            result = analyze_failed_run()

        assert "error" in result
        assert result["error"] == "No failed runs found"
        assert result["repository"] == "current"

    def test_analyze_failed_run_with_repo(self, sample_failed_run: dict) -> None:
        """Analyze failed run in specific repository."""
        jobs = {"jobs": []}
        mock_failed_run = MagicMock()
        mock_failed_run.stdout = json.dumps([sample_failed_run])

        mock_jobs = MagicMock()
        mock_jobs.stdout = json.dumps(jobs)

        def mock_subprocess_run(cmd, *args, **kwargs):
            if "list" in cmd:
                return mock_failed_run
            return mock_jobs

        with patch("subprocess.run", side_effect=mock_subprocess_run):
            result = analyze_failed_run(repo="owner/repo")

        assert result["repository"] == "owner/repo"
        assert "run" in result

    def test_analyze_failed_run_includes_run_details(
        self, sample_failed_run: dict
    ) -> None:
        """Include all run details in analysis."""
        jobs = {"jobs": []}
        mock_failed_run = MagicMock()
        mock_failed_run.stdout = json.dumps([sample_failed_run])

        mock_jobs = MagicMock()
        mock_jobs.stdout = json.dumps(jobs)

        def mock_subprocess_run(cmd, *args, **kwargs):
            if "list" in cmd:
                return mock_failed_run
            return mock_jobs

        with patch("subprocess.run", side_effect=mock_subprocess_run):
            result = analyze_failed_run()

        run = result["run"]
        assert run["number"] == 42
        assert run["database_id"] == 12345
        assert run["workflow"] == "CI/CD Pipeline"
        assert run["conclusion"] == "failure"
        assert run["branch"] == "main"
        assert run["commit"] == "abc1234567890def"

    def test_analyze_failed_run_includes_failed_jobs(
        self, sample_failed_run: dict, sample_jobs: dict[str, Any]
    ) -> None:
        """Include all failed jobs in analysis."""
        mock_failed_run = MagicMock()
        mock_failed_run.stdout = json.dumps([sample_failed_run])

        mock_jobs_result = MagicMock()
        mock_jobs_result.stdout = json.dumps(sample_jobs)

        mock_logs = MagicMock()
        mock_logs.stdout = ""

        def mock_subprocess_run(cmd, *args, **kwargs):
            if "list" in cmd:
                return mock_failed_run
            elif "view" in cmd:
                if "--job" in cmd or "--log-failed" in cmd:
                    return mock_logs
                return mock_jobs_result
            return mock_logs

        with patch("subprocess.run", side_effect=mock_subprocess_run):
            result = analyze_failed_run()

        assert len(result["failed_jobs"]) == 2
        assert result["failed_jobs"][0]["name"] == "Unit Tests"
        assert result["failed_jobs"][1]["name"] == "Integration Tests"

    def test_analyze_failed_run_extracts_error_excerpts(
        self, sample_failed_run: dict, sample_job_logs: str
    ) -> None:
        """Extract error excerpts from job logs."""
        jobs = {
            "jobs": [
                {
                    "databaseId": 1001,
                    "name": "Unit Tests",
                    "conclusion": "failure",
                    "status": "completed",
                    "startedAt": "2025-01-15T10:30:10Z",
                    "completedAt": "2025-01-15T10:31:00Z",
                }
            ]
        }

        mock_failed_run = MagicMock()
        mock_failed_run.stdout = json.dumps([sample_failed_run])

        mock_jobs_result = MagicMock()
        mock_jobs_result.stdout = json.dumps(jobs)

        mock_logs = MagicMock()
        mock_logs.stdout = sample_job_logs

        def mock_subprocess_run(cmd, *args, **kwargs):
            if "list" in cmd:
                return mock_failed_run
            elif "view" in cmd:
                if "--log-failed" in cmd:
                    return mock_logs
                return mock_jobs_result
            return mock_logs

        with patch("subprocess.run", side_effect=mock_subprocess_run):
            result = analyze_failed_run()

        job_info = result["failed_jobs"][0]
        assert "error_excerpts" in job_info
        assert len(job_info["error_excerpts"]) > 0

    def test_analyze_failed_run_handles_empty_logs(
        self, sample_failed_run: dict
    ) -> None:
        """Handle jobs with no logs gracefully."""
        jobs = {
            "jobs": [
                {
                    "databaseId": 1001,
                    "name": "Unit Tests",
                    "conclusion": "failure",
                    "status": "completed",
                }
            ]
        }

        mock_failed_run = MagicMock()
        mock_failed_run.stdout = json.dumps([sample_failed_run])

        mock_jobs_result = MagicMock()
        mock_jobs_result.stdout = json.dumps(jobs)

        mock_logs = MagicMock()
        mock_logs.stdout = ""

        def mock_subprocess_run(cmd, *args, **kwargs):
            if "list" in cmd:
                return mock_failed_run
            elif "view" in cmd:
                if "--log-failed" in cmd:
                    return mock_logs
                return mock_jobs_result
            return mock_logs

        with patch("subprocess.run", side_effect=mock_subprocess_run):
            result = analyze_failed_run()

        job_info = result["failed_jobs"][0]
        assert job_info["error_excerpts"] == []

    def test_analyze_failed_run_includes_job_details(
        self, sample_failed_run: dict
    ) -> None:
        """Include job metadata in analysis."""
        jobs = {
            "jobs": [
                {
                    "databaseId": 1001,
                    "name": "Build Job",
                    "conclusion": "failure",
                    "status": "completed",
                    "startedAt": "2025-01-15T10:30:10Z",
                    "completedAt": "2025-01-15T10:31:00Z",
                }
            ]
        }

        mock_failed_run = MagicMock()
        mock_failed_run.stdout = json.dumps([sample_failed_run])

        mock_jobs_result = MagicMock()
        mock_jobs_result.stdout = json.dumps(jobs)

        mock_logs = MagicMock()
        mock_logs.stdout = ""

        def mock_subprocess_run(cmd, *args, **kwargs):
            if "list" in cmd:
                return mock_failed_run
            elif "view" in cmd:
                if "--log-failed" in cmd:
                    return mock_logs
                return mock_jobs_result
            return mock_logs

        with patch("subprocess.run", side_effect=mock_subprocess_run):
            result = analyze_failed_run()

        job_info = result["failed_jobs"][0]
        assert job_info["name"] == "Build Job"
        assert job_info["conclusion"] == "failure"
        assert job_info["status"] == "completed"


# =============================================================================


class TestMain:
    """Tests for main() function entry point."""

    def test_main_gh_not_installed_exits(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Exit with error if gh CLI not installed."""
        with patch("gh_failed_run.subprocess.run", side_effect=FileNotFoundError()):
            with patch("sys.argv", ["gh_failed_run.py"]):
                with pytest.raises(SystemExit) as exc_info:
                    main()

        assert exc_info.value.code == 1
        captured = capsys.readouterr()
        assert "gh CLI is not installed" in captured.err

    def test_main_default_output_no_pretty(
        self, sample_failed_run: dict[str, Any], capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Output compact JSON by default."""
        jobs: dict[str, Any] = {"jobs": []}
        mock_failed_run = MagicMock()
        mock_failed_run.stdout = json.dumps([sample_failed_run])

        mock_jobs = MagicMock()
        mock_jobs.stdout = json.dumps(jobs)

        mock_version = MagicMock()

        def mock_subprocess_run(
            cmd: list[str], *args: Any, **kwargs: Any
        ) -> MagicMock:
            if "--version" in cmd:
                return mock_version
            if "list" in cmd:
                return mock_failed_run
            return mock_jobs

        with patch("gh_failed_run.subprocess.run", side_effect=mock_subprocess_run):
            with patch("sys.argv", ["gh_failed_run.py"]):
                main()

        captured = capsys.readouterr()
        output = json.loads(captured.out)
        assert "run" in output
        assert "failed_jobs" in output

    def test_main_pretty_print_output(
        self, sample_failed_run: dict[str, Any], capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Output pretty-printed JSON with --pretty flag."""
        jobs: dict[str, Any] = {"jobs": []}
        mock_failed_run = MagicMock()
        mock_failed_run.stdout = json.dumps([sample_failed_run])

        mock_jobs = MagicMock()
        mock_jobs.stdout = json.dumps(jobs)

        mock_version = MagicMock()

        def mock_subprocess_run(
            cmd: list[str], *args: Any, **kwargs: Any
        ) -> MagicMock:
            if "--version" in cmd:
                return mock_version
            if "list" in cmd:
                return mock_failed_run
            return mock_jobs

        with patch("gh_failed_run.subprocess.run", side_effect=mock_subprocess_run):
            with patch("sys.argv", ["gh_failed_run.py", "--pretty"]):
                main()

        captured = capsys.readouterr()
        assert "  " in captured.out

    def test_main_with_repo_argument(
        self, sample_failed_run: dict[str, Any], capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Accept --repo argument."""
        jobs: dict[str, Any] = {"jobs": []}
        mock_failed_run = MagicMock()
        mock_failed_run.stdout = json.dumps([sample_failed_run])

        mock_jobs = MagicMock()
        mock_jobs.stdout = json.dumps(jobs)

        mock_version = MagicMock()

        def mock_subprocess_run(
            cmd: list[str], *args: Any, **kwargs: Any
        ) -> MagicMock:
            if "--version" in cmd:
                return mock_version
            if "list" in cmd:
                return mock_failed_run
            return mock_jobs

        with patch("gh_failed_run.subprocess.run", side_effect=mock_subprocess_run):
            with patch("sys.argv", ["gh_failed_run.py", "--repo", "owner/repo"]):
                main()

        captured = capsys.readouterr()
        output = json.loads(captured.out)
        assert output["repository"] == "owner/repo"

    def test_main_no_failed_runs_outputs_error(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Output error message when no failed runs found."""
        mock_result = MagicMock()
        mock_result.stdout = "[]"

        mock_version = MagicMock()

        def mock_subprocess_run(
            cmd: list[str], *args: Any, **kwargs: Any
        ) -> MagicMock:
            if "--version" in cmd:
                return mock_version
            return mock_result

        with patch("gh_failed_run.subprocess.run", side_effect=mock_subprocess_run):
            with patch("sys.argv", ["gh_failed_run.py"]):
                main()

        captured = capsys.readouterr()
        output = json.loads(captured.out)
        assert "error" in output
        assert output["error"] == "No failed runs found"

    def test_main_completes_successfully(
        self, sample_failed_run: dict[str, Any], capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Main completes successfully without raising."""
        jobs: dict[str, Any] = {"jobs": []}
        mock_failed_run = MagicMock()
        mock_failed_run.stdout = json.dumps([sample_failed_run])

        mock_jobs = MagicMock()
        mock_jobs.stdout = json.dumps(jobs)

        mock_version = MagicMock()

        def mock_subprocess_run(
            cmd: list[str], *args: Any, **kwargs: Any
        ) -> MagicMock:
            if "--version" in cmd:
                return mock_version
            if "list" in cmd:
                return mock_failed_run
            return mock_jobs

        with patch("gh_failed_run.subprocess.run", side_effect=mock_subprocess_run):
            with patch("sys.argv", ["gh_failed_run.py"]):
                main()

        captured = capsys.readouterr()
        output = json.loads(captured.out)
        assert output["run"]["number"] == 42




class TestIntegration:
    """Integration tests combining multiple functions."""

    def test_full_analysis_workflow(
        self, sample_failed_run: dict, sample_job_logs: str
    ) -> None:
        """Complete workflow from run retrieval to analysis."""
        jobs = {
            "jobs": [
                {
                    "databaseId": 1001,
                    "name": "Unit Tests",
                    "conclusion": "failure",
                    "status": "completed",
                    "startedAt": "2025-01-15T10:30:10Z",
                    "completedAt": "2025-01-15T10:31:00Z",
                }
            ]
        }

        mock_failed_run = MagicMock()
        mock_failed_run.stdout = json.dumps([sample_failed_run])

        mock_jobs_result = MagicMock()
        mock_jobs_result.stdout = json.dumps(jobs)

        mock_logs = MagicMock()
        mock_logs.stdout = sample_job_logs

        def mock_subprocess_run(cmd, *args, **kwargs):
            if "list" in cmd:
                return mock_failed_run
            elif "view" in cmd:
                if "--log-failed" in cmd:
                    return mock_logs
                return mock_jobs_result
            return mock_logs

        with patch("subprocess.run", side_effect=mock_subprocess_run):
            result = analyze_failed_run()

        # Verify complete structure
        assert "run" in result
        assert "failed_jobs" in result
        assert result["run"]["number"] == 42
        assert len(result["failed_jobs"]) == 1
        assert len(result["failed_jobs"][0]["error_excerpts"]) > 0

    def test_multiple_failed_jobs_with_different_errors(
        self, sample_failed_run: dict
    ) -> None:
        """Analyze run with multiple failed jobs with different errors."""
        jobs = {
            "jobs": [
                {
                    "databaseId": 1001,
                    "name": "Unit Tests",
                    "conclusion": "failure",
                    "status": "completed",
                },
                {
                    "databaseId": 1002,
                    "name": "Integration Tests",
                    "conclusion": "timed_out",
                    "status": "completed",
                },
            ]
        }

        unit_test_logs = (
            "ERROR: Assertion failed\n"
            "Test failed at line 42\n"
        )
        integration_logs = (
            "Timeout waiting for server\n"
            "Test timeout after 300 seconds\n"
        )

        mock_failed_run = MagicMock()
        mock_failed_run.stdout = json.dumps([sample_failed_run])

        mock_jobs_result = MagicMock()
        mock_jobs_result.stdout = json.dumps(jobs)

        call_count = 0
        def mock_subprocess_run(cmd, *args, **kwargs):
            nonlocal call_count
            if "list" in cmd:
                return mock_failed_run
            elif "view" in cmd:
                if "--log-failed" in cmd:
                    logs = MagicMock()
                    if call_count % 2 == 0:
                        logs.stdout = unit_test_logs
                    else:
                        logs.stdout = integration_logs
                    call_count += 1
                    return logs
                return mock_jobs_result
            return mock_jobs_result

        with patch("subprocess.run", side_effect=mock_subprocess_run):
            result = analyze_failed_run()

        assert len(result["failed_jobs"]) == 2


# =============================================================================
# TestEdgeCases
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and error conditions."""

    def test_extract_error_excerpts_with_very_long_logs(self) -> None:
        """Handle very large log files efficiently."""
        # 1000 lines of mixed content
        logs = "\n".join(
            [
                f"Line {i}: {'ERROR: Problem' if i % 10 == 0 else 'Normal output'}"
                for i in range(1000)
            ]
        )

        result = extract_error_excerpts(logs, max_lines=100)

        assert len(result) <= 100

    def test_get_job_logs_with_special_characters_in_name(
        self,
    ) -> None:
        """Handle job names with special characters."""
        mock_result = MagicMock()
        mock_result.stdout = "logs content"

        with patch("subprocess.run", return_value=mock_result) as mock_run:
            get_job_logs(12345, "Test [1.0] (C++)")

        args = mock_run.call_args[0][0]
        assert "Test [1.0] (C++)" in args

    def test_analyze_failed_run_with_missing_optional_fields(
        self,
    ) -> None:
        """Handle runs with missing optional fields gracefully."""
        # Minimal run object
        minimal_run = {
            "databaseId": 1,
            "number": 1,
            "conclusion": "failure",
            "status": "completed",
            "createdAt": "2025-01-01T00:00:00Z",
            "displayTitle": "Test",
            "url": "http://example.com",
        }

        jobs = {"jobs": []}
        mock_failed_run = MagicMock()
        mock_failed_run.stdout = json.dumps([minimal_run])

        mock_jobs_result = MagicMock()
        mock_jobs_result.stdout = json.dumps(jobs)

        def mock_subprocess_run(cmd, *args, **kwargs):
            if "list" in cmd:
                return mock_failed_run
            return mock_jobs_result

        with patch("subprocess.run", side_effect=mock_subprocess_run):
            result = analyze_failed_run()

        # Should still produce valid output
        assert "run" in result
        assert result["run"]["number"] == 1

    def test_extract_error_excerpts_with_unicode_content(self) -> None:
        """Handle unicode characters in logs."""
        logs = (
            "ERROR: 连接失败\n"
            "Failed: 测试失败\n"
            "Exception: 异常 🔴\n"
        )

        result = extract_error_excerpts(logs)

        assert len(result) > 0
        assert any("失败" in line or "异常" in line for line in result)

    def test_get_failed_jobs_with_all_success(self, sample_jobs: dict[str, Any]) -> None:
        """Handle runs where all jobs succeeded."""
        all_success_jobs = {
            "jobs": [
                {
                    "name": "Job 1",
                    "conclusion": "success",
                    "status": "completed",
                },
                {
                    "name": "Job 2",
                    "conclusion": "success",
                    "status": "completed",
                },
            ]
        }

        mock_result = MagicMock()
        mock_result.stdout = json.dumps(all_success_jobs)

        with patch("subprocess.run", return_value=mock_result):
            result = get_failed_jobs(12345)

        assert result == []


# =============================================================================
# Main Entry Point
# =============================================================================


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
