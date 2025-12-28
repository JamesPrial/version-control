"""
Comprehensive test suite for gh_pages_deploy.py GitHub Pages manager.

Tests cover:
- GitHubPagesManager initialization and gh CLI verification
- GitHub API interaction via gh api calls
- GitHub Pages enabling with different configurations
- Pages status checking and parsing
- Rebuild triggering
- Latest build information retrieval
- GitHub Actions workflow file creation
- Main function CLI integration tests
- Error handling and edge cases
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from gh_pages_deploy import (
    GitHubPagesManager,
    create_workflow_file,
    main,
)

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def mock_gh_auth_success() -> MagicMock:
    """Mock successful gh auth status."""
    result = MagicMock()
    result.returncode = 0
    result.stdout = "Logged in to github.com"
    result.stderr = ""
    return result


@pytest.fixture
def mock_gh_auth_failure() -> MagicMock:
    """Mock failed gh auth status."""
    result = MagicMock()
    result.returncode = 1
    result.stdout = ""
    result.stderr = "Not authenticated"
    return result


@pytest.fixture
def sample_pages_response() -> dict[str, Any]:
    """Sample GitHub Pages API response when enabled."""
    return {
        "url": "https://api.github.com/repos/owner/repo/pages",
        "status": "built",
        "cname": None,
        "custom_404": False,
        "html_url": "https://owner.github.io/repo/",
        "build_type": "workflow",
        "source": {
            "branch": "main",
            "path": "/"
        },
        "https_enforced": True,
        "public": True,
        "pending_domain": None
    }


@pytest.fixture
def sample_pages_disabled_response() -> dict[str, Any]:
    """Sample GitHub Pages API response when disabled."""
    return {
        "message": "Not Found",
        "documentation_url": "https://docs.github.com/rest/pages/pages?apiVersion=2022-11-28"
    }


@pytest.fixture
def sample_build_response() -> dict[str, Any]:
    """Sample GitHub Pages build response."""
    return {
        "url": "https://api.github.com/repos/owner/repo/pages/builds/1",
        "status": "built",
        "commit": "abc123def456",
        "created_at": "2025-12-28T10:30:00Z",
        "updated_at": "2025-12-28T10:31:00Z"
    }


@pytest.fixture
def sample_build_error_response() -> dict[str, Any]:
    """Sample GitHub Pages build error response."""
    return {
        "url": "https://api.github.com/repos/owner/repo/pages/builds/1",
        "status": "failed",
        "commit": "abc123def456",
        "created_at": "2025-12-28T10:30:00Z",
        "updated_at": "2025-12-28T10:31:00Z",
        "error": {
            "message": "Permission denied"
        }
    }


# =============================================================================
# TestGitHubPagesManagerInit
# =============================================================================


class TestGitHubPagesManagerInit:
    """Tests for GitHubPagesManager initialization and gh CLI verification."""

    def test_init_with_valid_repo_format(
        self, mock_gh_auth_success: MagicMock
    ) -> None:
        """Initialize with valid owner/repo format."""
        with patch("subprocess.run", return_value=mock_gh_auth_success):
            manager = GitHubPagesManager("owner/repo")

        assert manager.repo == "owner/repo"

    def test_init_calls_verify_gh_cli(
        self, mock_gh_auth_success: MagicMock
    ) -> None:
        """Initialization calls _verify_gh_cli()."""
        with patch("subprocess.run", return_value=mock_gh_auth_success) as mock_run:
            GitHubPagesManager("owner/repo")

        # Verify gh auth status was called
        assert mock_run.called
        call_args = mock_run.call_args[0][0]
        assert "gh" in call_args
        assert "auth" in call_args
        assert "status" in call_args

    def test_init_with_unauthenticated_gh_cli_exits(
        self, mock_gh_auth_failure: MagicMock
    ) -> None:
        """Initialization fails when gh CLI is not authenticated."""
        with patch("subprocess.run", return_value=mock_gh_auth_failure):
            with pytest.raises(SystemExit) as exc_info:
                GitHubPagesManager("owner/repo")

        assert exc_info.value.code == 1

    def test_init_with_missing_gh_cli_exits(self) -> None:
        """Initialization fails when gh CLI is not installed."""
        with patch("subprocess.run", side_effect=FileNotFoundError()):
            with pytest.raises(SystemExit) as exc_info:
                GitHubPagesManager("owner/repo")

        assert exc_info.value.code == 1

    def test_init_prints_gh_cli_error_messages(
        self, mock_gh_auth_failure: MagicMock, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Error messages are printed to stderr on auth failure."""
        with patch("subprocess.run", return_value=mock_gh_auth_failure):
            with pytest.raises(SystemExit):
                GitHubPagesManager("owner/repo")

        captured = capsys.readouterr()
        assert "GitHub CLI is not authenticated" in captured.err
        assert "gh auth login" in captured.err

    def test_init_prints_gh_install_error_on_missing(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Installation message is printed on missing gh CLI."""
        with patch("subprocess.run", side_effect=FileNotFoundError()):
            with pytest.raises(SystemExit):
                GitHubPagesManager("owner/repo")

        captured = capsys.readouterr()
        assert "GitHub CLI (gh) is not installed" in captured.err
        assert "cli.github.com" in captured.err


# =============================================================================
# TestRunGhApi
# =============================================================================


class TestRunGhApi:
    """Tests for _run_gh_api() API call execution."""

    def test_run_gh_api_get_success(
        self, mock_gh_auth_success: MagicMock, sample_pages_response: dict[str, Any]
    ) -> None:
        """Successful GET request returns status and response."""
        mock_response = MagicMock()
        mock_response.returncode = 0
        mock_response.stdout = json.dumps(sample_pages_response)
        mock_response.stderr = ""

        with patch("subprocess.run", side_effect=[mock_gh_auth_success, mock_response]):
            manager = GitHubPagesManager("owner/repo")
            status, response = manager._run_gh_api("/repos/owner/repo/pages")

        assert status == 0
        assert response == sample_pages_response

    def test_run_gh_api_post_success(
        self, mock_gh_auth_success: MagicMock, sample_pages_response: dict[str, Any]
    ) -> None:
        """Successful POST request returns status and response."""
        mock_response = MagicMock()
        mock_response.returncode = 0
        mock_response.stdout = json.dumps(sample_pages_response)
        mock_response.stderr = ""

        with patch("subprocess.run", side_effect=[mock_gh_auth_success, mock_response]):
            manager = GitHubPagesManager("owner/repo")
            status, response = manager._run_gh_api(
                "/repos/owner/repo/pages",
                method="POST",
                data={"build_type": "workflow"}
            )

        assert status == 0
        assert response == sample_pages_response

    def test_run_gh_api_put_success(
        self, mock_gh_auth_success: MagicMock, sample_pages_response: dict[str, Any]
    ) -> None:
        """Successful PUT request returns status and response."""
        mock_response = MagicMock()
        mock_response.returncode = 0
        mock_response.stdout = json.dumps(sample_pages_response)
        mock_response.stderr = ""

        with patch("subprocess.run", side_effect=[mock_gh_auth_success, mock_response]):
            manager = GitHubPagesManager("owner/repo")
            status, response = manager._run_gh_api(
                "/repos/owner/repo/pages",
                method="PUT",
                data={"https_enforced": True}
            )

        assert status == 0
        assert response == sample_pages_response

    def test_run_gh_api_delete_success(
        self, mock_gh_auth_success: MagicMock
    ) -> None:
        """Successful DELETE request returns status and empty response."""
        mock_response = MagicMock()
        mock_response.returncode = 0
        mock_response.stdout = ""
        mock_response.stderr = ""

        with patch("subprocess.run", side_effect=[mock_gh_auth_success, mock_response]):
            manager = GitHubPagesManager("owner/repo")
            status, response = manager._run_gh_api(
                "/repos/owner/repo/pages",
                method="DELETE"
            )

        assert status == 0
        assert response == {}

    def test_run_gh_api_includes_headers(
        self, mock_gh_auth_success: MagicMock
    ) -> None:
        """API call includes required headers."""
        mock_response = MagicMock()
        mock_response.returncode = 0
        mock_response.stdout = "{}"
        mock_response.stderr = ""

        with patch("subprocess.run", side_effect=[mock_gh_auth_success, mock_response]) as mock_run:
            manager = GitHubPagesManager("owner/repo")
            manager._run_gh_api("/repos/owner/repo/pages")

        # Get the second call (first is auth verification)
        api_call = mock_run.call_args_list[1]
        cmd = api_call[0][0]
        assert "-H" in cmd
        assert "Accept: application/vnd.github+json" in cmd
        assert "X-GitHub-Api-Version: 2022-11-28" in cmd

    def test_run_gh_api_encodes_boolean_data(
        self, mock_gh_auth_success: MagicMock
    ) -> None:
        """Boolean values are encoded as lowercase strings."""
        mock_response = MagicMock()
        mock_response.returncode = 0
        mock_response.stdout = "{}"
        mock_response.stderr = ""

        with patch("subprocess.run", side_effect=[mock_gh_auth_success, mock_response]) as mock_run:
            manager = GitHubPagesManager("owner/repo")
            manager._run_gh_api(
                "/repos/owner/repo/pages",
                method="PUT",
                data={"https_enforced": True}
            )

        api_call = mock_run.call_args_list[1]
        cmd = api_call[0][0]
        assert "https_enforced=true" in cmd

    def test_run_gh_api_encodes_nested_dict_data(
        self, mock_gh_auth_success: MagicMock
    ) -> None:
        """Nested dict values are encoded with bracket notation."""
        mock_response = MagicMock()
        mock_response.returncode = 0
        mock_response.stdout = "{}"
        mock_response.stderr = ""

        with patch("subprocess.run", side_effect=[mock_gh_auth_success, mock_response]) as mock_run:
            manager = GitHubPagesManager("owner/repo")
            manager._run_gh_api(
                "/repos/owner/repo/pages",
                method="POST",
                data={"source": {"branch": "main", "path": "/"}}
            )

        api_call = mock_run.call_args_list[1]
        cmd = api_call[0][0]
        assert "source[branch]=main" in cmd
        assert "source[path]=/" in cmd

    def test_run_gh_api_non_zero_status_with_check_false(
        self, mock_gh_auth_success: MagicMock
    ) -> None:
        """Non-zero status with check=False returns status and response."""
        mock_response = MagicMock()
        mock_response.returncode = 422
        mock_response.stdout = json.dumps({"message": "Already exists"})
        mock_response.stderr = ""

        with patch("subprocess.run", side_effect=[mock_gh_auth_success, mock_response]):
            manager = GitHubPagesManager("owner/repo")
            status, response = manager._run_gh_api(
                "/repos/owner/repo/pages",
                method="POST",
                check=False
            )

        assert status == 422
        assert response["message"] == "Already exists"

    def test_run_gh_api_non_zero_status_with_check_true_raises(
        self, mock_gh_auth_success: MagicMock
    ) -> None:
        """Non-zero status with check=True raises RuntimeError."""
        mock_response = MagicMock()
        mock_response.returncode = 400
        mock_response.stdout = ""
        mock_response.stderr = "Bad request"

        with patch("subprocess.run", side_effect=[mock_gh_auth_success, mock_response]):
            manager = GitHubPagesManager("owner/repo")

            with pytest.raises(RuntimeError, match="API call failed"):
                manager._run_gh_api("/repos/owner/repo/pages", check=True)

    def test_run_gh_api_invalid_json_response(
        self, mock_gh_auth_success: MagicMock
    ) -> None:
        """Non-JSON response is captured as raw content."""
        mock_response = MagicMock()
        mock_response.returncode = 0
        mock_response.stdout = "Not JSON at all"
        mock_response.stderr = ""

        with patch("subprocess.run", side_effect=[mock_gh_auth_success, mock_response]):
            manager = GitHubPagesManager("owner/repo")
            status, response = manager._run_gh_api("/repos/owner/repo/pages")

        assert status == 0
        assert response["raw"] == "Not JSON at all"

    def test_run_gh_api_empty_response(
        self, mock_gh_auth_success: MagicMock
    ) -> None:
        """Empty response is handled gracefully."""
        mock_response = MagicMock()
        mock_response.returncode = 0
        mock_response.stdout = ""
        mock_response.stderr = ""

        with patch("subprocess.run", side_effect=[mock_gh_auth_success, mock_response]):
            manager = GitHubPagesManager("owner/repo")
            status, response = manager._run_gh_api("/repos/owner/repo/pages")

        assert status == 0
        assert response == {}


# =============================================================================
# TestEnablePages
# =============================================================================


class TestEnablePages:
    """Tests for enable_pages() method."""

    def test_enable_pages_success_with_defaults(
        self, mock_gh_auth_success: MagicMock, sample_pages_response: dict[str, Any],
        capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Enable pages with default parameters."""
        mock_enable = MagicMock()
        mock_enable.returncode = 0
        mock_enable.stdout = json.dumps(sample_pages_response)
        mock_enable.stderr = ""

        mock_https = MagicMock()
        mock_https.returncode = 0
        mock_https.stdout = json.dumps(sample_pages_response)
        mock_https.stderr = ""

        with patch("subprocess.run", side_effect=[mock_gh_auth_success, mock_enable, mock_https]):
            manager = GitHubPagesManager("owner/repo")
            result = manager.enable_pages()

        assert result == sample_pages_response
        captured = capsys.readouterr()
        assert "Enabling GitHub Pages" in captured.out
        assert "GitHub Pages enabled successfully" in captured.out

    def test_enable_pages_with_custom_branch(
        self, mock_gh_auth_success: MagicMock, sample_pages_response: dict[str, Any]
    ) -> None:
        """Enable pages with custom branch."""
        mock_enable = MagicMock()
        mock_enable.returncode = 0
        mock_enable.stdout = json.dumps(sample_pages_response)
        mock_enable.stderr = ""

        mock_https = MagicMock()
        mock_https.returncode = 0
        mock_https.stdout = json.dumps(sample_pages_response)
        mock_https.stderr = ""

        with patch("subprocess.run", side_effect=[mock_gh_auth_success, mock_enable, mock_https]) as mock_run:
            manager = GitHubPagesManager("owner/repo")
            manager.enable_pages(branch="develop")

        # Check that branch parameter was passed
        api_call = mock_run.call_args_list[1]
        cmd = api_call[0][0]
        assert "source[branch]=develop" in cmd

    def test_enable_pages_with_docs_path(
        self, mock_gh_auth_success: MagicMock, sample_pages_response: dict[str, Any]
    ) -> None:
        """Enable pages with /docs path."""
        mock_enable = MagicMock()
        mock_enable.returncode = 0
        mock_enable.stdout = json.dumps(sample_pages_response)
        mock_enable.stderr = ""

        mock_https = MagicMock()
        mock_https.returncode = 0
        mock_https.stdout = json.dumps(sample_pages_response)
        mock_https.stderr = ""

        with patch("subprocess.run", side_effect=[mock_gh_auth_success, mock_enable, mock_https]) as mock_run:
            manager = GitHubPagesManager("owner/repo")
            manager.enable_pages(path="/docs")

        api_call = mock_run.call_args_list[1]
        cmd = api_call[0][0]
        assert "source[path]=/docs" in cmd

    def test_enable_pages_with_legacy_build_type(
        self, mock_gh_auth_success: MagicMock, sample_pages_response: dict[str, Any]
    ) -> None:
        """Enable pages with legacy (Jekyll) build type."""
        mock_enable = MagicMock()
        mock_enable.returncode = 0
        mock_enable.stdout = json.dumps(sample_pages_response)
        mock_enable.stderr = ""

        mock_https = MagicMock()
        mock_https.returncode = 0
        mock_https.stdout = json.dumps(sample_pages_response)
        mock_https.stderr = ""

        with patch("subprocess.run", side_effect=[mock_gh_auth_success, mock_enable, mock_https]) as mock_run:
            manager = GitHubPagesManager("owner/repo")
            manager.enable_pages(build_type="legacy")

        api_call = mock_run.call_args_list[1]
        cmd = api_call[0][0]
        assert "build_type=legacy" in cmd

    def test_enable_pages_without_https_enforcement(
        self, mock_gh_auth_success: MagicMock, sample_pages_response: dict[str, Any]
    ) -> None:
        """Enable pages without enforcing HTTPS."""
        mock_enable = MagicMock()
        mock_enable.returncode = 0
        mock_enable.stdout = json.dumps(sample_pages_response)
        mock_enable.stderr = ""

        with patch("subprocess.run", side_effect=[mock_gh_auth_success, mock_enable]):
            manager = GitHubPagesManager("owner/repo")
            result = manager.enable_pages(https_enforced=False)

        # Should only have 2 API calls (auth + enable), no PUT for HTTPS
        assert result == sample_pages_response

    def test_enable_pages_already_enabled_422_status(
        self, mock_gh_auth_success: MagicMock, sample_pages_response: dict[str, Any],
        capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Handle 422 status when pages already enabled."""
        mock_enable = MagicMock()
        mock_enable.returncode = 422
        mock_enable.stdout = json.dumps({"message": "Already exists"})
        mock_enable.stderr = ""

        mock_check = MagicMock()
        mock_check.returncode = 0
        mock_check.stdout = json.dumps(sample_pages_response)
        mock_check.stderr = ""

        with patch("subprocess.run", side_effect=[mock_gh_auth_success, mock_enable, mock_check]):
            manager = GitHubPagesManager("owner/repo")
            result = manager.enable_pages()

        # Should return the status response
        assert result == sample_pages_response
        captured = capsys.readouterr()
        assert "may already be enabled" in captured.out

    def test_enable_pages_api_error(
        self, mock_gh_auth_success: MagicMock, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Handle API errors gracefully."""
        mock_enable = MagicMock()
        mock_enable.returncode = 403
        mock_enable.stdout = json.dumps({"message": "Access Denied"})
        mock_enable.stderr = ""

        with patch("subprocess.run", side_effect=[mock_gh_auth_success, mock_enable]):
            manager = GitHubPagesManager("owner/repo")
            result = manager.enable_pages()

        assert result["message"] == "Access Denied"
        captured = capsys.readouterr()
        assert "Failed to enable Pages" in captured.err


# =============================================================================
# TestCheckStatus
# =============================================================================


class TestCheckStatus:
    """Tests for check_status() method."""

    def test_check_status_enabled(
        self, mock_gh_auth_success: MagicMock, sample_pages_response: dict[str, Any],
        capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Check status when pages are enabled."""
        mock_response = MagicMock()
        mock_response.returncode = 0
        mock_response.stdout = json.dumps(sample_pages_response)
        mock_response.stderr = ""

        with patch("subprocess.run", side_effect=[mock_gh_auth_success, mock_response]):
            manager = GitHubPagesManager("owner/repo")
            result = manager.check_status()

        assert result == sample_pages_response
        captured = capsys.readouterr()
        assert "GitHub Pages is enabled" in captured.out
        assert sample_pages_response["html_url"] in captured.out

    def test_check_status_disabled(
        self, mock_gh_auth_success: MagicMock,
        capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Check status when pages are disabled."""
        mock_response = MagicMock()
        mock_response.returncode = 404
        mock_response.stdout = json.dumps({"message": "Not Found"})
        mock_response.stderr = ""

        with patch("subprocess.run", side_effect=[mock_gh_auth_success, mock_response]):
            manager = GitHubPagesManager("owner/repo")
            result = manager.check_status()

        captured = capsys.readouterr()
        assert "GitHub Pages is not enabled" in captured.err

    def test_check_status_displays_configuration(
        self, mock_gh_auth_success: MagicMock, sample_pages_response: dict[str, Any],
        capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Check status displays all configuration details."""
        mock_response = MagicMock()
        mock_response.returncode = 0
        mock_response.stdout = json.dumps(sample_pages_response)
        mock_response.stderr = ""

        with patch("subprocess.run", side_effect=[mock_gh_auth_success, mock_response]):
            manager = GitHubPagesManager("owner/repo")
            manager.check_status()

        captured = capsys.readouterr()
        assert "URL:" in captured.out
        assert "Status:" in captured.out
        assert "Build type:" in captured.out
        assert "Source branch:" in captured.out
        assert "Source path:" in captured.out
        assert "HTTPS enforced:" in captured.out

    def test_check_status_with_custom_domain(
        self, mock_gh_auth_success: MagicMock, sample_pages_response: dict[str, Any],
        capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Check status displays custom domain when present."""
        response = sample_pages_response.copy()
        response["cname"] = "example.com"

        mock_response = MagicMock()
        mock_response.returncode = 0
        mock_response.stdout = json.dumps(response)
        mock_response.stderr = ""

        with patch("subprocess.run", side_effect=[mock_gh_auth_success, mock_response]):
            manager = GitHubPagesManager("owner/repo")
            manager.check_status()

        captured = capsys.readouterr()
        assert "Custom domain:" in captured.out
        assert "example.com" in captured.out


# =============================================================================
# TestTriggerRebuild
# =============================================================================


class TestTriggerRebuild:
    """Tests for trigger_rebuild() method."""

    def test_trigger_rebuild_success(
        self, mock_gh_auth_success: MagicMock, sample_build_response: dict[str, Any],
        capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Trigger rebuild successfully."""
        mock_response = MagicMock()
        mock_response.returncode = 0
        mock_response.stdout = json.dumps(sample_build_response)
        mock_response.stderr = ""

        with patch("subprocess.run", side_effect=[mock_gh_auth_success, mock_response]):
            manager = GitHubPagesManager("owner/repo")
            result = manager.trigger_rebuild()

        assert result == sample_build_response
        captured = capsys.readouterr()
        assert "Build triggered successfully" in captured.out

    def test_trigger_rebuild_failure(
        self, mock_gh_auth_success: MagicMock,
        capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Handle rebuild trigger failure."""
        mock_response = MagicMock()
        mock_response.returncode = 403
        mock_response.stdout = json.dumps({"message": "Access Denied"})
        mock_response.stderr = ""

        with patch("subprocess.run", side_effect=[mock_gh_auth_success, mock_response]):
            manager = GitHubPagesManager("owner/repo")
            result = manager.trigger_rebuild()

        captured = capsys.readouterr()
        assert "Failed to trigger build" in captured.err

    def test_trigger_rebuild_displays_status(
        self, mock_gh_auth_success: MagicMock, sample_build_response: dict[str, Any],
        capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Trigger rebuild displays build status and URL."""
        mock_response = MagicMock()
        mock_response.returncode = 0
        mock_response.stdout = json.dumps(sample_build_response)
        mock_response.stderr = ""

        with patch("subprocess.run", side_effect=[mock_gh_auth_success, mock_response]):
            manager = GitHubPagesManager("owner/repo")
            manager.trigger_rebuild()

        captured = capsys.readouterr()
        assert "Status:" in captured.out
        assert "Build URL:" in captured.out


# =============================================================================
# TestGetLatestBuild
# =============================================================================


class TestGetLatestBuild:
    """Tests for get_latest_build() method."""

    def test_get_latest_build_success(
        self, mock_gh_auth_success: MagicMock, sample_build_response: dict[str, Any],
        capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Get latest build successfully."""
        mock_response = MagicMock()
        mock_response.returncode = 0
        mock_response.stdout = json.dumps(sample_build_response)
        mock_response.stderr = ""

        with patch("subprocess.run", side_effect=[mock_gh_auth_success, mock_response]):
            manager = GitHubPagesManager("owner/repo")
            result = manager.get_latest_build()

        assert result == sample_build_response
        captured = capsys.readouterr()
        assert "Latest build:" in captured.out

    def test_get_latest_build_with_error(
        self, mock_gh_auth_success: MagicMock, sample_build_error_response: dict[str, Any],
        capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Get latest build with error information."""
        mock_response = MagicMock()
        mock_response.returncode = 0
        mock_response.stdout = json.dumps(sample_build_error_response)
        mock_response.stderr = ""

        with patch("subprocess.run", side_effect=[mock_gh_auth_success, mock_response]):
            manager = GitHubPagesManager("owner/repo")
            result = manager.get_latest_build()

        assert result == sample_build_error_response
        captured = capsys.readouterr()
        assert "Error:" in captured.out
        assert "Permission denied" in captured.out

    def test_get_latest_build_displays_details(
        self, mock_gh_auth_success: MagicMock, sample_build_response: dict[str, Any],
        capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Get latest build displays all available details."""
        mock_response = MagicMock()
        mock_response.returncode = 0
        mock_response.stdout = json.dumps(sample_build_response)
        mock_response.stderr = ""

        with patch("subprocess.run", side_effect=[mock_gh_auth_success, mock_response]):
            manager = GitHubPagesManager("owner/repo")
            manager.get_latest_build()

        captured = capsys.readouterr()
        assert "Status:" in captured.out
        assert "Commit:" in captured.out
        assert "Created:" in captured.out

    def test_get_latest_build_no_error_field(
        self, mock_gh_auth_success: MagicMock, sample_build_response: dict[str, Any],
        capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Get latest build handles missing error field."""
        mock_response = MagicMock()
        mock_response.returncode = 0
        mock_response.stdout = json.dumps(sample_build_response)
        mock_response.stderr = ""

        with patch("subprocess.run", side_effect=[mock_gh_auth_success, mock_response]):
            manager = GitHubPagesManager("owner/repo")
            manager.get_latest_build()

        captured = capsys.readouterr()
        # Should not have Error line if error not in response
        assert "Error:" not in captured.out


# =============================================================================
# TestCreateWorkflowFile
# =============================================================================


class TestCreateWorkflowFile:
    """Tests for create_workflow_file() function."""

    def test_create_workflow_file_default_path(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str],
        monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Create workflow file at default path."""
        monkeypatch.chdir(tmp_path)
        create_workflow_file()

        workflow_file = tmp_path / ".github" / "workflows" / "pages.yml"
        assert workflow_file.exists()

        file_content = workflow_file.read_text()
        assert "Deploy to GitHub Pages" in file_content
        assert "deploy:" in file_content

    def test_create_workflow_file_custom_path(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Create workflow file at custom path."""
        custom_path = tmp_path / "custom" / "workflows" / "deploy.yml"
        create_workflow_file(str(custom_path))

        assert custom_path.exists()

    def test_create_workflow_file_creates_parent_directories(
        self, tmp_path: Path
    ) -> None:
        """Parent directories are created if they don't exist."""
        custom_path = tmp_path / "a" / "b" / "c" / "d" / "workflow.yml"
        create_workflow_file(str(custom_path))

        assert custom_path.parent.exists()
        assert custom_path.exists()

    def test_create_workflow_file_content_structure(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Created workflow file has correct structure."""
        monkeypatch.chdir(tmp_path)
        create_workflow_file()

        workflow_file = tmp_path / ".github" / "workflows" / "pages.yml"
        file_content = workflow_file.read_text()

        assert "name: Deploy to GitHub Pages" in file_content
        assert "on:" in file_content
        assert "push:" in file_content
        assert "workflow_dispatch:" in file_content
        assert "permissions:" in file_content
        assert "jobs:" in file_content
        assert "build:" in file_content
        assert "deploy:" in file_content

    def test_create_workflow_file_includes_permissions(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Workflow includes correct permissions."""
        monkeypatch.chdir(tmp_path)
        create_workflow_file()

        workflow_file = tmp_path / ".github" / "workflows" / "pages.yml"
        file_content = workflow_file.read_text()

        assert "contents: read" in file_content
        assert "pages: write" in file_content
        assert "id-token: write" in file_content

    def test_create_workflow_file_prints_success_message(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str],
        monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Success message is printed after creation."""
        monkeypatch.chdir(tmp_path)
        create_workflow_file()

        captured = capsys.readouterr()
        assert "Created workflow file at:" in captured.out

    def test_create_workflow_file_overwrites_existing(
        self, tmp_path: Path
    ) -> None:
        """Creating workflow file overwrites existing file."""
        custom_path = tmp_path / "workflow.yml"
        custom_path.write_text("OLD CONTENT")

        create_workflow_file(str(custom_path))

        content = custom_path.read_text()
        assert "OLD CONTENT" not in content
        assert "Deploy to GitHub Pages" in content


# =============================================================================
# TestMain
# =============================================================================


class TestMain:
    """Tests for main() CLI function."""

    def test_main_no_args_prints_help(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Running with no arguments prints help."""
        with patch("sys.argv", ["gh_pages_deploy.py"]):
            with pytest.raises(SystemExit) as exc_info:
                main()

        assert exc_info.value.code == 1
        captured = capsys.readouterr()
        assert "usage:" in captured.out

    def test_main_enable_command(
        self, mock_gh_auth_success: MagicMock, sample_pages_response: dict[str, Any],
        capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Enable command works correctly."""
        mock_enable = MagicMock()
        mock_enable.returncode = 0
        mock_enable.stdout = json.dumps(sample_pages_response)
        mock_enable.stderr = ""

        mock_https = MagicMock()
        mock_https.returncode = 0
        mock_https.stdout = json.dumps(sample_pages_response)
        mock_https.stderr = ""

        with patch(
            "sys.argv",
            ["gh_pages_deploy.py", "enable", "owner/repo"]
        ):
            with patch("subprocess.run", side_effect=[mock_gh_auth_success, mock_enable, mock_https]):
                main()  # Success: no exception raised

        captured = capsys.readouterr()
        assert "GitHub Pages enabled successfully" in captured.out

    def test_main_status_command(
        self, mock_gh_auth_success: MagicMock, sample_pages_response: dict[str, Any],
        capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Status command works correctly."""
        mock_response = MagicMock()
        mock_response.returncode = 0
        mock_response.stdout = json.dumps(sample_pages_response)
        mock_response.stderr = ""

        with patch(
            "sys.argv",
            ["gh_pages_deploy.py", "status", "owner/repo"]
        ):
            with patch("subprocess.run", side_effect=[mock_gh_auth_success, mock_response]):
                main()  # Success: no exception raised

        captured = capsys.readouterr()
        assert "GitHub Pages is enabled" in captured.out

    def test_main_rebuild_command(
        self, mock_gh_auth_success: MagicMock, sample_build_response: dict[str, Any],
        capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Rebuild command works correctly."""
        mock_response = MagicMock()
        mock_response.returncode = 0
        mock_response.stdout = json.dumps(sample_build_response)
        mock_response.stderr = ""

        with patch(
            "sys.argv",
            ["gh_pages_deploy.py", "rebuild", "owner/repo"]
        ):
            with patch("subprocess.run", side_effect=[mock_gh_auth_success, mock_response]):
                main()  # Success: no exception raised

        captured = capsys.readouterr()
        assert "Build triggered successfully" in captured.out

    def test_main_create_workflow_command(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Create-workflow command works correctly."""
        monkeypatch.chdir(tmp_path)
        with patch(
            "sys.argv",
            ["gh_pages_deploy.py", "create-workflow"]
        ):
            main()  # Success: no exception raised

        workflow_file = tmp_path / ".github" / "workflows" / "pages.yml"
        assert workflow_file.exists()

    def test_main_enable_with_custom_branch(
        self, mock_gh_auth_success: MagicMock, sample_pages_response: dict[str, Any]
    ) -> None:
        """Enable command with custom branch parameter."""
        mock_enable = MagicMock()
        mock_enable.returncode = 0
        mock_enable.stdout = json.dumps(sample_pages_response)
        mock_enable.stderr = ""

        mock_https = MagicMock()
        mock_https.returncode = 0
        mock_https.stdout = json.dumps(sample_pages_response)
        mock_https.stderr = ""

        with patch(
            "sys.argv",
            ["gh_pages_deploy.py", "enable", "owner/repo", "--branch", "develop"]
        ):
            with patch("subprocess.run", side_effect=[mock_gh_auth_success, mock_enable, mock_https]) as mock_run:
                main()  # Success: no exception raised

        api_call = mock_run.call_args_list[1]
        cmd = api_call[0][0]
        assert "source[branch]=develop" in cmd

    def test_main_enable_with_docs_path(
        self, mock_gh_auth_success: MagicMock, sample_pages_response: dict[str, Any]
    ) -> None:
        """Enable command with /docs path parameter."""
        mock_enable = MagicMock()
        mock_enable.returncode = 0
        mock_enable.stdout = json.dumps(sample_pages_response)
        mock_enable.stderr = ""

        mock_https = MagicMock()
        mock_https.returncode = 0
        mock_https.stdout = json.dumps(sample_pages_response)
        mock_https.stderr = ""

        with patch(
            "sys.argv",
            ["gh_pages_deploy.py", "enable", "owner/repo", "--path", "/docs"]
        ):
            with patch("subprocess.run", side_effect=[mock_gh_auth_success, mock_enable, mock_https]) as mock_run:
                main()  # Success: no exception raised

        api_call = mock_run.call_args_list[1]
        cmd = api_call[0][0]
        assert "source[path]=/docs" in cmd

    def test_main_enable_with_no_https(
        self, mock_gh_auth_success: MagicMock, sample_pages_response: dict[str, Any],
        capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Enable command with --no-https flag."""
        mock_enable = MagicMock()
        mock_enable.returncode = 0
        mock_enable.stdout = json.dumps(sample_pages_response)
        mock_enable.stderr = ""

        with patch(
            "sys.argv",
            ["gh_pages_deploy.py", "enable", "owner/repo", "--no-https"]
        ):
            with patch("subprocess.run", side_effect=[mock_gh_auth_success, mock_enable]):
                main()  # Success: no exception raised

        captured = capsys.readouterr()
        assert "GitHub Pages enabled successfully" in captured.out

    def test_main_enable_prints_workflow_tip(
        self, mock_gh_auth_success: MagicMock, sample_pages_response: dict[str, Any],
        capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Enable command prints workflow creation tip."""
        mock_enable = MagicMock()
        mock_enable.returncode = 0
        mock_enable.stdout = json.dumps(sample_pages_response)
        mock_enable.stderr = ""

        mock_https = MagicMock()
        mock_https.returncode = 0
        mock_https.stdout = json.dumps(sample_pages_response)
        mock_https.stderr = ""

        with patch(
            "sys.argv",
            ["gh_pages_deploy.py", "enable", "owner/repo"]
        ):
            with patch("subprocess.run", side_effect=[mock_gh_auth_success, mock_enable, mock_https]):
                main()  # Success: no exception raised

        captured = capsys.readouterr()
        assert "create-workflow" in captured.out

    def test_main_status_with_build_info(
        self, mock_gh_auth_success: MagicMock, sample_pages_response: dict[str, Any],
        sample_build_response: dict[str, Any], capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Status command with --build-info flag."""
        mock_status = MagicMock()
        mock_status.returncode = 0
        mock_status.stdout = json.dumps(sample_pages_response)
        mock_status.stderr = ""

        mock_build = MagicMock()
        mock_build.returncode = 0
        mock_build.stdout = json.dumps(sample_build_response)
        mock_build.stderr = ""

        with patch(
            "sys.argv",
            ["gh_pages_deploy.py", "status", "owner/repo", "--build-info"]
        ):
            with patch("subprocess.run", side_effect=[mock_gh_auth_success, mock_status, mock_build]):
                main()  # Success: no exception raised

        captured = capsys.readouterr()
        assert "Latest build:" in captured.out

    def test_main_error_handling_prints_error(
        self, mock_gh_auth_failure: MagicMock, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Main function handles errors gracefully."""
        with patch(
            "sys.argv",
            ["gh_pages_deploy.py", "enable", "owner/repo"]
        ):
            with patch("subprocess.run", return_value=mock_gh_auth_failure):
                with pytest.raises(SystemExit) as exc_info:
                    main()

        assert exc_info.value.code == 1

    def test_main_enable_with_legacy_build_type(
        self, mock_gh_auth_success: MagicMock, sample_pages_response: dict[str, Any]
    ) -> None:
        """Enable command with --build-type legacy parameter."""
        mock_enable = MagicMock()
        mock_enable.returncode = 0
        mock_enable.stdout = json.dumps(sample_pages_response)
        mock_enable.stderr = ""

        mock_https = MagicMock()
        mock_https.returncode = 0
        mock_https.stdout = json.dumps(sample_pages_response)
        mock_https.stderr = ""

        with patch(
            "sys.argv",
            ["gh_pages_deploy.py", "enable", "owner/repo", "--build-type", "legacy"]
        ):
            with patch("subprocess.run", side_effect=[mock_gh_auth_success, mock_enable, mock_https]) as mock_run:
                main()  # Success: no exception raised

        api_call = mock_run.call_args_list[1]
        cmd = api_call[0][0]
        assert "build_type=legacy" in cmd



# =============================================================================
# TestEdgeCases
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and error scenarios."""

    def test_repo_format_with_multiple_slashes(
        self, mock_gh_auth_success: MagicMock
    ) -> None:
        """Repository format with owner/repo is parsed correctly."""
        with patch("subprocess.run", return_value=mock_gh_auth_success):
            manager = GitHubPagesManager("my-org/my-repo")

        assert manager.repo == "my-org/my-repo"

    def test_api_response_with_null_values(
        self, mock_gh_auth_success: MagicMock, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """API response with null values is handled gracefully."""
        response = {
            "html_url": "https://example.com",
            "status": "built",
            "build_type": None,
            "cname": None
        }

        mock_api = MagicMock()
        mock_api.returncode = 0
        mock_api.stdout = json.dumps(response)
        mock_api.stderr = ""

        with patch("subprocess.run", side_effect=[mock_gh_auth_success, mock_api]):
            manager = GitHubPagesManager("owner/repo")
            manager.check_status()

        captured = capsys.readouterr()
        assert "N/A" in captured.out

    def test_subprocess_timeout_during_api_call(
        self, mock_gh_auth_success: MagicMock
    ) -> None:
        """Handle subprocess timeout gracefully."""
        with patch("subprocess.run", side_effect=[mock_gh_auth_success, subprocess.TimeoutExpired("gh", 30)]):
            manager = GitHubPagesManager("owner/repo")

            with pytest.raises(subprocess.TimeoutExpired):
                manager._run_gh_api("/repos/owner/repo/pages")

    def test_very_long_repo_name(
        self, mock_gh_auth_success: MagicMock
    ) -> None:
        """Handle very long repository names."""
        long_name = "a" * 100 + "/" + "b" * 100
        with patch("subprocess.run", return_value=mock_gh_auth_success):
            manager = GitHubPagesManager(long_name)

        assert manager.repo == long_name

    def test_special_characters_in_repo_name(
        self, mock_gh_auth_success: MagicMock
    ) -> None:
        """Handle special characters in repository name."""
        with patch("subprocess.run", return_value=mock_gh_auth_success):
            manager = GitHubPagesManager("my-org/my-repo.name")

        assert manager.repo == "my-org/my-repo.name"

    def test_connection_refused_during_api_call(
        self, mock_gh_auth_success: MagicMock
    ) -> None:
        """Handle connection refused errors gracefully."""
        error = subprocess.CalledProcessError(1, "gh")
        error.stderr = "Could not connect"
        
        with patch("subprocess.run", side_effect=[mock_gh_auth_success, error]):
            manager = GitHubPagesManager("owner/repo")

            with pytest.raises(RuntimeError, match="Failed to execute"):
                manager._run_gh_api("/repos/owner/repo/pages", check=True)

    def test_empty_repo_name(
        self, mock_gh_auth_success: MagicMock
    ) -> None:
        """Handle empty repository name."""
        with patch("subprocess.run", return_value=mock_gh_auth_success):
            manager = GitHubPagesManager("")

        assert manager.repo == ""

    def test_api_response_with_missing_source(
        self, mock_gh_auth_success: MagicMock, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """API response with missing source field is handled gracefully."""
        response = {
            "html_url": "https://example.com",
            "status": "built",
            "build_type": "workflow"
        }

        mock_api = MagicMock()
        mock_api.returncode = 0
        mock_api.stdout = json.dumps(response)
        mock_api.stderr = ""

        with patch("subprocess.run", side_effect=[mock_gh_auth_success, mock_api]):
            manager = GitHubPagesManager("owner/repo")
            manager.check_status()

        captured = capsys.readouterr()
        assert "Source branch: N/A" in captured.out



# =============================================================================
# TestIntegration
# =============================================================================


class TestIntegration:
    """Integration tests combining multiple operations."""

    def test_full_pages_workflow_enable_and_check(
        self, mock_gh_auth_success: MagicMock, sample_pages_response: dict[str, Any]
    ) -> None:
        """Full workflow: enable pages, then check status."""
        mock_enable = MagicMock()
        mock_enable.returncode = 0
        mock_enable.stdout = json.dumps(sample_pages_response)
        mock_enable.stderr = ""

        mock_https = MagicMock()
        mock_https.returncode = 0
        mock_https.stdout = json.dumps(sample_pages_response)
        mock_https.stderr = ""

        mock_status = MagicMock()
        mock_status.returncode = 0
        mock_status.stdout = json.dumps(sample_pages_response)
        mock_status.stderr = ""

        with patch("subprocess.run", side_effect=[mock_gh_auth_success, mock_enable, mock_https, mock_status]):
            manager = GitHubPagesManager("owner/repo")
            enable_result = manager.enable_pages()
            status_result = manager.check_status()

        assert enable_result == sample_pages_response
        assert status_result == sample_pages_response

    def test_full_workflow_enable_rebuild_check(
        self, mock_gh_auth_success: MagicMock, sample_pages_response: dict[str, Any],
        sample_build_response: dict[str, Any]
    ) -> None:
        """Full workflow: enable, trigger rebuild, and check latest build."""
        mock_enable = MagicMock()
        mock_enable.returncode = 0
        mock_enable.stdout = json.dumps(sample_pages_response)
        mock_enable.stderr = ""

        mock_https = MagicMock()
        mock_https.returncode = 0
        mock_https.stdout = json.dumps(sample_pages_response)
        mock_https.stderr = ""

        mock_rebuild = MagicMock()
        mock_rebuild.returncode = 0
        mock_rebuild.stdout = json.dumps(sample_build_response)
        mock_rebuild.stderr = ""

        mock_latest = MagicMock()
        mock_latest.returncode = 0
        mock_latest.stdout = json.dumps(sample_build_response)
        mock_latest.stderr = ""

        with patch("subprocess.run", side_effect=[mock_gh_auth_success, mock_enable, mock_https, mock_rebuild, mock_latest]):
            manager = GitHubPagesManager("owner/repo")
            manager.enable_pages()
            rebuild_result = manager.trigger_rebuild()
            build_result = manager.get_latest_build()

        assert rebuild_result == sample_build_response
        assert build_result == sample_build_response


# =============================================================================
# Main Entry Point
# =============================================================================


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
