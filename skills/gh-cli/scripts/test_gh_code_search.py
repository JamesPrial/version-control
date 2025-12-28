"""
Comprehensive test suite for gh_code_search.py GitHub code search wrapper.

Tests cover:
- Command building with various filter combinations
- GitHub API execution and error handling
- Result filtering (forks, private, min matches)
- Result sorting (matches, repo, path)
- Output formatting (JSON, pretty, summary)
- Error handling (rate limits, timeouts, JSON decode errors)
- Main function integration tests
"""
from __future__ import annotations

import argparse
import json
import subprocess
from typing import Any, TypedDict
from unittest.mock import MagicMock, patch

import pytest

from gh_code_search import (
    GHSearchError,
    build_gh_command,
    execute_search,
    filter_results,
    format_json,
    format_pretty,
    format_summary,
    main,
    sort_results,
)


# =============================================================================
# Type Definitions
# =============================================================================


class TextMatch(TypedDict):
    """Structure of a text match in GitHub search results."""

    fragment: str
    indices: list[int]


class Repository(TypedDict, total=False):
    """Structure of repository info in GitHub search results."""

    nameWithOwner: str
    isFork: bool
    isPrivate: bool


class SearchResult(TypedDict, total=False):
    """Structure of a GitHub code search result."""

    path: str
    repository: Repository
    sha: str
    textMatches: list[TextMatch]
    url: str


# =============================================================================
# Test Data / Fixtures
# =============================================================================


@pytest.fixture
def sample_search_result() -> SearchResult:
    """Standard GitHub search result for testing."""
    return {
        "path": "src/main.py",
        "repository": {
            "nameWithOwner": "octocat/Hello-World",
            "isFork": False,
            "isPrivate": False,
        },
        "sha": "abc123def456",
        "textMatches": [
            {"fragment": "def hello():", "indices": [0, 10]},
            {"fragment": "    print('Hello')", "indices": [20, 40]},
        ],
        "url": "https://github.com/octocat/Hello-World/blob/main/src/main.py",
    }


@pytest.fixture
def sample_results_list(sample_search_result: SearchResult) -> list[SearchResult]:
    """Standard list of GitHub search results for testing."""
    return [
        sample_search_result,
        {
            "path": "test/test_main.py",
            "repository": {
                "nameWithOwner": "octocat/Hello-World",
                "isFork": False,
                "isPrivate": False,
            },
            "sha": "def789ghi012",
            "textMatches": [
                {"fragment": "def test_hello():", "indices": [0, 16]},
            ],
            "url": "https://github.com/octocat/Hello-World/blob/main/test/test_main.py",
        },
        {
            "path": "hello.js",
            "repository": {
                "nameWithOwner": "microsoft/vscode",
                "isFork": True,
                "isPrivate": False,
            },
            "sha": "jkl345mno678",
            "textMatches": [
                {"fragment": "console.log('Hello');", "indices": [0, 21]},
            ],
            "url": "https://github.com/microsoft/vscode/blob/main/hello.js",
        },
    ]


@pytest.fixture
def basic_args() -> argparse.Namespace:
    """Basic command-line arguments."""
    return argparse.Namespace(
        query="hello",
        limit=30,
        language=None,
        filename=None,
        extension=None,
        repo=None,
        owner=None,
        match=None,
        size=None,
        exclude_forks=False,
        exclude_private=False,
        min_matches=None,
        output="pretty",
        sort_by=None,
    )


def make_minimal_result(
    path: str = "file.py",
    repo_name: str = "org/repo",
    is_fork: bool = False,
    is_private: bool = False,
    text_matches: list[TextMatch] | None = None,
    url: str = "https://example.com",
) -> SearchResult:
    """Factory function for creating minimal search results in tests."""
    return {
        "path": path,
        "repository": {
            "nameWithOwner": repo_name,
            "isFork": is_fork,
            "isPrivate": is_private,
        },
        "textMatches": text_matches or [],
        "url": url,
    }


def make_filter_args(
    exclude_forks: bool = False,
    exclude_private: bool = False,
    min_matches: int | None = None,
) -> argparse.Namespace:
    """Factory for creating filter-only argument namespaces."""
    return argparse.Namespace(
        exclude_forks=exclude_forks,
        exclude_private=exclude_private,
        min_matches=min_matches,
    )


# =============================================================================
# TestBuildGhCommand
# =============================================================================


class TestBuildGhCommand:
    """Tests for build_gh_command() function."""

    def test_build_basic_command(self, basic_args: argparse.Namespace) -> None:
        """Build basic gh search code command with minimal arguments."""
        cmd = build_gh_command(basic_args)

        assert cmd[0] == "gh"
        assert cmd[1] == "search"
        assert cmd[2] == "code"
        assert cmd[3] == "hello"
        assert "--json" in cmd
        assert "path,repository,sha,textMatches,url" in cmd

    def test_build_command_with_limit(self, basic_args: argparse.Namespace) -> None:
        """Build command with custom limit."""
        basic_args.limit = 100
        cmd = build_gh_command(basic_args)

        assert "--limit" in cmd
        limit_idx = cmd.index("--limit")
        assert cmd[limit_idx + 1] == "100"

    def test_build_command_with_language(self, basic_args: argparse.Namespace) -> None:
        """Build command with language filter."""
        basic_args.language = "python"
        cmd = build_gh_command(basic_args)

        assert "--language" in cmd
        lang_idx = cmd.index("--language")
        assert cmd[lang_idx + 1] == "python"

    def test_build_command_with_filename(self, basic_args: argparse.Namespace) -> None:
        """Build command with filename filter."""
        basic_args.filename = "README.md"
        cmd = build_gh_command(basic_args)

        assert "--filename" in cmd
        fn_idx = cmd.index("--filename")
        assert cmd[fn_idx + 1] == "README.md"

    def test_build_command_with_extension(
        self, basic_args: argparse.Namespace
    ) -> None:
        """Build command with extension filter."""
        basic_args.extension = "rs"
        cmd = build_gh_command(basic_args)

        assert "--extension" in cmd
        ext_idx = cmd.index("--extension")
        assert cmd[ext_idx + 1] == "rs"

    def test_build_command_with_single_repo(
        self, basic_args: argparse.Namespace
    ) -> None:
        """Build command with single repository filter."""
        basic_args.repo = ["microsoft/vscode"]
        cmd = build_gh_command(basic_args)

        assert "--repo" in cmd
        repo_idx = cmd.index("--repo")
        assert cmd[repo_idx + 1] == "microsoft/vscode"

    def test_build_command_with_multiple_repos(
        self, basic_args: argparse.Namespace
    ) -> None:
        """Build command with multiple repository filters."""
        basic_args.repo = ["microsoft/vscode", "torvalds/linux"]
        cmd = build_gh_command(basic_args)

        assert cmd.count("--repo") == 2
        assert "microsoft/vscode" in cmd
        assert "torvalds/linux" in cmd

    def test_build_command_with_single_owner(
        self, basic_args: argparse.Namespace
    ) -> None:
        """Build command with single owner filter."""
        basic_args.owner = ["torvalds"]
        cmd = build_gh_command(basic_args)

        assert "--owner" in cmd
        owner_idx = cmd.index("--owner")
        assert cmd[owner_idx + 1] == "torvalds"

    def test_build_command_with_multiple_owners(
        self, basic_args: argparse.Namespace
    ) -> None:
        """Build command with multiple owner filters."""
        basic_args.owner = ["google", "facebook"]
        cmd = build_gh_command(basic_args)

        assert cmd.count("--owner") == 2
        assert "google" in cmd
        assert "facebook" in cmd

    def test_build_command_with_match_file(
        self, basic_args: argparse.Namespace
    ) -> None:
        """Build command with match type 'file'."""
        basic_args.match = "file"
        cmd = build_gh_command(basic_args)

        assert "--match" in cmd
        match_idx = cmd.index("--match")
        assert cmd[match_idx + 1] == "file"

    def test_build_command_with_match_content(
        self, basic_args: argparse.Namespace
    ) -> None:
        """Build command with match type 'content'."""
        basic_args.match = "content"
        cmd = build_gh_command(basic_args)

        assert "--match" in cmd
        match_idx = cmd.index("--match")
        assert cmd[match_idx + 1] == "content"

    def test_build_command_with_size(self, basic_args: argparse.Namespace) -> None:
        """Build command with size filter."""
        basic_args.size = "10..100"
        cmd = build_gh_command(basic_args)

        assert "--size" in cmd
        size_idx = cmd.index("--size")
        assert cmd[size_idx + 1] == "10..100"

    def test_build_command_with_all_filters(
        self, basic_args: argparse.Namespace
    ) -> None:
        """Build command with all filters combined."""
        basic_args.limit = 50
        basic_args.language = "typescript"
        basic_args.filename = "*.test.ts"
        basic_args.extension = "ts"
        basic_args.repo = ["microsoft/vscode", "angular/angular"]
        basic_args.owner = ["google"]
        basic_args.match = "content"
        basic_args.size = "5..50"

        cmd = build_gh_command(basic_args)

        assert "--limit" in cmd
        assert "--language" in cmd
        assert "--filename" in cmd
        assert "--extension" in cmd
        assert cmd.count("--repo") == 2
        assert "--owner" in cmd
        assert "--match" in cmd
        assert "--size" in cmd

    def test_build_command_none_values_skipped(
        self, basic_args: argparse.Namespace
    ) -> None:
        """None values should not add flags to command."""
        cmd = build_gh_command(basic_args)
        assert None not in cmd

    @pytest.mark.parametrize("limit", [1, 10, 30, 100, 1000])
    def test_build_command_various_limits(
        self, basic_args: argparse.Namespace, limit: int
    ) -> None:
        """Build command with various limit values."""
        basic_args.limit = limit
        cmd = build_gh_command(basic_args)

        assert "--limit" in cmd
        limit_idx = cmd.index("--limit")
        assert cmd[limit_idx + 1] == str(limit)

    @pytest.mark.parametrize(
        "language", ["python", "javascript", "go", "rust", "typescript"]
    )
    def test_build_command_various_languages(
        self, basic_args: argparse.Namespace, language: str
    ) -> None:
        """Build command with various programming languages."""
        basic_args.language = language
        cmd = build_gh_command(basic_args)

        assert "--language" in cmd
        lang_idx = cmd.index("--language")
        assert cmd[lang_idx + 1] == language


# =============================================================================
# TestExecuteSearch
# =============================================================================


class TestExecuteSearch:
    """Tests for execute_search() function."""

    def test_execute_search_success(
        self, sample_results_list: list[SearchResult]
    ) -> None:
        """Successfully execute search and parse JSON output."""
        mock_result = MagicMock()
        mock_result.stdout = json.dumps(sample_results_list)

        with patch("subprocess.run", return_value=mock_result):
            results = execute_search(["gh", "search", "code", "test"])

        assert len(results) == 3
        assert results[0]["path"] == "src/main.py"
        assert results[1]["path"] == "test/test_main.py"

    def test_execute_search_empty_result(self) -> None:
        """Handle empty search results."""
        mock_result = MagicMock()
        mock_result.stdout = ""

        with patch("subprocess.run", return_value=mock_result):
            results = execute_search(["gh", "search", "code", "notfound"])

        assert results == []

    def test_execute_search_whitespace_only_result(self) -> None:
        """Handle whitespace-only output as empty result."""
        mock_result = MagicMock()
        mock_result.stdout = "   \n\n  "

        with patch("subprocess.run", return_value=mock_result):
            results = execute_search(["gh", "search", "code", "test"])

        assert results == []

    def test_execute_search_rate_limit_error(self) -> None:
        """Handle GitHub API rate limit error."""
        error = subprocess.CalledProcessError(1, "gh")
        error.stderr = (
            "Error: HTTP 403: API rate limit exceeded (GraphQL) "
            "with message rate_limit exceeded"
        )

        with patch("subprocess.run", side_effect=error):
            with pytest.raises(GHSearchError) as exc_info:
                execute_search(["gh", "search", "code", "test"])

        assert "rate limit" in str(exc_info.value).lower()

    def test_execute_search_timeout_error(self) -> None:
        """Handle search query timeout."""
        error = subprocess.CalledProcessError(1, "gh")
        error.stderr = "Error: HTTP 408: Request Timeout"

        with patch("subprocess.run", side_effect=error):
            with pytest.raises(GHSearchError) as exc_info:
                execute_search(["gh", "search", "code", "test"])

        assert "timed out" in str(exc_info.value).lower()

    def test_execute_search_generic_error(self) -> None:
        """Handle generic gh command error."""
        error = subprocess.CalledProcessError(1, "gh")
        error.stderr = "Error: authentication required"

        with patch("subprocess.run", side_effect=error):
            with pytest.raises(GHSearchError) as exc_info:
                execute_search(["gh", "search", "code", "test"])

        assert "GitHub search failed" in str(exc_info.value)

    def test_execute_search_json_decode_error(self) -> None:
        """Handle invalid JSON output."""
        mock_result = MagicMock()
        mock_result.stdout = "{ invalid json }"

        with patch("subprocess.run", return_value=mock_result):
            with pytest.raises(GHSearchError) as exc_info:
                execute_search(["gh", "search", "code", "test"])

        assert "Failed to parse JSON" in str(exc_info.value)

    def test_execute_search_calls_subprocess_with_correct_args(self) -> None:
        """Verify subprocess.run is called with correct arguments."""
        mock_result = MagicMock()
        mock_result.stdout = "[]"

        with patch("subprocess.run", return_value=mock_result) as mock_run:
            execute_search(["gh", "search", "code", "hello"])

        mock_run.assert_called_once_with(
            ["gh", "search", "code", "hello"],
            capture_output=True,
            text=True,
            check=True,
        )

    def test_execute_search_with_single_result(
        self, sample_search_result: SearchResult
    ) -> None:
        """Execute search returning single result."""
        mock_result = MagicMock()
        mock_result.stdout = json.dumps([sample_search_result])

        with patch("subprocess.run", return_value=mock_result):
            results = execute_search(["gh", "search", "code", "test"])

        assert len(results) == 1
        assert results[0] == sample_search_result

    def test_execute_search_preserves_result_structure(
        self, sample_results_list: list[SearchResult]
    ) -> None:
        """Search results preserve all fields."""
        mock_result = MagicMock()
        mock_result.stdout = json.dumps(sample_results_list)

        with patch("subprocess.run", return_value=mock_result):
            results = execute_search(["gh", "search", "code", "test"])

        for result in results:
            assert "path" in result
            assert "repository" in result
            assert "sha" in result
            assert "textMatches" in result
            assert "url" in result


# =============================================================================
# TestFilterResults
# =============================================================================


class TestFilterResults:
    """Tests for filter_results() function."""

    def test_filter_no_filters_returns_all(
        self,
        sample_results_list: list[SearchResult],
        basic_args: argparse.Namespace,
    ) -> None:
        """No filters applied returns all results."""
        filtered = filter_results(sample_results_list, basic_args)

        assert len(filtered) == 3

    def test_filter_exclude_forks(
        self,
        sample_results_list: list[SearchResult],
        basic_args: argparse.Namespace,
    ) -> None:
        """Exclude forked repositories."""
        basic_args.exclude_forks = True
        filtered = filter_results(sample_results_list, basic_args)

        assert len(filtered) == 2
        assert all(
            not r.get("repository", {}).get("isFork", False) for r in filtered
        )

    def test_filter_exclude_private(
        self,
        sample_results_list: list[SearchResult],
        basic_args: argparse.Namespace,
    ) -> None:
        """Exclude private repositories."""
        basic_args.exclude_private = True
        filtered = filter_results(sample_results_list, basic_args)

        # All samples are public, so should return all
        assert len(filtered) == 3

    def test_filter_exclude_private_with_private_repos(
        self, basic_args: argparse.Namespace
    ) -> None:
        """Exclude private repos when present."""
        results: list[SearchResult] = [
            make_minimal_result(path="public.py", repo_name="org/public"),
            make_minimal_result(
                path="private.py", repo_name="org/private", is_private=True
            ),
        ]
        basic_args.exclude_private = True
        filtered = filter_results(results, basic_args)

        assert len(filtered) == 1
        assert filtered[0]["path"] == "public.py"

    def test_filter_min_matches(
        self,
        sample_results_list: list[SearchResult],
        basic_args: argparse.Namespace,
    ) -> None:
        """Filter by minimum text matches."""
        basic_args.min_matches = 2
        filtered = filter_results(sample_results_list, basic_args)

        # Only first result has 2+ matches
        assert len(filtered) == 1
        assert filtered[0]["path"] == "src/main.py"

    def test_filter_min_matches_threshold(
        self, basic_args: argparse.Namespace
    ) -> None:
        """Filter respects minimum match threshold."""
        results: list[SearchResult] = [
            make_minimal_result(
                path="a.py",
                text_matches=[
                    {"fragment": "x", "indices": [0, 1]},
                    {"fragment": "y", "indices": [2, 3]},
                    {"fragment": "z", "indices": [4, 5]},
                ],
            ),
            make_minimal_result(
                path="b.py",
                text_matches=[{"fragment": "x", "indices": [0, 1]}],
            ),
            make_minimal_result(
                path="c.py",
                text_matches=[
                    {"fragment": "x", "indices": [0, 1]},
                    {"fragment": "y", "indices": [2, 3]},
                ],
            ),
        ]
        basic_args.min_matches = 2
        filtered = filter_results(results, basic_args)

        assert len(filtered) == 2
        assert any(r["path"] == "a.py" for r in filtered)
        assert any(r["path"] == "c.py" for r in filtered)

    def test_filter_combine_exclude_forks_and_private(
        self, basic_args: argparse.Namespace
    ) -> None:
        """Combine fork and private filters."""
        results: list[SearchResult] = [
            make_minimal_result(path="public_original.py", repo_name="org/public"),
            make_minimal_result(
                path="public_fork.py", repo_name="org/fork", is_fork=True
            ),
            make_minimal_result(
                path="private.py", repo_name="org/private", is_private=True
            ),
        ]
        basic_args.exclude_forks = True
        basic_args.exclude_private = True
        filtered = filter_results(results, basic_args)

        assert len(filtered) == 1
        assert filtered[0]["path"] == "public_original.py"

    def test_filter_all_three_conditions(
        self, basic_args: argparse.Namespace
    ) -> None:
        """Apply all three filter conditions simultaneously."""
        results: list[SearchResult] = [
            make_minimal_result(
                path="good.py",
                text_matches=[
                    {"fragment": "a", "indices": [0, 1]},
                    {"fragment": "b", "indices": [2, 3]},
                    {"fragment": "c", "indices": [4, 5]},
                ],
            ),
            make_minimal_result(
                path="fork.py",
                repo_name="org/fork",
                is_fork=True,
                text_matches=[
                    {"fragment": "a", "indices": [0, 1]},
                    {"fragment": "b", "indices": [2, 3]},
                    {"fragment": "c", "indices": [4, 5]},
                ],
            ),
            make_minimal_result(
                path="few_matches.py",
                text_matches=[{"fragment": "a", "indices": [0, 1]}],
            ),
        ]
        basic_args.exclude_forks = True
        basic_args.min_matches = 2
        filtered = filter_results(results, basic_args)

        assert len(filtered) == 1
        assert filtered[0]["path"] == "good.py"

    def test_filter_empty_results(self, basic_args: argparse.Namespace) -> None:
        """Filter empty result list returns empty list."""
        filtered = filter_results([], basic_args)

        assert filtered == []

    def test_filter_handles_missing_repository_field(
        self, basic_args: argparse.Namespace
    ) -> None:
        """Handle results with missing repository field."""
        # Deliberately using dict[str, Any] for edge case testing
        results: list[dict[str, Any]] = [{"path": "file.py", "textMatches": []}]
        basic_args.exclude_forks = True
        filtered = filter_results(results, basic_args)

        # Should not crash, isFork defaults to False
        assert len(filtered) == 1


# =============================================================================
# TestSortResults
# =============================================================================


class TestSortResults:
    """Tests for sort_results() function."""

    def test_sort_none_returns_unchanged(
        self, sample_results_list: list[SearchResult]
    ) -> None:
        """No sort specified returns results in original order."""
        sorted_results = sort_results(sample_results_list, None)

        assert sorted_results == sample_results_list

    def test_sort_by_matches_descending(
        self, sample_results_list: list[SearchResult]
    ) -> None:
        """Sort by text matches descending (most matches first)."""
        sorted_results = sort_results(sample_results_list, "matches")

        # First result has 2 matches, others have 1
        assert len(sorted_results[0].get("textMatches", [])) >= len(
            sorted_results[1].get("textMatches", [])
        )

    def test_sort_by_repo_alphabetical(self) -> None:
        """Sort by repository name alphabetically."""
        results: list[SearchResult] = [
            make_minimal_result(path="a.py", repo_name="zebra/repo"),
            make_minimal_result(path="b.py", repo_name="apple/repo"),
            make_minimal_result(path="c.py", repo_name="banana/repo"),
        ]
        sorted_results = sort_results(results, "repo")

        assert sorted_results[0]["repository"]["nameWithOwner"] == "apple/repo"
        assert sorted_results[1]["repository"]["nameWithOwner"] == "banana/repo"
        assert sorted_results[2]["repository"]["nameWithOwner"] == "zebra/repo"

    def test_sort_by_path_alphabetical(self) -> None:
        """Sort by file path alphabetically."""
        results: list[SearchResult] = [
            make_minimal_result(path="z_file.py"),
            make_minimal_result(path="a_file.py"),
            make_minimal_result(path="m_file.py"),
        ]
        sorted_results = sort_results(results, "path")

        assert sorted_results[0]["path"] == "a_file.py"
        assert sorted_results[1]["path"] == "m_file.py"
        assert sorted_results[2]["path"] == "z_file.py"

    def test_sort_invalid_sort_by_returns_unchanged(
        self, sample_results_list: list[SearchResult]
    ) -> None:
        """Invalid sort_by value returns unchanged results."""
        sorted_results = sort_results(sample_results_list, "invalid")

        assert sorted_results == sample_results_list

    def test_sort_matches_with_equal_counts(self) -> None:
        """Sort by matches handles equal match counts."""
        results: list[SearchResult] = [
            make_minimal_result(
                path="b.py",
                text_matches=[
                    {"fragment": "x", "indices": [0, 1]},
                    {"fragment": "y", "indices": [2, 3]},
                ],
            ),
            make_minimal_result(
                path="a.py",
                text_matches=[
                    {"fragment": "x", "indices": [0, 1]},
                    {"fragment": "y", "indices": [2, 3]},
                ],
            ),
            make_minimal_result(
                path="c.py",
                text_matches=[{"fragment": "x", "indices": [0, 1]}],
            ),
        ]
        sorted_results = sort_results(results, "matches")

        # Top results should have 2 matches
        assert len(sorted_results[0]["textMatches"]) == 2
        assert len(sorted_results[1]["textMatches"]) == 2
        assert len(sorted_results[2]["textMatches"]) == 1

    def test_sort_empty_results(self) -> None:
        """Sort empty results returns empty list."""
        sorted_results = sort_results([], "matches")

        assert sorted_results == []

    def test_sort_single_result(self, sample_search_result: SearchResult) -> None:
        """Sort single result returns single result."""
        sorted_results = sort_results([sample_search_result], "matches")

        assert len(sorted_results) == 1

    @pytest.mark.parametrize("sort_by", ["matches", "repo", "path"])
    def test_sort_by_options(
        self, sample_results_list: list[SearchResult], sort_by: str
    ) -> None:
        """All sort options work without error."""
        sorted_results = sort_results(sample_results_list, sort_by)

        assert len(sorted_results) == len(sample_results_list)


# =============================================================================
# TestFormatJson
# =============================================================================


class TestFormatJson:
    """Tests for format_json() function."""

    def test_format_json_valid_structure(
        self, sample_results_list: list[SearchResult]
    ) -> None:
        """Format results as valid JSON."""
        output = format_json(sample_results_list)

        # Should be valid JSON
        parsed = json.loads(output)
        assert len(parsed) == 3

    def test_format_json_preserves_all_fields(
        self, sample_search_result: SearchResult
    ) -> None:
        """JSON format preserves all result fields."""
        output = format_json([sample_search_result])
        parsed = json.loads(output)

        assert parsed[0]["path"] == sample_search_result["path"]
        assert parsed[0]["repository"] == sample_search_result["repository"]
        assert parsed[0]["textMatches"] == sample_search_result["textMatches"]

    def test_format_json_empty_list(self) -> None:
        """Format empty results as JSON array."""
        output = format_json([])

        assert output == "[]"

    def test_format_json_indentation(
        self, sample_results_list: list[SearchResult]
    ) -> None:
        """JSON is formatted with proper indentation."""
        output = format_json(sample_results_list)

        # Check for proper indentation (2 spaces)
        lines = output.split("\n")
        assert any(line.startswith("  ") for line in lines)

    def test_format_json_single_result(
        self, sample_search_result: SearchResult
    ) -> None:
        """Format single result as JSON."""
        output = format_json([sample_search_result])
        parsed = json.loads(output)

        assert len(parsed) == 1
        assert parsed[0] == sample_search_result


# =============================================================================
# TestFormatPretty
# =============================================================================


class TestFormatPretty:
    """Tests for format_pretty() function."""

    def test_format_pretty_empty_results(self) -> None:
        """Format empty results shows 'No results found'."""
        output = format_pretty([])

        assert "No results found" in output

    def test_format_pretty_shows_result_count(
        self, sample_results_list: list[SearchResult]
    ) -> None:
        """Pretty format shows total result count."""
        output = format_pretty(sample_results_list)

        assert "3 result" in output

    def test_format_pretty_shows_repository_and_path(
        self, sample_search_result: SearchResult
    ) -> None:
        """Pretty format includes repository and path."""
        output = format_pretty([sample_search_result])

        assert "octocat/Hello-World" in output
        assert "src/main.py" in output

    def test_format_pretty_shows_url(
        self, sample_search_result: SearchResult
    ) -> None:
        """Pretty format includes GitHub URL."""
        output = format_pretty([sample_search_result])

        assert "https://github.com" in output

    def test_format_pretty_shows_match_count(
        self, sample_search_result: SearchResult
    ) -> None:
        """Pretty format shows number of matches."""
        output = format_pretty([sample_search_result])

        # Should show "Matches: 2"
        assert "Matches: 2" in output

    def test_format_pretty_shows_preview(
        self, sample_search_result: SearchResult
    ) -> None:
        """Pretty format includes text match preview."""
        output = format_pretty([sample_search_result])

        assert "Preview:" in output
        assert "def hello():" in output or "hello" in output

    def test_format_pretty_truncates_long_fragments(self) -> None:
        """Long match fragments are truncated."""
        long_fragment = "x" * 200
        results: list[SearchResult] = [
            {
                "path": "long.py",
                "repository": {"nameWithOwner": "org/repo"},
                "url": "https://example.com",
                "textMatches": [
                    {"fragment": long_fragment, "indices": [0, 200]},
                ],
            }
        ]
        output = format_pretty(results)

        # Fragment should be truncated (100 chars max) with ellipsis
        assert long_fragment not in output
        assert "..." in output
        # The preview line should contain truncated fragment
        assert "Preview:" in output

    def test_format_pretty_numbered_results(
        self, sample_results_list: list[SearchResult]
    ) -> None:
        """Results are numbered in pretty output."""
        output = format_pretty(sample_results_list)

        assert "1." in output
        assert "2." in output
        assert "3." in output

    def test_format_pretty_includes_separators(
        self, sample_results_list: list[SearchResult]
    ) -> None:
        """Pretty output includes visual separators."""
        output = format_pretty(sample_results_list)

        assert "=" in output
        assert "-" in output

    def test_format_pretty_single_result(
        self, sample_search_result: SearchResult
    ) -> None:
        """Pretty format single result."""
        output = format_pretty([sample_search_result])

        assert "Found 1 result" in output
        assert "1." in output


# =============================================================================
# TestFormatSummary
# =============================================================================


class TestFormatSummary:
    """Tests for format_summary() function."""

    def test_format_summary_empty_results(self) -> None:
        """Summary of empty results shows 'No results found'."""
        output = format_summary([])

        assert "No results found" in output

    def test_format_summary_shows_header(
        self, sample_results_list: list[SearchResult]
    ) -> None:
        """Summary includes header."""
        output = format_summary(sample_results_list)

        assert "SEARCH SUMMARY" in output

    def test_format_summary_shows_file_count(
        self, sample_results_list: list[SearchResult]
    ) -> None:
        """Summary shows total files found."""
        output = format_summary(sample_results_list)

        assert "Total files found: 3" in output

    def test_format_summary_shows_total_matches(
        self, sample_results_list: list[SearchResult]
    ) -> None:
        """Summary shows total text matches."""
        output = format_summary(sample_results_list)

        assert "Total text matches:" in output

    def test_format_summary_shows_unique_repos(
        self, sample_results_list: list[SearchResult]
    ) -> None:
        """Summary shows count of unique repositories."""
        output = format_summary(sample_results_list)

        assert "Unique repositories: 2" in output

    def test_format_summary_lists_top_repos(
        self, sample_results_list: list[SearchResult]
    ) -> None:
        """Summary lists top repositories by file count."""
        output = format_summary(sample_results_list)

        assert "Top Repositories:" in output
        assert "octocat/Hello-World" in output

    def test_format_summary_shows_file_extensions(
        self, sample_results_list: list[SearchResult]
    ) -> None:
        """Summary shows file extension breakdown."""
        output = format_summary(sample_results_list)

        assert "File Extensions:" in output
        assert ".py" in output
        assert ".js" in output

    def test_format_summary_single_result(
        self, sample_search_result: SearchResult
    ) -> None:
        """Summary for single result."""
        output = format_summary([sample_search_result])

        assert "Total files found: 1" in output
        assert "Unique repositories: 1" in output

    def test_format_summary_extension_count_accuracy(self) -> None:
        """Extension counts are accurate."""
        results: list[SearchResult] = [
            make_minimal_result(path="file1.py"),
            make_minimal_result(path="file2.py"),
            make_minimal_result(path="script.js"),
        ]
        output = format_summary(results)

        assert ".py: 2" in output
        assert ".js: 1" in output

    def test_format_summary_repo_count_accuracy(self) -> None:
        """Repository counts are accurate."""
        results: list[SearchResult] = [
            make_minimal_result(path="a.py", repo_name="org/repo1"),
            make_minimal_result(path="b.py", repo_name="org/repo1"),
            make_minimal_result(path="c.py", repo_name="org/repo2"),
        ]
        output = format_summary(results)

        assert "org/repo1: 2" in output
        assert "org/repo2: 1" in output


# =============================================================================
# TestGHSearchError
# =============================================================================


class TestGHSearchError:
    """Tests for GHSearchError exception."""

    def test_exception_is_exception(self) -> None:
        """GHSearchError is an Exception."""
        assert issubclass(GHSearchError, Exception)

    def test_exception_with_message(self) -> None:
        """GHSearchError can be created with message."""
        error = GHSearchError("Test error message")

        assert str(error) == "Test error message"

    def test_exception_can_be_raised_and_caught(self) -> None:
        """GHSearchError can be raised and caught."""
        with pytest.raises(GHSearchError) as exc_info:
            raise GHSearchError("Test error")

        assert "Test error" in str(exc_info.value)


# =============================================================================
# TestMain
# =============================================================================


class TestMain:
    """Tests for main() function entry point."""

    def test_main_basic_search(
        self,
        sample_results_list: list[SearchResult],
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """Main function executes basic search and outputs results."""
        with patch("sys.argv", ["gh_code_search.py", "test", "--output", "pretty"]):
            with patch(
                "gh_code_search.execute_search", return_value=sample_results_list
            ):
                with pytest.raises(SystemExit) as exc_info:
                    main()

        assert exc_info.value.code == 0
        captured = capsys.readouterr()
        assert "3 result" in captured.out or "Found" in captured.out

    def test_main_json_output(
        self,
        sample_results_list: list[SearchResult],
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """Main outputs JSON format when requested."""
        with patch("sys.argv", ["gh_code_search.py", "test", "--output", "json"]):
            with patch(
                "gh_code_search.execute_search", return_value=sample_results_list
            ):
                with pytest.raises(SystemExit) as exc_info:
                    main()

        captured = capsys.readouterr()
        parsed = json.loads(captured.out)
        assert len(parsed) == 3

    def test_main_summary_output(
        self,
        sample_results_list: list[SearchResult],
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """Main outputs summary format when requested."""
        with patch("sys.argv", ["gh_code_search.py", "test", "--output", "summary"]):
            with patch(
                "gh_code_search.execute_search", return_value=sample_results_list
            ):
                with pytest.raises(SystemExit) as exc_info:
                    main()

        assert exc_info.value.code == 0
        captured = capsys.readouterr()
        assert "SEARCH SUMMARY" in captured.out

    def test_main_with_language_filter(
        self,
        sample_results_list: list[SearchResult],
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """Main applies language filter to command."""
        with patch(
            "sys.argv",
            ["gh_code_search.py", "test", "--language", "python"],
        ):
            with patch(
                "gh_code_search.execute_search", return_value=sample_results_list
            ) as mock_exec:
                with pytest.raises(SystemExit):
                    main()

        cmd = mock_exec.call_args[0][0]
        assert "--language" in cmd
        assert "python" in cmd

    def test_main_with_sorting(
        self,
        sample_results_list: list[SearchResult],
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """Main sorts results by specified criteria."""
        with patch(
            "sys.argv",
            ["gh_code_search.py", "test", "--sort-by", "repo"],
        ):
            with patch(
                "gh_code_search.execute_search", return_value=sample_results_list
            ):
                with pytest.raises(SystemExit) as exc_info:
                    main()

        assert exc_info.value.code == 0

    def test_main_search_error_exits_1(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Main exits with code 1 on search error."""
        with patch("sys.argv", ["gh_code_search.py", "test"]):
            with patch(
                "gh_code_search.execute_search",
                side_effect=GHSearchError("Test error"),
            ):
                with pytest.raises(SystemExit) as exc_info:
                    main()

        assert exc_info.value.code == 1
        captured = capsys.readouterr()
        assert "Error:" in captured.err or "error" in captured.err.lower()

    def test_main_keyboard_interrupt_exits_130(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Main exits with code 130 on keyboard interrupt."""
        with patch("sys.argv", ["gh_code_search.py", "test"]):
            with patch(
                "gh_code_search.execute_search",
                side_effect=KeyboardInterrupt(),
            ):
                with pytest.raises(SystemExit) as exc_info:
                    main()

        assert exc_info.value.code == 130

    def test_main_unexpected_error_exits_1(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Main exits with code 1 on unexpected error."""
        with patch("sys.argv", ["gh_code_search.py", "test"]):
            with patch(
                "gh_code_search.execute_search",
                side_effect=RuntimeError("Unexpected error"),
            ):
                with pytest.raises(SystemExit) as exc_info:
                    main()

        assert exc_info.value.code == 1

    def test_main_empty_results(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Main handles empty search results."""
        with patch("sys.argv", ["gh_code_search.py", "notfound"]):
            with patch("gh_code_search.execute_search", return_value=[]):
                with pytest.raises(SystemExit) as exc_info:
                    main()

        assert exc_info.value.code == 0
        captured = capsys.readouterr()
        assert "No results found" in captured.out

    def test_main_with_exclude_forks(
        self, sample_results_list: list[SearchResult]
    ) -> None:
        """Main applies exclude-forks filter."""
        with patch(
            "sys.argv",
            ["gh_code_search.py", "test", "--exclude-forks"],
        ):
            with patch(
                "gh_code_search.execute_search", return_value=sample_results_list
            ):
                with pytest.raises(SystemExit) as exc_info:
                    main()

        assert exc_info.value.code == 0

    def test_main_with_min_matches(
        self, sample_results_list: list[SearchResult]
    ) -> None:
        """Main applies minimum matches filter."""
        with patch(
            "sys.argv",
            ["gh_code_search.py", "test", "--min-matches", "2"],
        ):
            with patch(
                "gh_code_search.execute_search", return_value=sample_results_list
            ):
                with pytest.raises(SystemExit) as exc_info:
                    main()

        assert exc_info.value.code == 0

    def test_main_help_shows_examples(self) -> None:
        """Main help includes usage examples."""
        with patch("sys.argv", ["gh_code_search.py", "--help"]):
            with pytest.raises(SystemExit) as exc_info:
                main()

        # Help exits with 0
        assert exc_info.value.code == 0

    def test_main_with_multiple_repos(
        self, sample_results_list: list[SearchResult]
    ) -> None:
        """Main handles multiple repository filters."""
        with patch(
            "sys.argv",
            [
                "gh_code_search.py",
                "test",
                "--repo",
                "microsoft/vscode",
                "--repo",
                "torvalds/linux",
            ],
        ):
            with patch(
                "gh_code_search.execute_search", return_value=sample_results_list
            ) as mock_exec:
                with pytest.raises(SystemExit):
                    main()

        cmd = mock_exec.call_args[0][0]
        assert cmd.count("--repo") == 2

    def test_main_with_multiple_owners(
        self, sample_results_list: list[SearchResult]
    ) -> None:
        """Main handles multiple owner filters."""
        with patch(
            "sys.argv",
            [
                "gh_code_search.py",
                "test",
                "--owner",
                "google",
                "--owner",
                "facebook",
            ],
        ):
            with patch(
                "gh_code_search.execute_search", return_value=sample_results_list
            ) as mock_exec:
                with pytest.raises(SystemExit):
                    main()

        cmd = mock_exec.call_args[0][0]
        assert cmd.count("--owner") == 2


# =============================================================================
# TestIntegration
# =============================================================================


class TestIntegration:
    """Integration tests combining multiple functions."""

    def test_full_search_with_filters_and_sorting(
        self, sample_results_list: list[SearchResult]
    ) -> None:
        """Full workflow: build command, execute, filter, sort, format."""
        args = argparse.Namespace(
            query="test",
            limit=50,
            language="python",
            filename=None,
            extension=None,
            repo=None,
            owner=None,
            match=None,
            size=None,
            exclude_forks=True,
            exclude_private=False,
            min_matches=1,
            output="pretty",
            sort_by="matches",
        )

        cmd = build_gh_command(args)
        assert "--language" in cmd

        filtered = filter_results(sample_results_list, args)
        assert all(
            not r.get("repository", {}).get("isFork", False) for r in filtered
        )

        sorted_results = sort_results(filtered, args.sort_by)
        assert len(sorted_results) > 0

    def test_workflow_json_output_format(
        self, sample_results_list: list[SearchResult]
    ) -> None:
        """Workflow with JSON output format."""
        args = make_filter_args()
        filtered = filter_results(sample_results_list, args)
        sorted_results = sort_results(filtered, None)
        output = format_json(sorted_results)

        parsed = json.loads(output)
        assert len(parsed) == len(sample_results_list)

    def test_workflow_pretty_output_format(
        self, sample_results_list: list[SearchResult]
    ) -> None:
        """Workflow with pretty output format."""
        args = make_filter_args(exclude_forks=True)
        filtered = filter_results(sample_results_list, args)
        sorted_results = sort_results(filtered, "repo")
        output = format_pretty(sorted_results)

        assert "result" in output.lower()
        assert "http" in output.lower()

    def test_workflow_summary_output_format(
        self, sample_results_list: list[SearchResult]
    ) -> None:
        """Workflow with summary output format."""
        args = make_filter_args()
        filtered = filter_results(sample_results_list, args)
        sorted_results = sort_results(filtered, "path")
        output = format_summary(sorted_results)

        assert "SUMMARY" in output
        assert "files found" in output.lower()


# =============================================================================
# Main Entry Point
# =============================================================================


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
