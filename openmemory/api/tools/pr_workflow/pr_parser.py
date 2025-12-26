"""PR Diff Parser.

This module provides unified diff parsing functionality:
- PRLine: Individual line in a diff
- PRHunk: Hunk of changes in a file
- PRFile: File with changes
- PRDiff: Complete diff with multiple files
- DiffParser: Parser for unified diff format
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


# =============================================================================
# Exceptions
# =============================================================================


class DiffParseError(Exception):
    """Raised when diff parsing fails."""

    pass


# =============================================================================
# Language Detection
# =============================================================================

LANGUAGE_MAP = {
    ".py": "python",
    ".pyi": "python",
    ".pyx": "python",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".js": "javascript",
    ".jsx": "javascript",
    ".java": "java",
    ".kt": "kotlin",
    ".kts": "kotlin",
    ".go": "go",
    ".rs": "rust",
    ".rb": "ruby",
    ".php": "php",
    ".cs": "csharp",
    ".cpp": "cpp",
    ".cc": "cpp",
    ".cxx": "cpp",
    ".c": "c",
    ".h": "c",
    ".hpp": "cpp",
    ".swift": "swift",
    ".scala": "scala",
    ".r": "r",
    ".R": "r",
    ".sql": "sql",
    ".sh": "shell",
    ".bash": "shell",
    ".zsh": "shell",
    ".yml": "yaml",
    ".yaml": "yaml",
    ".json": "json",
    ".xml": "xml",
    ".html": "html",
    ".css": "css",
    ".scss": "scss",
    ".sass": "sass",
    ".less": "less",
    ".md": "markdown",
    ".rst": "restructuredtext",
    ".txt": "text",
}


def detect_language(path: str) -> str:
    """Detect language from file path."""
    if not path:
        return "unknown"
    for ext, lang in LANGUAGE_MAP.items():
        if path.endswith(ext):
            return lang
    return "unknown"


# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class PRLine:
    """A single line in a diff.

    Args:
        content: The line content (with +/- prefix for changes)
        line_type: Type of line - "context", "addition", or "deletion"
        old_line_number: Line number in the old file (None for additions)
        new_line_number: Line number in the new file (None for deletions)
    """

    content: str
    line_type: str  # "context", "addition", "deletion"
    old_line_number: Optional[int]
    new_line_number: Optional[int]

    def __eq__(self, other):
        if not isinstance(other, PRLine):
            return False
        return (
            self.content == other.content
            and self.line_type == other.line_type
            and self.old_line_number == other.old_line_number
            and self.new_line_number == other.new_line_number
        )


@dataclass
class PRHunk:
    """A hunk of changes in a diff.

    Args:
        old_start: Starting line number in old file
        old_count: Number of lines in old file
        new_start: Starting line number in new file
        new_count: Number of lines in new file
        header: Full hunk header line (e.g., "@@ -10,3 +10,4 @@ def func():")
        lines: List of lines in this hunk
    """

    old_start: int
    old_count: int
    new_start: int
    new_count: int
    header: str
    lines: list[PRLine] = field(default_factory=list)

    @property
    def changed_lines(self) -> list[PRLine]:
        """Get only added and deleted lines (no context)."""
        return [l for l in self.lines if l.line_type in ("addition", "deletion")]


@dataclass
class PRFile:
    """A file in a diff.

    Args:
        old_path: Path in the old version (None for new files)
        new_path: Path in the new version (None for deleted files)
        status: File status - "modified", "added", "deleted", "renamed", "copied"
        hunks: List of hunks with changes
        similarity_index: Similarity percentage for renamed/copied files
        is_binary: Whether this is a binary file
    """

    old_path: Optional[str]
    new_path: Optional[str]
    status: str  # "modified", "added", "deleted", "renamed", "copied"
    hunks: list[PRHunk] = field(default_factory=list)
    similarity_index: Optional[int] = None
    is_binary: bool = False

    @property
    def path(self) -> str:
        """Get the effective path (new_path for additions, old_path for deletions)."""
        return self.new_path or self.old_path or ""

    @property
    def language(self) -> str:
        """Detect language from file extension."""
        return detect_language(self.path)

    @property
    def additions(self) -> int:
        """Count total additions in this file."""
        return sum(
            len([l for l in h.lines if l.line_type == "addition"]) for h in self.hunks
        )

    @property
    def deletions(self) -> int:
        """Count total deletions in this file."""
        return sum(
            len([l for l in h.lines if l.line_type == "deletion"]) for h in self.hunks
        )


@dataclass
class PRDiff:
    """A complete diff with multiple files.

    Args:
        files: List of files in the diff
    """

    files: list[PRFile] = field(default_factory=list)

    @property
    def files_changed(self) -> int:
        """Total number of files changed."""
        return len(self.files)

    @property
    def files_added(self) -> int:
        """Number of files added."""
        return len([f for f in self.files if f.status == "added"])

    @property
    def files_deleted(self) -> int:
        """Number of files deleted."""
        return len([f for f in self.files if f.status == "deleted"])

    @property
    def files_modified(self) -> int:
        """Number of files modified."""
        return len([f for f in self.files if f.status == "modified"])

    @property
    def total_additions(self) -> int:
        """Total additions across all files."""
        return sum(f.additions for f in self.files)

    @property
    def total_deletions(self) -> int:
        """Total deletions across all files."""
        return sum(f.deletions for f in self.files)

    @property
    def files_by_language(self) -> dict[str, list[PRFile]]:
        """Group files by language."""
        result: dict[str, list[PRFile]] = {}
        for f in self.files:
            lang = f.language
            if lang not in result:
                result[lang] = []
            result[lang].append(f)
        return result

    def get_file(self, path: str) -> Optional[PRFile]:
        """Get file by path."""
        for f in self.files:
            if f.new_path == path or f.old_path == path:
                return f
        return None

    def filter_by_extension(self, extension: str) -> list[PRFile]:
        """Filter files by extension."""
        return [f for f in self.files if f.path.endswith(extension)]


# =============================================================================
# Parser
# =============================================================================


class DiffParser:
    """Parser for unified diff format.

    Handles git diff output including:
    - Modified files
    - Added files
    - Deleted files
    - Renamed files
    - Copied files
    - Binary files
    """

    # Regex patterns for parsing
    DIFF_HEADER_RE = re.compile(r"^diff --git a/(.+) b/(.+)$")
    HUNK_HEADER_RE = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@(.*)$")
    OLD_FILE_RE = re.compile(r"^--- (.+)$")
    NEW_FILE_RE = re.compile(r"^\+\+\+ (.+)$")
    RENAME_FROM_RE = re.compile(r"^rename from (.+)$")
    RENAME_TO_RE = re.compile(r"^rename to (.+)$")
    COPY_FROM_RE = re.compile(r"^copy from (.+)$")
    COPY_TO_RE = re.compile(r"^copy to (.+)$")
    SIMILARITY_RE = re.compile(r"^similarity index (\d+)%$")
    BINARY_RE = re.compile(r"^Binary files")
    INDEX_RE = re.compile(r"^index [a-f0-9]+\.\.[a-f0-9]+")

    def parse(self, diff_text: str) -> PRDiff:
        """Parse a unified diff.

        Args:
            diff_text: The diff text to parse

        Returns:
            PRDiff containing all parsed files
        """
        if not diff_text or not diff_text.strip():
            return PRDiff(files=[])

        files: list[PRFile] = []
        lines = diff_text.split("\n")
        i = 0

        while i < len(lines):
            line = lines[i]

            # Look for diff header
            match = self.DIFF_HEADER_RE.match(line)
            if match:
                file_data, i = self._parse_file(lines, i)
                if file_data:
                    files.append(file_data)
            else:
                i += 1

        return PRDiff(files=files)

    def _parse_file(self, lines: list[str], start_idx: int) -> tuple[Optional[PRFile], int]:
        """Parse a single file from the diff.

        Returns:
            Tuple of (PRFile or None, next index to process)
        """
        i = start_idx
        line = lines[i]

        # Parse diff header
        match = self.DIFF_HEADER_RE.match(line)
        if not match:
            return None, i + 1

        header_old_path = match.group(1)
        header_new_path = match.group(2)
        i += 1

        # Parse metadata lines
        old_path = header_old_path
        new_path = header_new_path
        status = "modified"
        similarity_index = None
        is_binary = False
        rename_from = None
        rename_to = None
        copy_from = None
        copy_to = None

        while i < len(lines):
            line = lines[i]

            # Check for next file
            if self.DIFF_HEADER_RE.match(line):
                break

            # Check for hunk start
            if self.HUNK_HEADER_RE.match(line):
                break

            # Parse metadata
            if line.startswith("old mode") or line.startswith("new mode"):
                i += 1
                continue

            if line.startswith("deleted file"):
                status = "deleted"
                i += 1
                continue

            if line.startswith("new file"):
                status = "added"
                i += 1
                continue

            sim_match = self.SIMILARITY_RE.match(line)
            if sim_match:
                similarity_index = int(sim_match.group(1))
                i += 1
                continue

            rename_from_match = self.RENAME_FROM_RE.match(line)
            if rename_from_match:
                rename_from = rename_from_match.group(1)
                status = "renamed"
                i += 1
                continue

            rename_to_match = self.RENAME_TO_RE.match(line)
            if rename_to_match:
                rename_to = rename_to_match.group(1)
                status = "renamed"
                i += 1
                continue

            copy_from_match = self.COPY_FROM_RE.match(line)
            if copy_from_match:
                copy_from = copy_from_match.group(1)
                status = "copied"
                i += 1
                continue

            copy_to_match = self.COPY_TO_RE.match(line)
            if copy_to_match:
                copy_to = copy_to_match.group(1)
                status = "copied"
                i += 1
                continue

            if self.BINARY_RE.match(line):
                is_binary = True
                i += 1
                continue

            if self.INDEX_RE.match(line):
                i += 1
                continue

            old_match = self.OLD_FILE_RE.match(line)
            if old_match:
                path = old_match.group(1)
                if path.startswith("a/"):
                    path = path[2:]
                if path != "/dev/null":
                    old_path = path
                else:
                    old_path = None
                    status = "added"
                i += 1
                continue

            new_match = self.NEW_FILE_RE.match(line)
            if new_match:
                path = new_match.group(1)
                if path.startswith("b/"):
                    path = path[2:]
                if path != "/dev/null":
                    new_path = path
                else:
                    new_path = None
                    status = "deleted"
                i += 1
                continue

            # Skip other lines until we hit something we recognize
            i += 1

        # Apply rename/copy paths
        if rename_from:
            old_path = rename_from
        if rename_to:
            new_path = rename_to
        if copy_from:
            old_path = copy_from
        if copy_to:
            new_path = copy_to

        # Parse hunks
        hunks: list[PRHunk] = []
        while i < len(lines):
            line = lines[i]

            # Check for next file
            if self.DIFF_HEADER_RE.match(line):
                break

            # Parse hunk
            hunk_match = self.HUNK_HEADER_RE.match(line)
            if hunk_match:
                hunk, i = self._parse_hunk(lines, i)
                if hunk:
                    hunks.append(hunk)
            else:
                i += 1

        return PRFile(
            old_path=old_path,
            new_path=new_path,
            status=status,
            hunks=hunks,
            similarity_index=similarity_index,
            is_binary=is_binary,
        ), i

    def _parse_hunk(self, lines: list[str], start_idx: int) -> tuple[Optional[PRHunk], int]:
        """Parse a single hunk.

        Returns:
            Tuple of (PRHunk or None, next index to process)
        """
        i = start_idx
        line = lines[i]

        # Parse hunk header
        match = self.HUNK_HEADER_RE.match(line)
        if not match:
            return None, i + 1

        old_start = int(match.group(1))
        old_count = int(match.group(2)) if match.group(2) else 1
        new_start = int(match.group(3))
        new_count = int(match.group(4)) if match.group(4) else 1
        header = line
        i += 1

        # Parse lines
        hunk_lines: list[PRLine] = []
        old_line = old_start
        new_line = new_start

        while i < len(lines):
            line = lines[i]

            # Check for next hunk or file
            if self.HUNK_HEADER_RE.match(line) or self.DIFF_HEADER_RE.match(line):
                break

            # Skip "no newline" markers
            if line.startswith("\\ No newline"):
                i += 1
                continue

            # Parse line
            if line.startswith("+"):
                hunk_lines.append(
                    PRLine(
                        content=line,
                        line_type="addition",
                        old_line_number=None,
                        new_line_number=new_line,
                    )
                )
                new_line += 1
            elif line.startswith("-"):
                hunk_lines.append(
                    PRLine(
                        content=line,
                        line_type="deletion",
                        old_line_number=old_line,
                        new_line_number=None,
                    )
                )
                old_line += 1
            elif line.startswith(" ") or line == "":
                # Context line
                hunk_lines.append(
                    PRLine(
                        content=line,
                        line_type="context",
                        old_line_number=old_line,
                        new_line_number=new_line,
                    )
                )
                old_line += 1
                new_line += 1
            else:
                # Unknown line type - might be end of hunk
                break

            i += 1

        return PRHunk(
            old_start=old_start,
            old_count=old_count,
            new_start=new_start,
            new_count=new_count,
            header=header,
            lines=hunk_lines,
        ), i
