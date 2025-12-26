"""Tests for PR diff parser.

This module tests the PR diff parsing functionality with TDD approach:
- PRLine: Individual line in a diff
- PRHunk: Hunk of changes in a file
- PRFile: File with changes
- PRDiff: Complete diff with multiple files
- DiffParser: Parser for unified diff format
"""

from __future__ import annotations

import pytest


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def simple_diff() -> str:
    """Return a simple unified diff."""
    return """diff --git a/src/main.py b/src/main.py
index abc1234..def5678 100644
--- a/src/main.py
+++ b/src/main.py
@@ -10,7 +10,7 @@ def main():
     print("Hello")
-    old_function()
+    new_function()
     print("World")
"""


@pytest.fixture
def multi_file_diff() -> str:
    """Return a diff with multiple files."""
    return """diff --git a/src/main.py b/src/main.py
index abc1234..def5678 100644
--- a/src/main.py
+++ b/src/main.py
@@ -10,7 +10,7 @@ def main():
     print("Hello")
-    old_function()
+    new_function()
     print("World")
diff --git a/src/utils.py b/src/utils.py
new file mode 100644
index 0000000..abc1234
--- /dev/null
+++ b/src/utils.py
@@ -0,0 +1,5 @@
+def helper():
+    return 42
+
+def another_helper():
+    return "test"
"""


@pytest.fixture
def deleted_file_diff() -> str:
    """Return a diff with a deleted file."""
    return """diff --git a/src/old_module.py b/src/old_module.py
deleted file mode 100644
index abc1234..0000000
--- a/src/old_module.py
+++ /dev/null
@@ -1,10 +0,0 @@
-def old_function():
-    pass
-
-def another_old():
-    return None
"""


@pytest.fixture
def renamed_file_diff() -> str:
    """Return a diff with a renamed file."""
    return """diff --git a/src/old_name.py b/src/new_name.py
similarity index 90%
rename from src/old_name.py
rename to src/new_name.py
index abc1234..def5678 100644
--- a/src/old_name.py
+++ b/src/new_name.py
@@ -5,7 +5,7 @@ class MyClass:
     def __init__(self):
-        self.value = 0
+        self.value = 1
"""


@pytest.fixture
def multi_hunk_diff() -> str:
    """Return a diff with multiple hunks in one file."""
    return """diff --git a/src/large.py b/src/large.py
index abc1234..def5678 100644
--- a/src/large.py
+++ b/src/large.py
@@ -10,7 +10,7 @@ def first_function():
     print("Hello")
-    old_call()
+    new_call()
     print("World")
@@ -50,6 +50,8 @@ def second_function():
     result = compute()
+    # Added validation
+    validate(result)
     return result
"""


@pytest.fixture
def binary_file_diff() -> str:
    """Return a diff with a binary file."""
    return """diff --git a/assets/image.png b/assets/image.png
new file mode 100644
index 0000000..abc1234
Binary files /dev/null and b/assets/image.png differ
"""


# =============================================================================
# PRLine Tests
# =============================================================================


class TestPRLine:
    """Tests for PRLine dataclass."""

    def test_line_creation(self):
        """Test creating a diff line."""
        from openmemory.api.tools.pr_workflow.pr_parser import PRLine

        line = PRLine(
            content="    return value",
            line_type="context",
            old_line_number=10,
            new_line_number=10,
        )

        assert line.content == "    return value"
        assert line.line_type == "context"
        assert line.old_line_number == 10
        assert line.new_line_number == 10

    def test_added_line(self):
        """Test added line type."""
        from openmemory.api.tools.pr_workflow.pr_parser import PRLine

        line = PRLine(
            content="+    new_code()",
            line_type="addition",
            old_line_number=None,
            new_line_number=15,
        )

        assert line.line_type == "addition"
        assert line.old_line_number is None
        assert line.new_line_number == 15

    def test_removed_line(self):
        """Test removed line type."""
        from openmemory.api.tools.pr_workflow.pr_parser import PRLine

        line = PRLine(
            content="-    old_code()",
            line_type="deletion",
            old_line_number=15,
            new_line_number=None,
        )

        assert line.line_type == "deletion"
        assert line.old_line_number == 15
        assert line.new_line_number is None

    def test_line_equality(self):
        """Test line equality."""
        from openmemory.api.tools.pr_workflow.pr_parser import PRLine

        line1 = PRLine("content", "context", 1, 1)
        line2 = PRLine("content", "context", 1, 1)

        assert line1 == line2


# =============================================================================
# PRHunk Tests
# =============================================================================


class TestPRHunk:
    """Tests for PRHunk dataclass."""

    def test_hunk_creation(self):
        """Test creating a diff hunk."""
        from openmemory.api.tools.pr_workflow.pr_parser import PRHunk, PRLine

        lines = [
            PRLine(" context", "context", 10, 10),
            PRLine("-old", "deletion", 11, None),
            PRLine("+new", "addition", None, 11),
        ]

        hunk = PRHunk(
            old_start=10,
            old_count=2,
            new_start=10,
            new_count=2,
            header="@@ -10,2 +10,2 @@ def main():",
            lines=lines,
        )

        assert hunk.old_start == 10
        assert hunk.old_count == 2
        assert hunk.new_start == 10
        assert hunk.new_count == 2
        assert len(hunk.lines) == 3

    def test_hunk_additions_count(self):
        """Test counting additions in a hunk."""
        from openmemory.api.tools.pr_workflow.pr_parser import PRHunk, PRLine

        lines = [
            PRLine(" context", "context", 10, 10),
            PRLine("-old", "deletion", 11, None),
            PRLine("+new1", "addition", None, 11),
            PRLine("+new2", "addition", None, 12),
        ]

        hunk = PRHunk(
            old_start=10,
            old_count=2,
            new_start=10,
            new_count=3,
            header="@@ -10,2 +10,3 @@",
            lines=lines,
        )

        additions = [l for l in hunk.lines if l.line_type == "addition"]
        assert len(additions) == 2

    def test_hunk_deletions_count(self):
        """Test counting deletions in a hunk."""
        from openmemory.api.tools.pr_workflow.pr_parser import PRHunk, PRLine

        lines = [
            PRLine("-old1", "deletion", 10, None),
            PRLine("-old2", "deletion", 11, None),
            PRLine("+new", "addition", None, 10),
        ]

        hunk = PRHunk(
            old_start=10,
            old_count=2,
            new_start=10,
            new_count=1,
            header="@@ -10,2 +10,1 @@",
            lines=lines,
        )

        deletions = [l for l in hunk.lines if l.line_type == "deletion"]
        assert len(deletions) == 2

    def test_hunk_function_context(self):
        """Test extracting function context from header."""
        from openmemory.api.tools.pr_workflow.pr_parser import PRHunk

        hunk = PRHunk(
            old_start=10,
            old_count=2,
            new_start=10,
            new_count=2,
            header="@@ -10,2 +10,2 @@ def process_data():",
            lines=[],
        )

        assert "def process_data():" in hunk.header


# =============================================================================
# PRFile Tests
# =============================================================================


class TestPRFile:
    """Tests for PRFile dataclass."""

    def test_file_creation(self):
        """Test creating a diff file."""
        from openmemory.api.tools.pr_workflow.pr_parser import PRFile, PRHunk

        hunks = [
            PRHunk(
                old_start=10,
                old_count=2,
                new_start=10,
                new_count=2,
                header="@@ -10,2 +10,2 @@",
                lines=[],
            )
        ]

        file = PRFile(
            old_path="src/main.py",
            new_path="src/main.py",
            status="modified",
            hunks=hunks,
        )

        assert file.old_path == "src/main.py"
        assert file.new_path == "src/main.py"
        assert file.status == "modified"
        assert len(file.hunks) == 1

    def test_new_file(self):
        """Test new file representation."""
        from openmemory.api.tools.pr_workflow.pr_parser import PRFile

        file = PRFile(
            old_path=None,
            new_path="src/new_module.py",
            status="added",
            hunks=[],
        )

        assert file.old_path is None
        assert file.new_path == "src/new_module.py"
        assert file.status == "added"

    def test_deleted_file(self):
        """Test deleted file representation."""
        from openmemory.api.tools.pr_workflow.pr_parser import PRFile

        file = PRFile(
            old_path="src/old_module.py",
            new_path=None,
            status="deleted",
            hunks=[],
        )

        assert file.old_path == "src/old_module.py"
        assert file.new_path is None
        assert file.status == "deleted"

    def test_renamed_file(self):
        """Test renamed file representation."""
        from openmemory.api.tools.pr_workflow.pr_parser import PRFile

        file = PRFile(
            old_path="src/old_name.py",
            new_path="src/new_name.py",
            status="renamed",
            hunks=[],
            similarity_index=90,
        )

        assert file.old_path == "src/old_name.py"
        assert file.new_path == "src/new_name.py"
        assert file.status == "renamed"
        assert file.similarity_index == 90

    def test_binary_file(self):
        """Test binary file representation."""
        from openmemory.api.tools.pr_workflow.pr_parser import PRFile

        file = PRFile(
            old_path=None,
            new_path="assets/image.png",
            status="added",
            hunks=[],
            is_binary=True,
        )

        assert file.is_binary is True

    def test_file_language_detection(self):
        """Test language detection from file extension."""
        from openmemory.api.tools.pr_workflow.pr_parser import PRFile

        py_file = PRFile(
            old_path="src/main.py",
            new_path="src/main.py",
            status="modified",
            hunks=[],
        )
        assert py_file.language == "python"

        ts_file = PRFile(
            old_path="src/app.ts",
            new_path="src/app.ts",
            status="modified",
            hunks=[],
        )
        assert ts_file.language == "typescript"

        java_file = PRFile(
            old_path="src/Main.java",
            new_path="src/Main.java",
            status="modified",
            hunks=[],
        )
        assert java_file.language == "java"

    def test_file_additions_total(self):
        """Test total additions count."""
        from openmemory.api.tools.pr_workflow.pr_parser import PRFile, PRHunk, PRLine

        hunks = [
            PRHunk(
                old_start=10,
                old_count=1,
                new_start=10,
                new_count=2,
                header="@@ -10,1 +10,2 @@",
                lines=[
                    PRLine("+new1", "addition", None, 10),
                    PRLine("+new2", "addition", None, 11),
                ],
            )
        ]

        file = PRFile(
            old_path="src/main.py",
            new_path="src/main.py",
            status="modified",
            hunks=hunks,
        )

        assert file.additions == 2

    def test_file_deletions_total(self):
        """Test total deletions count."""
        from openmemory.api.tools.pr_workflow.pr_parser import PRFile, PRHunk, PRLine

        hunks = [
            PRHunk(
                old_start=10,
                old_count=3,
                new_start=10,
                new_count=0,
                header="@@ -10,3 +10,0 @@",
                lines=[
                    PRLine("-old1", "deletion", 10, None),
                    PRLine("-old2", "deletion", 11, None),
                    PRLine("-old3", "deletion", 12, None),
                ],
            )
        ]

        file = PRFile(
            old_path="src/main.py",
            new_path="src/main.py",
            status="modified",
            hunks=hunks,
        )

        assert file.deletions == 3


# =============================================================================
# PRDiff Tests
# =============================================================================


class TestPRDiff:
    """Tests for PRDiff dataclass."""

    def test_diff_creation(self):
        """Test creating a complete diff."""
        from openmemory.api.tools.pr_workflow.pr_parser import PRDiff, PRFile

        files = [
            PRFile(
                old_path="src/main.py",
                new_path="src/main.py",
                status="modified",
                hunks=[],
            )
        ]

        diff = PRDiff(files=files)

        assert len(diff.files) == 1

    def test_diff_file_count(self):
        """Test counting files in diff."""
        from openmemory.api.tools.pr_workflow.pr_parser import PRDiff, PRFile

        files = [
            PRFile(
                old_path="src/a.py",
                new_path="src/a.py",
                status="modified",
                hunks=[],
            ),
            PRFile(
                old_path=None,
                new_path="src/b.py",
                status="added",
                hunks=[],
            ),
            PRFile(
                old_path="src/c.py",
                new_path=None,
                status="deleted",
                hunks=[],
            ),
        ]

        diff = PRDiff(files=files)

        assert diff.files_changed == 3
        assert diff.files_added == 1
        assert diff.files_deleted == 1
        assert diff.files_modified == 1

    def test_diff_total_changes(self):
        """Test counting total changes."""
        from openmemory.api.tools.pr_workflow.pr_parser import (
            PRDiff,
            PRFile,
            PRHunk,
            PRLine,
        )

        hunks = [
            PRHunk(
                old_start=10,
                old_count=2,
                new_start=10,
                new_count=3,
                header="@@ -10,2 +10,3 @@",
                lines=[
                    PRLine("-old", "deletion", 10, None),
                    PRLine("+new1", "addition", None, 10),
                    PRLine("+new2", "addition", None, 11),
                ],
            )
        ]

        files = [
            PRFile(
                old_path="src/main.py",
                new_path="src/main.py",
                status="modified",
                hunks=hunks,
            )
        ]

        diff = PRDiff(files=files)

        assert diff.total_additions == 2
        assert diff.total_deletions == 1

    def test_diff_by_language(self):
        """Test grouping files by language."""
        from openmemory.api.tools.pr_workflow.pr_parser import PRDiff, PRFile

        files = [
            PRFile("a.py", "a.py", "modified", []),
            PRFile("b.py", "b.py", "modified", []),
            PRFile("c.ts", "c.ts", "modified", []),
        ]

        diff = PRDiff(files=files)
        by_lang = diff.files_by_language

        assert len(by_lang.get("python", [])) == 2
        assert len(by_lang.get("typescript", [])) == 1


# =============================================================================
# DiffParser Tests
# =============================================================================


class TestDiffParser:
    """Tests for DiffParser."""

    def test_parse_simple_diff(self, simple_diff):
        """Test parsing a simple diff."""
        from openmemory.api.tools.pr_workflow.pr_parser import DiffParser

        parser = DiffParser()
        result = parser.parse(simple_diff)

        assert result is not None
        assert len(result.files) == 1
        assert result.files[0].old_path == "src/main.py"
        assert result.files[0].new_path == "src/main.py"
        assert result.files[0].status == "modified"

    def test_parse_multi_file_diff(self, multi_file_diff):
        """Test parsing a diff with multiple files."""
        from openmemory.api.tools.pr_workflow.pr_parser import DiffParser

        parser = DiffParser()
        result = parser.parse(multi_file_diff)

        assert len(result.files) == 2
        assert result.files[0].new_path == "src/main.py"
        assert result.files[1].new_path == "src/utils.py"
        assert result.files[1].status == "added"

    def test_parse_deleted_file(self, deleted_file_diff):
        """Test parsing a deleted file diff."""
        from openmemory.api.tools.pr_workflow.pr_parser import DiffParser

        parser = DiffParser()
        result = parser.parse(deleted_file_diff)

        assert len(result.files) == 1
        assert result.files[0].old_path == "src/old_module.py"
        assert result.files[0].status == "deleted"

    def test_parse_renamed_file(self, renamed_file_diff):
        """Test parsing a renamed file diff."""
        from openmemory.api.tools.pr_workflow.pr_parser import DiffParser

        parser = DiffParser()
        result = parser.parse(renamed_file_diff)

        assert len(result.files) == 1
        assert result.files[0].old_path == "src/old_name.py"
        assert result.files[0].new_path == "src/new_name.py"
        assert result.files[0].status == "renamed"
        assert result.files[0].similarity_index == 90

    def test_parse_multi_hunk_diff(self, multi_hunk_diff):
        """Test parsing a diff with multiple hunks."""
        from openmemory.api.tools.pr_workflow.pr_parser import DiffParser

        parser = DiffParser()
        result = parser.parse(multi_hunk_diff)

        assert len(result.files) == 1
        assert len(result.files[0].hunks) == 2

    def test_parse_binary_file(self, binary_file_diff):
        """Test parsing a binary file diff."""
        from openmemory.api.tools.pr_workflow.pr_parser import DiffParser

        parser = DiffParser()
        result = parser.parse(binary_file_diff)

        assert len(result.files) == 1
        assert result.files[0].is_binary is True

    def test_parse_hunk_header(self):
        """Test parsing hunk headers with line numbers."""
        from openmemory.api.tools.pr_workflow.pr_parser import DiffParser

        diff = """diff --git a/test.py b/test.py
index abc..def 100644
--- a/test.py
+++ b/test.py
@@ -10,3 +10,4 @@ def function():
     line1
+    added
     line2
"""

        parser = DiffParser()
        result = parser.parse(diff)

        hunk = result.files[0].hunks[0]
        assert hunk.old_start == 10
        assert hunk.old_count == 3
        assert hunk.new_start == 10
        assert hunk.new_count == 4

    def test_parse_lines_with_types(self, simple_diff):
        """Test parsing lines with correct types."""
        from openmemory.api.tools.pr_workflow.pr_parser import DiffParser

        parser = DiffParser()
        result = parser.parse(simple_diff)

        hunk = result.files[0].hunks[0]

        # Find addition and deletion lines
        additions = [l for l in hunk.lines if l.line_type == "addition"]
        deletions = [l for l in hunk.lines if l.line_type == "deletion"]
        context = [l for l in hunk.lines if l.line_type == "context"]

        assert len(additions) >= 1
        assert len(deletions) >= 1
        assert len(context) >= 1

    def test_parse_empty_diff(self):
        """Test parsing an empty diff."""
        from openmemory.api.tools.pr_workflow.pr_parser import DiffParser

        parser = DiffParser()
        result = parser.parse("")

        assert result is not None
        assert len(result.files) == 0

    def test_parse_invalid_diff(self):
        """Test parsing invalid diff content."""
        from openmemory.api.tools.pr_workflow.pr_parser import DiffParser, DiffParseError

        parser = DiffParser()

        # Invalid diff that looks like a diff but is malformed
        invalid = "not a valid diff at all"
        result = parser.parse(invalid)

        # Should return empty diff, not raise
        assert len(result.files) == 0

    def test_parse_no_newline_at_end(self):
        """Test parsing diff with no newline at end marker."""
        diff = """diff --git a/test.py b/test.py
index abc..def 100644
--- a/test.py
+++ b/test.py
@@ -1,2 +1,2 @@
 line1
-old
+new
\\ No newline at end of file
"""

        from openmemory.api.tools.pr_workflow.pr_parser import DiffParser

        parser = DiffParser()
        result = parser.parse(diff)

        assert len(result.files) == 1

    def test_parse_mode_changes(self):
        """Test parsing diff with mode changes."""
        diff = """diff --git a/script.sh b/script.sh
old mode 100644
new mode 100755
index abc..def
--- a/script.sh
+++ b/script.sh
@@ -1,1 +1,1 @@
-echo hello
+echo world
"""

        from openmemory.api.tools.pr_workflow.pr_parser import DiffParser

        parser = DiffParser()
        result = parser.parse(diff)

        assert len(result.files) == 1
        assert result.files[0].new_path == "script.sh"


# =============================================================================
# Edge Cases Tests
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and special scenarios."""

    def test_file_with_spaces_in_path(self):
        """Test parsing file with spaces in path."""
        diff = """diff --git a/src/my file.py b/src/my file.py
index abc..def 100644
--- a/src/my file.py
+++ b/src/my file.py
@@ -1,1 +1,1 @@
-old
+new
"""

        from openmemory.api.tools.pr_workflow.pr_parser import DiffParser

        parser = DiffParser()
        result = parser.parse(diff)

        assert result.files[0].new_path == "src/my file.py"

    def test_very_large_hunk(self):
        """Test parsing a very large hunk."""
        lines = "\n".join([f"+line{i}" for i in range(1000)])
        diff = f"""diff --git a/large.py b/large.py
new file mode 100644
index 0000000..abc1234
--- /dev/null
+++ b/large.py
@@ -0,0 +1,1000 @@
{lines}
"""

        from openmemory.api.tools.pr_workflow.pr_parser import DiffParser

        parser = DiffParser()
        result = parser.parse(diff)

        assert len(result.files) == 1
        assert result.files[0].additions >= 1000

    def test_unicode_content(self):
        """Test parsing diff with unicode content."""
        diff = """diff --git a/i18n.py b/i18n.py
index abc..def 100644
--- a/i18n.py
+++ b/i18n.py
@@ -1,1 +1,1 @@
-greeting = "Hello"
+greeting = "Bonjour \u4f60\u597d"
"""

        from openmemory.api.tools.pr_workflow.pr_parser import DiffParser

        parser = DiffParser()
        result = parser.parse(diff)

        assert len(result.files) == 1

    def test_multiple_rename_operations(self):
        """Test parsing multiple renames."""
        diff = """diff --git a/old1.py b/new1.py
similarity index 100%
rename from old1.py
rename to new1.py
diff --git a/old2.py b/new2.py
similarity index 100%
rename from old2.py
rename to new2.py
"""

        from openmemory.api.tools.pr_workflow.pr_parser import DiffParser

        parser = DiffParser()
        result = parser.parse(diff)

        assert len(result.files) == 2
        assert all(f.status == "renamed" for f in result.files)

    def test_copy_operation(self):
        """Test parsing copy operations."""
        diff = """diff --git a/original.py b/copy.py
similarity index 100%
copy from original.py
copy to copy.py
"""

        from openmemory.api.tools.pr_workflow.pr_parser import DiffParser

        parser = DiffParser()
        result = parser.parse(diff)

        assert len(result.files) == 1
        assert result.files[0].status == "copied"


# =============================================================================
# Helper Method Tests
# =============================================================================


class TestHelperMethods:
    """Tests for helper methods on data classes."""

    def test_prfile_path_property(self):
        """Test path property returns effective path."""
        from openmemory.api.tools.pr_workflow.pr_parser import PRFile

        # For modified file
        modified = PRFile("a.py", "a.py", "modified", [])
        assert modified.path == "a.py"

        # For added file
        added = PRFile(None, "new.py", "added", [])
        assert added.path == "new.py"

        # For deleted file
        deleted = PRFile("old.py", None, "deleted", [])
        assert deleted.path == "old.py"

    def test_prdiff_get_file_by_path(self):
        """Test getting file by path."""
        from openmemory.api.tools.pr_workflow.pr_parser import PRDiff, PRFile

        files = [
            PRFile("a.py", "a.py", "modified", []),
            PRFile("b.py", "b.py", "modified", []),
        ]
        diff = PRDiff(files=files)

        result = diff.get_file("a.py")
        assert result is not None
        assert result.new_path == "a.py"

        missing = diff.get_file("missing.py")
        assert missing is None

    def test_prdiff_filter_by_extension(self):
        """Test filtering files by extension."""
        from openmemory.api.tools.pr_workflow.pr_parser import PRDiff, PRFile

        files = [
            PRFile("a.py", "a.py", "modified", []),
            PRFile("b.ts", "b.ts", "modified", []),
            PRFile("c.py", "c.py", "modified", []),
        ]
        diff = PRDiff(files=files)

        py_files = diff.filter_by_extension(".py")
        assert len(py_files) == 2

    def test_prhunk_changed_lines_only(self):
        """Test getting only changed lines from hunk."""
        from openmemory.api.tools.pr_workflow.pr_parser import PRHunk, PRLine

        lines = [
            PRLine(" context", "context", 1, 1),
            PRLine("-deleted", "deletion", 2, None),
            PRLine("+added", "addition", None, 2),
            PRLine(" more context", "context", 3, 3),
        ]

        hunk = PRHunk(
            old_start=1,
            old_count=3,
            new_start=1,
            new_count=3,
            header="@@ -1,3 +1,3 @@",
            lines=lines,
        )

        changed = hunk.changed_lines
        assert len(changed) == 2
        assert all(l.line_type in ("addition", "deletion") for l in changed)
