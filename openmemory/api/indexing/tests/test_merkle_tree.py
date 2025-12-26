"""Tests for Merkle tree incremental indexer.

Following TDD: Write tests first, then implement.
Covers:
- Content-based hashing for files
- Merkle tree structure for directory hierarchies
- Change detection (additions, modifications, deletions)
- Incremental updates (only re-index changed files)
- Persistence and restoration of tree state
- Atomic transactions (all-or-nothing updates)
"""

import pytest
import hashlib
from pathlib import Path
from typing import Optional
from unittest.mock import Mock, patch
import json
import tempfile

from openmemory.api.indexing.merkle_tree import (
    # Core types
    HashValue,
    FileNode,
    DirectoryNode,
    MerkleTree,
    ChangeType,
    Change,
    ChangeSet,
    # State management
    TreeState,
    StateStore,
    MemoryStateStore,
    FileStateStore,
    # Incremental indexer
    IncrementalIndexer,
    IndexerConfig,
    IndexTransaction,
    TransactionStatus,
    # Exceptions
    TransactionFailedError,
    StateCorruptedError,
    # Factory
    create_indexer,
)


# =============================================================================
# Hash Value Tests
# =============================================================================


class TestHashValue:
    """Tests for HashValue type."""

    def test_hash_from_content(self):
        """Create hash from content bytes."""
        content = b"hello world"
        h = HashValue.from_content(content)
        assert h is not None
        assert len(h.hex) == 64  # SHA-256 = 32 bytes = 64 hex chars

    def test_hash_deterministic(self):
        """Same content produces same hash."""
        content = b"test content"
        h1 = HashValue.from_content(content)
        h2 = HashValue.from_content(content)
        assert h1 == h2

    def test_different_content_different_hash(self):
        """Different content produces different hash."""
        h1 = HashValue.from_content(b"content a")
        h2 = HashValue.from_content(b"content b")
        assert h1 != h2

    def test_hash_from_children(self):
        """Create hash from child hashes."""
        h1 = HashValue.from_content(b"child1")
        h2 = HashValue.from_content(b"child2")
        parent = HashValue.from_children([h1, h2])
        assert parent is not None

    def test_hash_from_children_order_independent(self):
        """Child order does NOT affect parent hash (sorted internally for consistency)."""
        h1 = HashValue.from_content(b"a")
        h2 = HashValue.from_content(b"b")
        parent1 = HashValue.from_children([h1, h2])
        parent2 = HashValue.from_children([h2, h1])
        # Hashes are sorted internally so order doesn't matter
        assert parent1 == parent2

    def test_hash_serialization(self):
        """Hash can be serialized and deserialized."""
        h = HashValue.from_content(b"test")
        serialized = h.hex
        restored = HashValue.from_hex(serialized)
        assert h == restored

    def test_empty_content_hash(self):
        """Empty content has a valid hash."""
        h = HashValue.from_content(b"")
        assert h is not None
        assert len(h.hex) == 64


# =============================================================================
# File Node Tests
# =============================================================================


class TestFileNode:
    """Tests for FileNode representing a file in the tree."""

    def test_create_file_node(self):
        """Create FileNode from file path and content."""
        node = FileNode(
            path=Path("test.py"),
            content_hash=HashValue.from_content(b"def foo(): pass"),
            size=15,
            modified_time=1234567890.0,
        )
        assert node.path == Path("test.py")
        assert node.size == 15
        assert node.modified_time == 1234567890.0

    def test_file_node_equality(self):
        """FileNodes are equal if paths and hashes match."""
        h = HashValue.from_content(b"content")
        n1 = FileNode(Path("a.py"), h, 10, 1.0)
        n2 = FileNode(Path("a.py"), h, 10, 1.0)
        assert n1 == n2

    def test_file_node_from_path(self, tmp_path):
        """Create FileNode from actual file path."""
        test_file = tmp_path / "test.py"
        test_file.write_text("def foo(): pass")

        node = FileNode.from_path(test_file)
        assert node is not None
        assert node.path == test_file
        assert node.size > 0

    def test_file_node_tracks_mtime(self, tmp_path):
        """FileNode tracks modification time."""
        test_file = tmp_path / "test.py"
        test_file.write_text("v1")
        node1 = FileNode.from_path(test_file)

        # Modify file
        test_file.write_text("v2 with more content")
        node2 = FileNode.from_path(test_file)

        assert node1.content_hash != node2.content_hash


# =============================================================================
# Directory Node Tests
# =============================================================================


class TestDirectoryNode:
    """Tests for DirectoryNode representing a directory in the tree."""

    def test_create_directory_node(self):
        """Create DirectoryNode from path and children."""
        child = FileNode(
            path=Path("src/a.py"),
            content_hash=HashValue.from_content(b"content"),
            size=10,
            modified_time=1.0,
        )
        dir_node = DirectoryNode(
            path=Path("src"),
            children={"a.py": child},
        )
        assert dir_node.path == Path("src")
        assert "a.py" in dir_node.children

    def test_directory_hash_from_children(self):
        """Directory hash is derived from children."""
        child1 = FileNode(
            Path("a.py"),
            HashValue.from_content(b"a"),
            5,
            1.0,
        )
        child2 = FileNode(
            Path("b.py"),
            HashValue.from_content(b"b"),
            5,
            1.0,
        )
        dir_node = DirectoryNode(Path("src"), {"a.py": child1, "b.py": child2})
        assert dir_node.content_hash is not None

    def test_empty_directory(self):
        """Empty directory has a hash."""
        dir_node = DirectoryNode(Path("empty"), {})
        assert dir_node.content_hash is not None

    def test_nested_directories(self):
        """Support nested directory structures."""
        file_node = FileNode(
            Path("src/lib/util.py"),
            HashValue.from_content(b"util"),
            10,
            1.0,
        )
        lib_dir = DirectoryNode(Path("src/lib"), {"util.py": file_node})
        src_dir = DirectoryNode(Path("src"), {"lib": lib_dir})

        assert src_dir.content_hash is not None
        assert "lib" in src_dir.children


# =============================================================================
# Merkle Tree Tests
# =============================================================================


class TestMerkleTree:
    """Tests for MerkleTree structure."""

    def test_create_empty_tree(self):
        """Create empty Merkle tree."""
        tree = MerkleTree()
        assert tree.root is None
        assert tree.is_empty

    def test_build_tree_from_directory(self, tmp_path):
        """Build Merkle tree from directory."""
        # Create test structure
        (tmp_path / "a.py").write_text("file a")
        (tmp_path / "b.py").write_text("file b")
        subdir = tmp_path / "sub"
        subdir.mkdir()
        (subdir / "c.py").write_text("file c")

        tree = MerkleTree.from_directory(tmp_path)
        assert tree.root is not None
        assert not tree.is_empty

    def test_tree_hash_changes_with_content(self, tmp_path):
        """Tree root hash changes when content changes."""
        (tmp_path / "a.py").write_text("version 1")

        tree1 = MerkleTree.from_directory(tmp_path)
        hash1 = tree1.root_hash

        (tmp_path / "a.py").write_text("version 2")
        tree2 = MerkleTree.from_directory(tmp_path)
        hash2 = tree2.root_hash

        assert hash1 != hash2

    def test_tree_hash_stable(self, tmp_path):
        """Tree hash is stable for unchanged content."""
        (tmp_path / "a.py").write_text("content")

        tree1 = MerkleTree.from_directory(tmp_path)
        tree2 = MerkleTree.from_directory(tmp_path)

        assert tree1.root_hash == tree2.root_hash

    def test_get_file_node(self, tmp_path):
        """Get specific file node from tree."""
        (tmp_path / "a.py").write_text("content")
        tree = MerkleTree.from_directory(tmp_path)

        node = tree.get_node(Path("a.py"))
        assert node is not None
        assert isinstance(node, FileNode)

    def test_filter_by_extensions(self, tmp_path):
        """Build tree with extension filter."""
        (tmp_path / "a.py").write_text("python")
        (tmp_path / "b.ts").write_text("typescript")
        (tmp_path / "c.txt").write_text("text")

        tree = MerkleTree.from_directory(tmp_path, extensions=[".py", ".ts"])

        # Should include .py and .ts, exclude .txt
        assert tree.get_node(Path("a.py")) is not None
        assert tree.get_node(Path("b.ts")) is not None
        assert tree.get_node(Path("c.txt")) is None


# =============================================================================
# Change Detection Tests
# =============================================================================


class TestChangeDetection:
    """Tests for change detection between tree states."""

    def test_detect_no_changes(self, tmp_path):
        """No changes when trees are identical."""
        (tmp_path / "a.py").write_text("content")

        tree1 = MerkleTree.from_directory(tmp_path)
        tree2 = MerkleTree.from_directory(tmp_path)

        changes = tree1.diff(tree2)
        assert len(changes) == 0

    def test_detect_file_added(self, tmp_path):
        """Detect added file."""
        tree1 = MerkleTree.from_directory(tmp_path)

        (tmp_path / "new.py").write_text("new file")
        tree2 = MerkleTree.from_directory(tmp_path)

        changes = tree1.diff(tree2)
        assert len(changes) == 1
        assert changes[0].change_type == ChangeType.ADDED
        assert changes[0].path == Path("new.py")

    def test_detect_file_modified(self, tmp_path):
        """Detect modified file."""
        (tmp_path / "a.py").write_text("v1")
        tree1 = MerkleTree.from_directory(tmp_path)

        (tmp_path / "a.py").write_text("v2")
        tree2 = MerkleTree.from_directory(tmp_path)

        changes = tree1.diff(tree2)
        assert len(changes) == 1
        assert changes[0].change_type == ChangeType.MODIFIED
        assert changes[0].path == Path("a.py")

    def test_detect_file_deleted(self, tmp_path):
        """Detect deleted file."""
        (tmp_path / "a.py").write_text("content")
        tree1 = MerkleTree.from_directory(tmp_path)

        (tmp_path / "a.py").unlink()
        tree2 = MerkleTree.from_directory(tmp_path)

        changes = tree1.diff(tree2)
        assert len(changes) == 1
        assert changes[0].change_type == ChangeType.DELETED
        assert changes[0].path == Path("a.py")

    def test_detect_multiple_changes(self, tmp_path):
        """Detect multiple changes at once."""
        (tmp_path / "a.py").write_text("a")
        (tmp_path / "b.py").write_text("b")
        tree1 = MerkleTree.from_directory(tmp_path)

        (tmp_path / "a.py").write_text("a modified")
        (tmp_path / "b.py").unlink()
        (tmp_path / "c.py").write_text("new")
        tree2 = MerkleTree.from_directory(tmp_path)

        changes = tree1.diff(tree2)
        assert len(changes) == 3

        change_types = {c.change_type for c in changes}
        assert ChangeType.ADDED in change_types
        assert ChangeType.MODIFIED in change_types
        assert ChangeType.DELETED in change_types

    def test_detect_directory_added(self, tmp_path):
        """Detect added directory with files."""
        tree1 = MerkleTree.from_directory(tmp_path)

        subdir = tmp_path / "new_dir"
        subdir.mkdir()
        (subdir / "file.py").write_text("content")
        tree2 = MerkleTree.from_directory(tmp_path)

        changes = tree1.diff(tree2)
        # Should see the new file, not just the directory
        file_changes = [c for c in changes if c.change_type == ChangeType.ADDED]
        assert len(file_changes) >= 1


# =============================================================================
# Change Type and ChangeSet Tests
# =============================================================================


class TestChangeType:
    """Tests for ChangeType enum."""

    def test_change_type_values(self):
        """ChangeType has expected values."""
        assert ChangeType.ADDED.value == "added"
        assert ChangeType.MODIFIED.value == "modified"
        assert ChangeType.DELETED.value == "deleted"


class TestChangeSet:
    """Tests for ChangeSet collection."""

    def test_empty_changeset(self):
        """Empty changeset."""
        cs = ChangeSet([])
        assert len(cs) == 0
        assert not cs.has_changes

    def test_changeset_iteration(self):
        """Iterate over changes."""
        changes = [
            Change(ChangeType.ADDED, Path("a.py")),
            Change(ChangeType.MODIFIED, Path("b.py")),
        ]
        cs = ChangeSet(changes)
        assert len(cs) == 2
        assert cs.has_changes

    def test_changeset_filter_by_type(self):
        """Filter changes by type."""
        changes = [
            Change(ChangeType.ADDED, Path("a.py")),
            Change(ChangeType.MODIFIED, Path("b.py")),
            Change(ChangeType.ADDED, Path("c.py")),
        ]
        cs = ChangeSet(changes)

        added = cs.filter(ChangeType.ADDED)
        assert len(added) == 2


# =============================================================================
# State Store Tests
# =============================================================================


class TestTreeState:
    """Tests for TreeState serialization."""

    def test_state_from_tree(self, tmp_path):
        """Create state from tree."""
        (tmp_path / "a.py").write_text("content")
        tree = MerkleTree.from_directory(tmp_path)

        state = TreeState.from_tree(tree)
        assert state is not None
        assert state.root_hash == tree.root_hash

    def test_state_serialization(self, tmp_path):
        """Serialize and deserialize state."""
        (tmp_path / "a.py").write_text("content")
        tree = MerkleTree.from_directory(tmp_path)
        state = TreeState.from_tree(tree)

        json_data = state.to_json()
        restored = TreeState.from_json(json_data)

        assert restored.root_hash == state.root_hash


class TestMemoryStateStore:
    """Tests for in-memory state store."""

    def test_save_and_load(self, tmp_path):
        """Save and load state."""
        store = MemoryStateStore()
        (tmp_path / "a.py").write_text("content")
        tree = MerkleTree.from_directory(tmp_path)

        store.save(tree)
        loaded = store.load()

        assert loaded is not None
        assert loaded.root_hash == tree.root_hash

    def test_load_empty(self):
        """Load from empty store."""
        store = MemoryStateStore()
        loaded = store.load()
        assert loaded is None

    def test_clear(self, tmp_path):
        """Clear stored state."""
        store = MemoryStateStore()
        (tmp_path / "a.py").write_text("content")
        tree = MerkleTree.from_directory(tmp_path)

        store.save(tree)
        store.clear()
        assert store.load() is None


class TestFileStateStore:
    """Tests for file-based state store."""

    def test_save_and_load(self, tmp_path):
        """Save and load state to file."""
        state_file = tmp_path / ".index_state.json"
        store = FileStateStore(state_file)

        project_dir = tmp_path / "project"
        project_dir.mkdir()
        (project_dir / "a.py").write_text("content")
        tree = MerkleTree.from_directory(project_dir)

        store.save(tree)
        assert state_file.exists()

        loaded = store.load()
        assert loaded is not None
        assert loaded.root_hash == tree.root_hash

    def test_corrupted_state_file(self, tmp_path):
        """Handle corrupted state file."""
        state_file = tmp_path / ".index_state.json"
        state_file.write_text("invalid json {{{")

        store = FileStateStore(state_file)
        with pytest.raises(StateCorruptedError):
            store.load()


# =============================================================================
# Incremental Indexer Tests
# =============================================================================


class TestIndexerConfig:
    """Tests for IndexerConfig."""

    def test_default_config(self):
        """Default configuration."""
        config = IndexerConfig()
        assert config.extensions == [".py", ".ts", ".tsx", ".java"]
        assert config.ignore_patterns == ["__pycache__", "node_modules", ".git"]
        assert config.max_file_size == 1_000_000

    def test_custom_config(self):
        """Custom configuration."""
        config = IndexerConfig(
            extensions=[".py"],
            ignore_patterns=["build"],
            max_file_size=500_000,
        )
        assert config.extensions == [".py"]


class TestIncrementalIndexer:
    """Tests for IncrementalIndexer."""

    @pytest.fixture
    def indexer(self, tmp_path):
        """Create indexer for test directory."""
        config = IndexerConfig(extensions=[".py", ".ts"])
        return create_indexer(tmp_path, config)

    def test_initial_scan(self, tmp_path, indexer):
        """Initial scan indexes all files."""
        (tmp_path / "a.py").write_text("def a(): pass")
        (tmp_path / "b.py").write_text("def b(): pass")

        result = indexer.scan()
        assert result.files_indexed == 2
        assert result.files_skipped == 0

    def test_incremental_scan(self, tmp_path, indexer):
        """Incremental scan only processes changes."""
        (tmp_path / "a.py").write_text("def a(): pass")

        # Initial scan
        indexer.scan()

        # Add a new file
        (tmp_path / "b.py").write_text("def b(): pass")

        # Incremental scan
        result = indexer.scan()
        assert result.files_indexed == 1  # Only new file

    def test_modification_detection(self, tmp_path, indexer):
        """Detect and re-index modified files."""
        (tmp_path / "a.py").write_text("v1")

        indexer.scan()

        (tmp_path / "a.py").write_text("v2")

        result = indexer.scan()
        assert result.files_indexed == 1
        assert result.changes.filter(ChangeType.MODIFIED)

    def test_deletion_detection(self, tmp_path, indexer):
        """Detect deleted files."""
        (tmp_path / "a.py").write_text("content")

        indexer.scan()

        (tmp_path / "a.py").unlink()

        result = indexer.scan()
        assert len(result.changes.filter(ChangeType.DELETED)) == 1

    def test_ignore_patterns(self, tmp_path, indexer):
        """Ignore files matching patterns."""
        (tmp_path / "a.py").write_text("good")
        pycache = tmp_path / "__pycache__"
        pycache.mkdir()
        (pycache / "b.pyc").write_text("bytecode")

        result = indexer.scan()
        assert result.files_indexed == 1  # Only a.py

    def test_extension_filter(self, tmp_path, indexer):
        """Only index files with matching extensions."""
        (tmp_path / "a.py").write_text("python")
        (tmp_path / "b.ts").write_text("typescript")
        (tmp_path / "c.txt").write_text("text")

        result = indexer.scan()
        assert result.files_indexed == 2  # py and ts only

    def test_max_file_size(self, tmp_path):
        """Skip files exceeding size limit."""
        config = IndexerConfig(max_file_size=10)
        indexer = create_indexer(tmp_path, config)

        (tmp_path / "small.py").write_text("x=1")
        (tmp_path / "large.py").write_text("x" * 100)

        result = indexer.scan()
        assert result.files_indexed == 1
        assert result.files_skipped == 1


# =============================================================================
# Transaction Tests
# =============================================================================


class TestIndexTransaction:
    """Tests for IndexTransaction atomicity."""

    @pytest.fixture
    def indexer(self, tmp_path):
        """Create indexer for tests."""
        return create_indexer(tmp_path)

    def test_transaction_commit(self, tmp_path, indexer):
        """Successful transaction commits changes."""
        (tmp_path / "a.py").write_text("content")

        with indexer.transaction() as tx:
            tx.index_file(tmp_path / "a.py")

        # State should be saved after commit
        assert indexer.state_store.load() is not None

    def test_transaction_rollback(self, tmp_path, indexer):
        """Failed transaction rolls back changes."""
        (tmp_path / "a.py").write_text("content")

        try:
            with indexer.transaction() as tx:
                tx.index_file(tmp_path / "a.py")
                raise RuntimeError("Simulated failure")
        except RuntimeError:
            pass

        # State should not be saved
        assert indexer.state_store.load() is None

    def test_transaction_status(self, tmp_path, indexer):
        """Transaction tracks status."""
        (tmp_path / "a.py").write_text("content")

        with indexer.transaction() as tx:
            assert tx.status == TransactionStatus.PENDING
            tx.index_file(tmp_path / "a.py")

        assert tx.status == TransactionStatus.COMMITTED

    def test_nested_transactions_not_allowed(self, tmp_path, indexer):
        """Nested transactions raise error."""
        (tmp_path / "a.py").write_text("content")

        with indexer.transaction():
            with pytest.raises(TransactionFailedError):
                with indexer.transaction():
                    pass

    def test_transaction_batch_operations(self, tmp_path, indexer):
        """Transaction can batch multiple operations."""
        for i in range(5):
            (tmp_path / f"file{i}.py").write_text(f"content {i}")

        with indexer.transaction() as tx:
            for i in range(5):
                tx.index_file(tmp_path / f"file{i}.py")

        state = indexer.state_store.load()
        assert state is not None


# =============================================================================
# Factory Function Tests
# =============================================================================


class TestFactory:
    """Tests for create_indexer factory."""

    def test_create_default_indexer(self, tmp_path):
        """Create indexer with defaults."""
        indexer = create_indexer(tmp_path)
        assert indexer is not None
        assert indexer.root_path == tmp_path

    def test_create_with_config(self, tmp_path):
        """Create indexer with custom config."""
        config = IndexerConfig(extensions=[".py"])
        indexer = create_indexer(tmp_path, config)
        assert indexer.config.extensions == [".py"]

    def test_create_with_custom_store(self, tmp_path):
        """Create indexer with custom state store."""
        store = MemoryStateStore()
        indexer = create_indexer(tmp_path, state_store=store)
        assert indexer.state_store is store


# =============================================================================
# Edge Cases
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_empty_directory(self, tmp_path):
        """Handle empty directory."""
        indexer = create_indexer(tmp_path)
        result = indexer.scan()
        assert result.files_indexed == 0

    def test_deeply_nested_structure(self, tmp_path):
        """Handle deeply nested directories."""
        deep_path = tmp_path
        for i in range(20):
            deep_path = deep_path / f"level{i}"
        deep_path.mkdir(parents=True)
        (deep_path / "deep.py").write_text("content")

        indexer = create_indexer(tmp_path)
        result = indexer.scan()
        assert result.files_indexed == 1

    def test_symlinks_ignored(self, tmp_path):
        """Symlinks are ignored by default."""
        (tmp_path / "real.py").write_text("content")
        (tmp_path / "link.py").symlink_to(tmp_path / "real.py")

        indexer = create_indexer(tmp_path)
        result = indexer.scan()
        # Should only count real file, not symlink
        assert result.files_indexed == 1

    def test_unicode_filenames(self, tmp_path):
        """Handle unicode in filenames."""
        (tmp_path / "日本語.py").write_text("# Japanese")
        (tmp_path / "émoji_🎉.py").write_text("# Emoji")

        indexer = create_indexer(tmp_path)
        result = indexer.scan()
        assert result.files_indexed == 2

    def test_binary_files_skipped(self, tmp_path):
        """Binary files are skipped."""
        (tmp_path / "code.py").write_text("valid python")
        (tmp_path / "binary.py").write_bytes(b"\x00\x01\x02\x03")

        indexer = create_indexer(tmp_path)
        result = indexer.scan()
        assert result.files_indexed == 1
        assert result.files_skipped == 1

    def test_permission_error_handling(self, tmp_path):
        """Handle permission errors gracefully."""
        (tmp_path / "readable.py").write_text("content")

        # Create unreadable file (skip on Windows)
        import platform
        if platform.system() != "Windows":
            unreadable = tmp_path / "unreadable.py"
            unreadable.write_text("secret")
            unreadable.chmod(0o000)

            indexer = create_indexer(tmp_path)
            result = indexer.scan()

            # Cleanup
            unreadable.chmod(0o644)

            assert result.files_indexed == 1
            assert result.files_skipped >= 1


# =============================================================================
# Integration Tests
# =============================================================================


class TestIntegration:
    """Integration tests for full workflow."""

    def test_full_workflow(self, tmp_path):
        """Test complete index-modify-reindex workflow."""
        # Initial state
        (tmp_path / "a.py").write_text("def a(): pass")
        (tmp_path / "b.py").write_text("def b(): pass")

        indexer = create_indexer(tmp_path)

        # Initial scan
        result1 = indexer.scan()
        assert result1.files_indexed == 2

        # Modify and add
        (tmp_path / "a.py").write_text("def a(): return 1")
        (tmp_path / "c.py").write_text("def c(): pass")

        # Incremental scan
        result2 = indexer.scan()
        assert result2.files_indexed == 2  # Modified + added
        assert len(result2.changes.filter(ChangeType.MODIFIED)) == 1
        assert len(result2.changes.filter(ChangeType.ADDED)) == 1

        # Delete
        (tmp_path / "b.py").unlink()

        # Another scan
        result3 = indexer.scan()
        assert len(result3.changes.filter(ChangeType.DELETED)) == 1

    def test_persistence_across_sessions(self, tmp_path):
        """State persists across indexer instances."""
        state_file = tmp_path / ".index_state.json"
        config = IndexerConfig()

        # Session 1
        (tmp_path / "a.py").write_text("content")
        indexer1 = create_indexer(
            tmp_path,
            config,
            state_store=FileStateStore(state_file),
        )
        indexer1.scan()

        # Session 2 (new indexer instance)
        (tmp_path / "b.py").write_text("new file")
        indexer2 = create_indexer(
            tmp_path,
            config,
            state_store=FileStateStore(state_file),
        )
        result = indexer2.scan()

        # Should only see the new file
        assert result.files_indexed == 1
        assert len(result.changes.filter(ChangeType.ADDED)) == 1
