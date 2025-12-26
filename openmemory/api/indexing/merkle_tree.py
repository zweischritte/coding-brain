"""Merkle tree incremental indexer.

This module provides:
- Content-based hashing for files
- Merkle tree structure for directory hierarchies
- Change detection (additions, modifications, deletions)
- Incremental updates (only re-index changed files)
- Persistence and restoration of tree state
- Atomic transactions (all-or-nothing updates)
"""

from __future__ import annotations

import hashlib
import json
import logging
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Iterator, Optional, Union

logger = logging.getLogger(__name__)


# =============================================================================
# Exceptions
# =============================================================================


class TransactionFailedError(Exception):
    """Raised when a transaction fails."""

    pass


class StateCorruptedError(Exception):
    """Raised when state file is corrupted."""

    pass


# =============================================================================
# Hash Value
# =============================================================================


@dataclass(frozen=True)
class HashValue:
    """Immutable hash value for content identification."""

    _hex: str

    @property
    def hex(self) -> str:
        """Return hex representation of hash."""
        return self._hex

    @classmethod
    def from_content(cls, content: bytes) -> "HashValue":
        """Create hash from content bytes using SHA-256."""
        digest = hashlib.sha256(content).hexdigest()
        return cls(_hex=digest)

    @classmethod
    def from_children(cls, children: list["HashValue"]) -> "HashValue":
        """Create hash from child hashes."""
        combined = b"".join(h._hex.encode() for h in sorted(children, key=lambda h: h._hex))
        return cls.from_content(combined)

    @classmethod
    def from_hex(cls, hex_str: str) -> "HashValue":
        """Create HashValue from hex string."""
        return cls(_hex=hex_str)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, HashValue):
            return False
        return self._hex == other._hex

    def __hash__(self) -> int:
        return hash(self._hex)


# =============================================================================
# Tree Nodes
# =============================================================================


@dataclass
class FileNode:
    """A file node in the Merkle tree."""

    path: Path
    content_hash: HashValue
    size: int
    modified_time: float

    @classmethod
    def from_path(cls, file_path: Path) -> "FileNode":
        """Create FileNode from actual file."""
        stat = file_path.stat()
        content = file_path.read_bytes()
        return cls(
            path=file_path,
            content_hash=HashValue.from_content(content),
            size=stat.st_size,
            modified_time=stat.st_mtime,
        )

    def to_dict(self) -> dict:
        """Serialize to dictionary."""
        return {
            "path": str(self.path),
            "content_hash": self.content_hash.hex,
            "size": self.size,
            "modified_time": self.modified_time,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "FileNode":
        """Deserialize from dictionary."""
        return cls(
            path=Path(data["path"]),
            content_hash=HashValue.from_hex(data["content_hash"]),
            size=data["size"],
            modified_time=data["modified_time"],
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, FileNode):
            return False
        return self.path == other.path and self.content_hash == other.content_hash


@dataclass
class DirectoryNode:
    """A directory node in the Merkle tree."""

    path: Path
    children: dict[str, Union["DirectoryNode", FileNode]]
    _hash: Optional[HashValue] = field(default=None, init=False)

    @property
    def content_hash(self) -> HashValue:
        """Compute hash from children."""
        if self._hash is None:
            if not self.children:
                self._hash = HashValue.from_content(b"empty_dir")
            else:
                child_hashes = []
                for name in sorted(self.children.keys()):
                    child = self.children[name]
                    if isinstance(child, FileNode):
                        child_hashes.append(child.content_hash)
                    else:
                        child_hashes.append(child.content_hash)
                self._hash = HashValue.from_children(child_hashes)
        return self._hash

    def to_dict(self) -> dict:
        """Serialize to dictionary."""
        children_data = {}
        for name, child in self.children.items():
            if isinstance(child, FileNode):
                children_data[name] = {"type": "file", "data": child.to_dict()}
            else:
                children_data[name] = {"type": "dir", "data": child.to_dict()}

        return {
            "path": str(self.path),
            "children": children_data,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "DirectoryNode":
        """Deserialize from dictionary."""
        children = {}
        for name, child_data in data.get("children", {}).items():
            if child_data["type"] == "file":
                children[name] = FileNode.from_dict(child_data["data"])
            else:
                children[name] = DirectoryNode.from_dict(child_data["data"])

        return cls(path=Path(data["path"]), children=children)


# =============================================================================
# Change Types
# =============================================================================


class ChangeType(Enum):
    """Type of change detected."""

    ADDED = "added"
    MODIFIED = "modified"
    DELETED = "deleted"


@dataclass
class Change:
    """A detected change."""

    change_type: ChangeType
    path: Path
    old_hash: Optional[HashValue] = None
    new_hash: Optional[HashValue] = None


@dataclass
class ChangeSet:
    """Collection of changes."""

    changes: list[Change]

    def __len__(self) -> int:
        return len(self.changes)

    def __iter__(self) -> Iterator[Change]:
        return iter(self.changes)

    @property
    def has_changes(self) -> bool:
        """Check if there are any changes."""
        return len(self.changes) > 0

    def filter(self, change_type: ChangeType) -> list[Change]:
        """Filter changes by type."""
        return [c for c in self.changes if c.change_type == change_type]


# =============================================================================
# Merkle Tree
# =============================================================================


class MerkleTree:
    """Merkle tree for directory contents."""

    def __init__(self, root: Optional[DirectoryNode] = None):
        self.root = root

    @property
    def is_empty(self) -> bool:
        """Check if tree is empty."""
        return self.root is None

    @property
    def root_hash(self) -> Optional[HashValue]:
        """Get root hash."""
        return self.root.content_hash if self.root else None

    @classmethod
    def from_directory(
        cls,
        directory: Path,
        extensions: Optional[list[str]] = None,
        ignore_patterns: Optional[list[str]] = None,
        max_file_size: Optional[int] = None,
        skip_binary: bool = True,
    ) -> "MerkleTree":
        """Build Merkle tree from directory."""
        if ignore_patterns is None:
            ignore_patterns = []

        root = cls._build_node(
            directory, directory, extensions, ignore_patterns, max_file_size, skip_binary
        )
        return cls(root)

    @classmethod
    def _build_node(
        cls,
        base_path: Path,
        current_path: Path,
        extensions: Optional[list[str]],
        ignore_patterns: list[str],
        max_file_size: Optional[int],
        skip_binary: bool,
    ) -> DirectoryNode:
        """Recursively build tree nodes."""
        children: dict[str, Union[DirectoryNode, FileNode]] = {}

        try:
            entries = list(current_path.iterdir())
        except PermissionError:
            return DirectoryNode(path=current_path.relative_to(base_path), children={})

        for entry in sorted(entries):
            # Skip ignored patterns
            if any(pattern in str(entry) for pattern in ignore_patterns):
                continue

            # Skip symlinks
            if entry.is_symlink():
                continue

            rel_path = entry.relative_to(base_path)

            if entry.is_dir():
                child_node = cls._build_node(
                    base_path, entry, extensions, ignore_patterns, max_file_size, skip_binary
                )
                if child_node.children:  # Only add non-empty directories
                    children[entry.name] = child_node
            elif entry.is_file():
                # Check extension filter
                if extensions and entry.suffix not in extensions:
                    continue

                try:
                    # Check file size
                    stat = entry.stat()
                    if max_file_size is not None and stat.st_size > max_file_size:
                        continue

                    # Check for binary content
                    content = entry.read_bytes()
                    if skip_binary and b"\x00" in content[:8192]:
                        continue

                    file_node = FileNode(
                        path=rel_path,
                        content_hash=HashValue.from_content(content),
                        size=stat.st_size,
                        modified_time=stat.st_mtime,
                    )
                    children[entry.name] = file_node
                except (PermissionError, OSError):
                    continue

        return DirectoryNode(
            path=current_path.relative_to(base_path) if current_path != base_path else Path("."),
            children=children,
        )

    def get_node(self, path: Path) -> Optional[Union[DirectoryNode, FileNode]]:
        """Get node at path."""
        if self.root is None:
            return None

        parts = path.parts
        current: Union[DirectoryNode, FileNode] = self.root

        for part in parts:
            if isinstance(current, FileNode):
                return None
            if part not in current.children:
                return None
            current = current.children[part]

        return current

    def diff(self, other: "MerkleTree") -> list[Change]:
        """Compute differences between trees."""
        changes = []
        self._diff_nodes(self.root, other.root, changes)
        return changes

    def _diff_nodes(
        self,
        old_node: Optional[Union[DirectoryNode, FileNode]],
        new_node: Optional[Union[DirectoryNode, FileNode]],
        changes: list[Change],
    ) -> None:
        """Recursively diff nodes."""
        # Get all file nodes from each tree
        old_files = self._collect_files(old_node) if old_node else {}
        new_files = self._collect_files(new_node) if new_node else {}

        all_paths = set(old_files.keys()) | set(new_files.keys())

        for path in all_paths:
            old_file = old_files.get(path)
            new_file = new_files.get(path)

            if old_file is None and new_file is not None:
                changes.append(
                    Change(
                        change_type=ChangeType.ADDED,
                        path=path,
                        new_hash=new_file.content_hash,
                    )
                )
            elif old_file is not None and new_file is None:
                changes.append(
                    Change(
                        change_type=ChangeType.DELETED,
                        path=path,
                        old_hash=old_file.content_hash,
                    )
                )
            elif old_file is not None and new_file is not None:
                if old_file.content_hash != new_file.content_hash:
                    changes.append(
                        Change(
                            change_type=ChangeType.MODIFIED,
                            path=path,
                            old_hash=old_file.content_hash,
                            new_hash=new_file.content_hash,
                        )
                    )

    def _collect_files(self, node: Union[DirectoryNode, FileNode]) -> dict[Path, FileNode]:
        """Collect all file nodes."""
        files: dict[Path, FileNode] = {}

        if isinstance(node, FileNode):
            files[node.path] = node
        else:
            for child in node.children.values():
                files.update(self._collect_files(child))

        return files


# =============================================================================
# State Management
# =============================================================================


@dataclass
class TreeState:
    """Serializable tree state."""

    root_hash: Optional[HashValue]
    tree_data: Optional[dict]

    @classmethod
    def from_tree(cls, tree: MerkleTree) -> "TreeState":
        """Create state from tree."""
        return cls(
            root_hash=tree.root_hash,
            tree_data=tree.root.to_dict() if tree.root else None,
        )

    def to_json(self) -> str:
        """Serialize to JSON."""
        return json.dumps(
            {
                "root_hash": self.root_hash.hex if self.root_hash else None,
                "tree_data": self.tree_data,
            }
        )

    @classmethod
    def from_json(cls, json_str: str) -> "TreeState":
        """Deserialize from JSON."""
        data = json.loads(json_str)
        root_hash = HashValue.from_hex(data["root_hash"]) if data["root_hash"] else None
        return cls(
            root_hash=root_hash,
            tree_data=data["tree_data"],
        )

    def to_tree(self) -> MerkleTree:
        """Reconstruct tree from state."""
        if self.tree_data is None:
            return MerkleTree()
        root = DirectoryNode.from_dict(self.tree_data)
        return MerkleTree(root)


class StateStore:
    """Abstract state store interface."""

    def save(self, tree: MerkleTree) -> None:
        """Save tree state."""
        raise NotImplementedError

    def load(self) -> Optional[MerkleTree]:
        """Load tree state."""
        raise NotImplementedError

    def clear(self) -> None:
        """Clear stored state."""
        raise NotImplementedError


class MemoryStateStore(StateStore):
    """In-memory state store."""

    def __init__(self):
        self._state: Optional[TreeState] = None

    def save(self, tree: MerkleTree) -> None:
        """Save tree state to memory."""
        self._state = TreeState.from_tree(tree)

    def load(self) -> Optional[MerkleTree]:
        """Load tree state from memory."""
        if self._state is None:
            return None
        return self._state.to_tree()

    def clear(self) -> None:
        """Clear stored state."""
        self._state = None


class FileStateStore(StateStore):
    """File-based state store."""

    def __init__(self, path: Path):
        self.path = path

    def save(self, tree: MerkleTree) -> None:
        """Save tree state to file."""
        state = TreeState.from_tree(tree)
        self.path.write_text(state.to_json())

    def load(self) -> Optional[MerkleTree]:
        """Load tree state from file."""
        if not self.path.exists():
            return None

        try:
            json_str = self.path.read_text()
            state = TreeState.from_json(json_str)
            return state.to_tree()
        except (json.JSONDecodeError, KeyError) as e:
            raise StateCorruptedError(f"Invalid state file: {e}")

    def clear(self) -> None:
        """Clear stored state."""
        if self.path.exists():
            self.path.unlink()


# =============================================================================
# Transaction
# =============================================================================


class TransactionStatus(Enum):
    """Transaction status."""

    PENDING = "pending"
    COMMITTED = "committed"
    ROLLED_BACK = "rolled_back"


@dataclass
class IndexTransaction:
    """Atomic transaction for index operations."""

    indexer: "IncrementalIndexer"
    status: TransactionStatus = TransactionStatus.PENDING
    _indexed_files: list[Path] = field(default_factory=list)

    def index_file(self, path: Path) -> None:
        """Add file to transaction."""
        self._indexed_files.append(path)

    def commit(self) -> None:
        """Commit transaction."""
        self.status = TransactionStatus.COMMITTED
        self.indexer._commit_transaction(self)

    def rollback(self) -> None:
        """Rollback transaction."""
        self.status = TransactionStatus.ROLLED_BACK


# =============================================================================
# Scan Result
# =============================================================================


@dataclass
class ScanResult:
    """Result of a scan operation."""

    files_indexed: int
    files_skipped: int
    changes: ChangeSet
    tree: MerkleTree


# =============================================================================
# Incremental Indexer
# =============================================================================


@dataclass
class IndexerConfig:
    """Configuration for incremental indexer."""

    extensions: list[str] = field(default_factory=lambda: [".py", ".ts", ".tsx", ".java"])
    ignore_patterns: list[str] = field(default_factory=lambda: ["__pycache__", "node_modules", ".git"])
    max_file_size: int = 1_000_000


class IncrementalIndexer:
    """Incremental indexer using Merkle tree for change detection."""

    def __init__(
        self,
        root_path: Path,
        config: Optional[IndexerConfig] = None,
        state_store: Optional[StateStore] = None,
    ):
        self.root_path = root_path
        self.config = config or IndexerConfig()
        self.state_store = state_store or MemoryStateStore()
        self._in_transaction = False
        self._pending_tree: Optional[MerkleTree] = None

    def scan(self) -> ScanResult:
        """Scan directory and return changes."""
        # Load previous state
        old_tree = self.state_store.load()

        # Build new tree with filtering
        new_tree = self._build_filtered_tree()

        # Compute changes
        if old_tree is None:
            changes = self._all_files_as_added(new_tree)
        else:
            changes = old_tree.diff(new_tree)

        # Count files
        files_indexed = len([c for c in changes if c.change_type != ChangeType.DELETED])
        files_skipped = self._count_skipped_files()

        # Save new state
        self.state_store.save(new_tree)

        return ScanResult(
            files_indexed=files_indexed,
            files_skipped=files_skipped,
            changes=ChangeSet(changes),
            tree=new_tree,
        )

    def _build_filtered_tree(self) -> MerkleTree:
        """Build tree with config filters applied."""
        return MerkleTree.from_directory(
            self.root_path,
            extensions=self.config.extensions,
            ignore_patterns=self.config.ignore_patterns,
            max_file_size=self.config.max_file_size,
            skip_binary=True,
        )

    def _all_files_as_added(self, tree: MerkleTree) -> list[Change]:
        """Create ADDED changes for all files in tree."""
        if tree.root is None:
            return []

        changes = []
        files = tree._collect_files(tree.root)
        for path, file_node in files.items():
            changes.append(
                Change(
                    change_type=ChangeType.ADDED,
                    path=path,
                    new_hash=file_node.content_hash,
                )
            )
        return changes

    def _count_skipped_files(self) -> int:
        """Count files skipped due to size or binary content."""
        skipped = 0
        for file_path in self.root_path.rglob("*"):
            if not file_path.is_file():
                continue

            # Check extension
            if file_path.suffix not in self.config.extensions:
                continue

            # Check ignore patterns
            if any(p in str(file_path) for p in self.config.ignore_patterns):
                continue

            # Check size
            try:
                if file_path.stat().st_size > self.config.max_file_size:
                    skipped += 1
                    continue

                # Check binary
                content = file_path.read_bytes()[:8192]
                if b"\x00" in content:
                    skipped += 1
            except (PermissionError, OSError):
                skipped += 1

        return skipped

    @contextmanager
    def transaction(self) -> Iterator[IndexTransaction]:
        """Create atomic transaction context."""
        if self._in_transaction:
            raise TransactionFailedError("Nested transactions not allowed")

        self._in_transaction = True
        tx = IndexTransaction(indexer=self)

        try:
            yield tx
            if tx.status == TransactionStatus.PENDING:
                tx.commit()
        except Exception:
            tx.rollback()
            raise
        finally:
            self._in_transaction = False

    def _commit_transaction(self, tx: IndexTransaction) -> None:
        """Commit transaction and save state."""
        # Build and save tree during commit
        tree = self._build_filtered_tree()
        self.state_store.save(tree)


# =============================================================================
# Factory Function
# =============================================================================


def create_indexer(
    root_path: Path,
    config: Optional[IndexerConfig] = None,
    state_store: Optional[StateStore] = None,
) -> IncrementalIndexer:
    """Create an incremental indexer."""
    return IncrementalIndexer(
        root_path=root_path,
        config=config,
        state_store=state_store,
    )
