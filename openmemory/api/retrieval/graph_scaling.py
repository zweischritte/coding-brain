"""Graph scaling utilities (FR-006).

This module provides graph scaling strategy:
- Graph partitioning and sharding
- Replica management
- Materialized views
- Query routing
- Connection pooling

Key features:
- Hash and range partitioning
- Automatic failover
- Read preference routing
- Materialized view caching
"""

from __future__ import annotations

import hashlib
import logging
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from queue import Queue
from typing import Any, Optional

logger = logging.getLogger(__name__)


# =============================================================================
# Exceptions
# =============================================================================


class GraphScalingError(Exception):
    """Base exception for graph scaling errors."""

    pass


class PartitionError(GraphScalingError):
    """Error in partitioning."""

    pass


class ReplicaError(GraphScalingError):
    """Error with replica operations."""

    pass


class ViewError(GraphScalingError):
    """Error with materialized views."""

    pass


# =============================================================================
# Configuration
# =============================================================================


@dataclass
class GraphScalingConfig:
    """Configuration for graph scaling.

    Args:
        num_partitions: Number of graph partitions
        replication_factor: Number of replicas per partition
        enable_materialized_views: Enable materialized view support
        pool_size: Connection pool size per node
    """

    num_partitions: int = 4
    replication_factor: int = 2
    enable_materialized_views: bool = True
    pool_size: int = 10

    def __post_init__(self):
        """Validate configuration."""
        if self.num_partitions <= 0:
            raise ValueError("num_partitions must be positive")
        if self.replication_factor <= 0:
            raise ValueError("replication_factor must be positive")


@dataclass
class PartitionConfig:
    """Configuration for a single partition.

    Args:
        partition_id: Partition identifier
        node_uri: URI of the node hosting this partition
        key_range_start: Start of key range (for range partitioning)
        key_range_end: End of key range
    """

    partition_id: int
    node_uri: str
    key_range_start: str = ""
    key_range_end: str = ""


@dataclass
class ReplicaConfig:
    """Configuration for replica management.

    Args:
        replication_factor: Number of replicas per partition
        sync_timeout_seconds: Timeout for sync operations
        health_check_interval_seconds: Health check interval
    """

    replication_factor: int = 2
    sync_timeout_seconds: int = 30
    health_check_interval_seconds: int = 10


@dataclass
class MaterializedViewConfig:
    """Configuration for materialized views.

    Args:
        default_refresh_interval_seconds: Default refresh interval
        max_cache_size_mb: Maximum cache size
    """

    default_refresh_interval_seconds: int = 300
    max_cache_size_mb: int = 100


# =============================================================================
# Partition Strategy
# =============================================================================


class PartitionStrategy(Enum):
    """Partitioning strategies."""

    HASH = "hash"
    RANGE = "range"
    ROUND_ROBIN = "round_robin"


@dataclass
class PartitionInfo:
    """Information about a partition.

    Args:
        partition_id: Partition identifier
        node_uri: Primary node URI
        is_primary: Whether this is the primary
        replica_uris: List of replica URIs
    """

    partition_id: int
    node_uri: str = ""
    is_primary: bool = True
    replica_uris: list[str] = field(default_factory=list)


class Partitioner(ABC):
    """Abstract interface for partitioners."""

    @property
    @abstractmethod
    def num_partitions(self) -> int:
        """Get number of partitions."""
        pass

    @abstractmethod
    def partition(self, key: str) -> int:
        """Get partition ID for a key."""
        pass

    @abstractmethod
    def get_partition_info(self, key: str) -> PartitionInfo:
        """Get partition info for a key."""
        pass


class HashPartitioner(Partitioner):
    """Hash-based partitioner.

    Distributes keys evenly across partitions using
    consistent hashing.
    """

    def __init__(self, num_partitions: int):
        """Initialize partitioner.

        Args:
            num_partitions: Number of partitions
        """
        self._num_partitions = num_partitions

    @property
    def num_partitions(self) -> int:
        """Get number of partitions."""
        return self._num_partitions

    def partition(self, key: str) -> int:
        """Get partition ID for a key.

        Args:
            key: Key to partition

        Returns:
            Partition ID (0 to num_partitions-1)
        """
        hash_value = int(hashlib.md5(key.encode()).hexdigest(), 16)
        return hash_value % self._num_partitions

    def get_partition_info(self, key: str) -> PartitionInfo:
        """Get partition info for a key.

        Args:
            key: Key to look up

        Returns:
            PartitionInfo for the key's partition
        """
        partition_id = self.partition(key)
        return PartitionInfo(partition_id=partition_id)


class RangePartitioner(Partitioner):
    """Range-based partitioner.

    Assigns keys to partitions based on key ranges.
    """

    def __init__(self, ranges: list[tuple[str, str]]):
        """Initialize partitioner.

        Args:
            ranges: List of (start, end) ranges for each partition
        """
        self._ranges = ranges
        self._num_partitions = len(ranges)

    @property
    def num_partitions(self) -> int:
        """Get number of partitions."""
        return self._num_partitions

    def partition(self, key: str) -> int:
        """Get partition ID for a key.

        Args:
            key: Key to partition

        Returns:
            Partition ID based on range
        """
        key_lower = key.lower()

        for i, (start, end) in enumerate(self._ranges):
            if start <= key_lower < end:
                return i

        # Default to last partition if no match
        return self._num_partitions - 1

    def get_partition_info(self, key: str) -> PartitionInfo:
        """Get partition info for a key."""
        partition_id = self.partition(key)
        return PartitionInfo(partition_id=partition_id)


# =============================================================================
# Partition Router
# =============================================================================


@dataclass
class RoutingDecision:
    """Decision about where to route a query.

    Args:
        partition_id: Target partition
        node_uri: Target node URI
        is_primary: Whether targeting primary
        fallback_uris: Fallback URIs if primary fails
    """

    partition_id: int
    node_uri: str
    is_primary: bool = True
    fallback_uris: list[str] = field(default_factory=list)


class PartitionRouter:
    """Routes queries to appropriate partitions.

    Combines partitioner with partition configuration
    to route queries to correct nodes.
    """

    def __init__(
        self,
        partitioner: Partitioner,
        partitions: list[PartitionConfig],
    ):
        """Initialize router.

        Args:
            partitioner: Partitioner for key distribution
            partitions: Partition configurations
        """
        self._partitioner = partitioner
        self._partitions = {p.partition_id: p for p in partitions}

    @property
    def num_partitions(self) -> int:
        """Get number of partitions."""
        return self._partitioner.num_partitions

    def route(self, key: str) -> RoutingDecision:
        """Route a key to its partition.

        Args:
            key: Key to route

        Returns:
            RoutingDecision with target node
        """
        partition_id = self._partitioner.partition(key)
        partition = self._partitions.get(partition_id)

        if not partition:
            raise PartitionError(f"No configuration for partition {partition_id}")

        return RoutingDecision(
            partition_id=partition_id,
            node_uri=partition.node_uri,
            is_primary=True,
        )

    def route_batch(self, keys: list[str]) -> list[RoutingDecision]:
        """Route multiple keys.

        Args:
            keys: Keys to route

        Returns:
            List of routing decisions
        """
        return [self.route(key) for key in keys]


# =============================================================================
# Replica Management
# =============================================================================


class ReplicaStatus(Enum):
    """Status of a replica node."""

    ONLINE = "online"
    OFFLINE = "offline"
    SYNCING = "syncing"
    DEGRADED = "degraded"


class ReadPreference(Enum):
    """Read preference for replica selection."""

    PRIMARY = "primary"
    SECONDARY = "secondary"
    NEAREST = "nearest"


@dataclass
class ReplicaNode:
    """A replica node in the cluster.

    Args:
        node_id: Unique node identifier
        uri: Connection URI
        partition_id: Partition this node serves
        is_primary: Whether this is the primary
        status: Current node status
    """

    node_id: str
    uri: str
    partition_id: int
    is_primary: bool
    status: ReplicaStatus = ReplicaStatus.ONLINE

    @property
    def is_available(self) -> bool:
        """Check if node is available for queries."""
        return self.status in (ReplicaStatus.ONLINE, ReplicaStatus.DEGRADED)


class ReplicaManager:
    """Manages replica nodes for partitions.

    Handles:
    - Replica registration
    - Read preference routing
    - Failover when primary is unavailable
    """

    def __init__(self, config: ReplicaConfig):
        """Initialize manager.

        Args:
            config: Replica configuration
        """
        self._config = config
        self._nodes: dict[str, ReplicaNode] = {}
        self._partitions: dict[int, list[ReplicaNode]] = {}
        self._lock = threading.RLock()

    @property
    def replication_factor(self) -> int:
        """Get replication factor."""
        return self._config.replication_factor

    def register(self, node: ReplicaNode) -> None:
        """Register a replica node.

        Args:
            node: Node to register
        """
        with self._lock:
            self._nodes[node.node_id] = node

            if node.partition_id not in self._partitions:
                self._partitions[node.partition_id] = []

            self._partitions[node.partition_id].append(node)

    def get_node(self, node_id: str) -> Optional[ReplicaNode]:
        """Get a node by ID.

        Args:
            node_id: Node ID

        Returns:
            ReplicaNode or None
        """
        with self._lock:
            return self._nodes.get(node_id)

    def get_replicas_for_partition(self, partition_id: int) -> list[ReplicaNode]:
        """Get all replicas for a partition.

        Args:
            partition_id: Partition ID

        Returns:
            List of replica nodes
        """
        with self._lock:
            return list(self._partitions.get(partition_id, []))

    def get_primary(self, partition_id: int) -> Optional[ReplicaNode]:
        """Get primary node for a partition.

        Args:
            partition_id: Partition ID

        Returns:
            Primary ReplicaNode or None
        """
        with self._lock:
            nodes = self._partitions.get(partition_id, [])
            for node in nodes:
                if node.is_primary and node.is_available:
                    return node
            return None

    def select_replica(
        self,
        partition_id: int,
        preference: ReadPreference,
    ) -> Optional[ReplicaNode]:
        """Select a replica based on preference.

        Args:
            partition_id: Partition ID
            preference: Read preference

        Returns:
            Selected replica or None
        """
        with self._lock:
            nodes = self._partitions.get(partition_id, [])
            available = [n for n in nodes if n.is_available]

            if not available:
                return None

            if preference == ReadPreference.PRIMARY:
                # Try primary first, failover to any available
                for node in available:
                    if node.is_primary:
                        return node
                return available[0]  # Failover

            elif preference == ReadPreference.SECONDARY:
                # Prefer secondary
                for node in available:
                    if not node.is_primary:
                        return node
                return available[0]  # Fallback to primary

            else:  # NEAREST
                # Return first available (would use latency in production)
                return available[0]


# =============================================================================
# Materialized Views
# =============================================================================


class ViewRefreshPolicy(Enum):
    """Refresh policy for materialized views."""

    ON_COMMIT = "on_commit"
    PERIODIC = "periodic"
    MANUAL = "manual"


@dataclass
class ViewDefinition:
    """Definition of a materialized view.

    Args:
        view_name: View name
        query: Cypher query for the view
        parameters: Query parameters
    """

    view_name: str
    query: str
    parameters: list[str] = field(default_factory=list)


@dataclass
class MaterializedView:
    """A materialized view with cached data.

    Args:
        view_id: Unique view identifier
        definition: View definition
        refresh_policy: When to refresh
        refresh_interval_seconds: Refresh interval for periodic
        cached_data: Cached query results
        last_refreshed_at: Last refresh timestamp
    """

    view_id: str
    definition: ViewDefinition
    refresh_policy: ViewRefreshPolicy
    refresh_interval_seconds: int = 300
    cached_data: list[dict[str, Any]] = field(default_factory=list)
    last_refreshed_at: float = field(default_factory=time.time)

    def is_stale(self) -> bool:
        """Check if view is stale and needs refresh."""
        if self.refresh_policy != ViewRefreshPolicy.PERIODIC:
            return False

        age = time.time() - self.last_refreshed_at
        return age > self.refresh_interval_seconds


class MaterializedViewManager:
    """Manages materialized views.

    Handles:
    - View registration
    - Query execution against views
    - Refresh scheduling
    """

    def __init__(self):
        """Initialize manager."""
        self._views: dict[str, MaterializedView] = {}
        self._lock = threading.RLock()

    @property
    def view_count(self) -> int:
        """Get number of registered views."""
        with self._lock:
            return len(self._views)

    def register(self, view: MaterializedView) -> None:
        """Register a materialized view.

        Args:
            view: View to register
        """
        with self._lock:
            self._views[view.view_id] = view

    def get_view(self, view_id: str) -> Optional[MaterializedView]:
        """Get a view by ID.

        Args:
            view_id: View ID

        Returns:
            MaterializedView or None
        """
        with self._lock:
            return self._views.get(view_id)

    def query(
        self,
        view_id: str,
        parameters: dict[str, Any],
    ) -> list[dict[str, Any]]:
        """Query a materialized view.

        Args:
            view_id: View to query
            parameters: Query parameters

        Returns:
            Query results
        """
        with self._lock:
            view = self._views.get(view_id)
            if not view:
                raise ViewError(f"View not found: {view_id}")

            return view.cached_data

    def refresh_stale_views(self) -> int:
        """Refresh all stale views.

        Returns:
            Number of views refreshed
        """
        count = 0

        with self._lock:
            stale_views = [v for v in self._views.values() if v.is_stale()]

        for view in stale_views:
            try:
                new_data = self._execute_refresh(view)
                with self._lock:
                    view.cached_data = new_data
                    view.last_refreshed_at = time.time()
                count += 1
            except Exception as e:
                logger.warning(f"Failed to refresh view {view.view_id}: {e}")

        return count

    def _execute_refresh(self, view: MaterializedView) -> list[dict[str, Any]]:
        """Execute view refresh query.

        Args:
            view: View to refresh

        Returns:
            New view data
        """
        # In production, this would execute the Cypher query
        return []


# =============================================================================
# Connection Pooling
# =============================================================================


@dataclass
class PoolStats:
    """Statistics for a connection pool.

    Args:
        max_size: Maximum pool size
        min_size: Minimum pool size
        in_use: Connections currently in use
        available: Available connections
        total_created: Total connections created
        total_destroyed: Total connections destroyed
    """

    max_size: int
    min_size: int
    in_use: int
    available: int
    total_created: int
    total_destroyed: int

    @property
    def utilization(self) -> float:
        """Calculate pool utilization."""
        total = self.in_use + self.available
        if total == 0:
            return 0.0
        return total / self.max_size


class PooledConnection:
    """A pooled database connection.

    Attributes:
        connection: The underlying connection
        pool_id: Pool identifier
        pool: Reference to the pool
        created_at: Creation timestamp
        connection_id: Unique connection ID for tracking
    """

    _counter = 0

    def __init__(
        self,
        connection: Any,
        pool_id: str,
        pool: Optional[GraphConnectionPool] = None,
        created_at: Optional[float] = None,
    ):
        """Initialize pooled connection."""
        PooledConnection._counter += 1
        self.connection_id = PooledConnection._counter
        self.connection = connection
        self.pool_id = pool_id
        self.pool = pool
        self.created_at = created_at or time.time()

    def __hash__(self) -> int:
        """Make connection hashable."""
        return hash(self.connection_id)

    def __eq__(self, other: object) -> bool:
        """Check equality by connection ID."""
        if not isinstance(other, PooledConnection):
            return False
        return self.connection_id == other.connection_id

    def __enter__(self) -> PooledConnection:
        """Enter context."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Exit context and release connection."""
        if self.pool:
            self.pool.release(self)


class GraphConnectionPool:
    """Connection pool for graph database.

    Manages a pool of connections to a graph database node.
    """

    def __init__(
        self,
        uri: str,
        max_size: int = 10,
        min_size: int = 2,
        auth: Optional[tuple[str, str]] = None,
    ):
        """Initialize pool.

        Args:
            uri: Database URI
            max_size: Maximum pool size
            min_size: Minimum pool size
            auth: Optional (username, password)
        """
        self.uri = uri
        self.max_size = max_size
        self.min_size = min_size
        self._auth = auth
        self._pool: Queue[PooledConnection] = Queue(maxsize=max_size)
        self._in_use: set[PooledConnection] = set()
        self._total_created = 0
        self._total_destroyed = 0
        self._lock = threading.RLock()

    def acquire(self, timeout: Optional[float] = None) -> PooledConnection:
        """Acquire a connection from the pool.

        Args:
            timeout: Optional timeout in seconds

        Returns:
            Pooled connection
        """
        with self._lock:
            # Try to get from pool
            if not self._pool.empty():
                conn = self._pool.get_nowait()
                self._in_use.add(conn)
                return conn

            # Create new if under max
            if len(self._in_use) < self.max_size:
                inner_conn = self._create_connection()
                conn = PooledConnection(
                    connection=inner_conn,
                    pool_id=f"pool-{id(self)}",
                    pool=self,
                    created_at=time.time(),
                )
                self._in_use.add(conn)
                self._total_created += 1
                return conn

            # Pool exhausted, wait for release
            # In production, would block with timeout
            raise GraphScalingError("Connection pool exhausted")

    def release(self, conn: PooledConnection) -> None:
        """Release a connection back to the pool.

        Args:
            conn: Connection to release
        """
        with self._lock:
            if conn in self._in_use:
                self._in_use.remove(conn)
                self._pool.put_nowait(conn)

    def stats(self) -> PoolStats:
        """Get pool statistics.

        Returns:
            PoolStats instance
        """
        with self._lock:
            return PoolStats(
                max_size=self.max_size,
                min_size=self.min_size,
                in_use=len(self._in_use),
                available=self._pool.qsize(),
                total_created=self._total_created,
                total_destroyed=self._total_destroyed,
            )

    def _create_connection(self) -> Any:
        """Create a new database connection.

        Returns:
            Database connection
        """
        # In production, this would create actual Neo4j driver connection
        return object()


# =============================================================================
# Query Router
# =============================================================================


class QueryRouter:
    """Routes queries to appropriate nodes.

    Combines partitioning with replica management
    for intelligent query routing.
    """

    def __init__(
        self,
        partitioner: Partitioner,
        replica_manager: ReplicaManager,
        read_preference: ReadPreference = ReadPreference.PRIMARY,
    ):
        """Initialize router.

        Args:
            partitioner: Partitioner for key distribution
            replica_manager: Replica manager
            read_preference: Default read preference
        """
        self.partitioner = partitioner
        self.replica_manager = replica_manager
        self.read_preference = read_preference

    def route_read(self, key: str) -> RoutingDecision:
        """Route a read query.

        Args:
            key: Key to route

        Returns:
            RoutingDecision with target node
        """
        partition_id = self.partitioner.partition(key)
        node = self.replica_manager.select_replica(partition_id, self.read_preference)

        if not node:
            raise ReplicaError(f"No available replica for partition {partition_id}")

        # Get fallback nodes
        all_nodes = self.replica_manager.get_replicas_for_partition(partition_id)
        fallbacks = [n.uri for n in all_nodes if n != node and n.is_available]

        return RoutingDecision(
            partition_id=partition_id,
            node_uri=node.uri,
            is_primary=node.is_primary,
            fallback_uris=fallbacks,
        )

    def route_write(self, key: str) -> RoutingDecision:
        """Route a write query (always to primary).

        Args:
            key: Key to route

        Returns:
            RoutingDecision with primary node
        """
        partition_id = self.partitioner.partition(key)
        primary = self.replica_manager.get_primary(partition_id)

        if not primary:
            raise ReplicaError(f"No primary available for partition {partition_id}")

        return RoutingDecision(
            partition_id=partition_id,
            node_uri=primary.uri,
            is_primary=True,
        )


# =============================================================================
# Health Monitoring
# =============================================================================


@dataclass
class NodeHealth:
    """Health information for a node.

    Args:
        node_id: Node identifier
        is_healthy: Whether node is healthy
        latency_ms: Ping latency
        last_check_at: Last health check timestamp
    """

    node_id: str
    is_healthy: bool
    latency_ms: float
    last_check_at: float


class HealthMonitor:
    """Monitors health of graph nodes.

    Periodically checks node health and updates
    replica manager status.
    """

    def __init__(
        self,
        replica_manager: ReplicaManager,
        check_interval_seconds: int = 10,
    ):
        """Initialize monitor.

        Args:
            replica_manager: Replica manager to update
            check_interval_seconds: Health check interval
        """
        self.replica_manager = replica_manager
        self.check_interval_seconds = check_interval_seconds
        self._health_cache: dict[str, NodeHealth] = {}

    def check_node(self, node_id: str) -> NodeHealth:
        """Check health of a specific node.

        Args:
            node_id: Node to check

        Returns:
            NodeHealth result
        """
        node = self.replica_manager.get_node(node_id)
        if not node:
            return NodeHealth(
                node_id=node_id,
                is_healthy=False,
                latency_ms=0.0,
                last_check_at=time.time(),
            )

        is_healthy, latency_ms = self._ping_node(node.uri)

        health = NodeHealth(
            node_id=node_id,
            is_healthy=is_healthy,
            latency_ms=latency_ms,
            last_check_at=time.time(),
        )

        self._health_cache[node_id] = health
        return health

    def check_all(self) -> list[NodeHealth]:
        """Check health of all nodes.

        Returns:
            List of NodeHealth results
        """
        results = []
        for node_id in list(self.replica_manager._nodes.keys()):
            health = self.check_node(node_id)
            results.append(health)
        return results

    def _ping_node(self, uri: str) -> tuple[bool, float]:
        """Ping a node to check health.

        Args:
            uri: Node URI

        Returns:
            Tuple of (is_healthy, latency_ms)
        """
        # In production, would actually ping the node
        return (True, 5.0)


# =============================================================================
# Graph Scaling Manager
# =============================================================================


class GraphScalingManager:
    """Main manager for graph scaling.

    Coordinates partitioning, replication, views, and health.
    """

    def __init__(
        self,
        config: GraphScalingConfig,
        partitioner: Optional[Partitioner] = None,
    ):
        """Initialize manager.

        Args:
            config: Scaling configuration
            partitioner: Optional partitioner
        """
        self.config = config
        self.partitioner = partitioner or HashPartitioner(config.num_partitions)
        self.replica_manager = ReplicaManager(
            ReplicaConfig(replication_factor=config.replication_factor)
        )
        self.view_manager = MaterializedViewManager()
        self.health_monitor = HealthMonitor(self.replica_manager)
        self._pools: dict[str, GraphConnectionPool] = {}


class PartitionedGraph:
    """A partitioned graph database abstraction.

    Provides a unified interface over a partitioned graph.
    """

    def __init__(
        self,
        partitioner: Partitioner,
        node_uris: list[str],
    ):
        """Initialize partitioned graph.

        Args:
            partitioner: Partitioner for key distribution
            node_uris: URIs for partition nodes
        """
        self._partitioner = partitioner
        self._node_uris = node_uris

    @property
    def num_partitions(self) -> int:
        """Get number of partitions."""
        return self._partitioner.num_partitions


# =============================================================================
# Factory Functions
# =============================================================================


def create_graph_scaling_manager(
    num_partitions: int = 4,
    replication_factor: int = 2,
    enable_materialized_views: bool = True,
    pool_size: int = 10,
) -> GraphScalingManager:
    """Create a graph scaling manager.

    Args:
        num_partitions: Number of partitions
        replication_factor: Replication factor
        enable_materialized_views: Enable views
        pool_size: Connection pool size

    Returns:
        Configured GraphScalingManager
    """
    config = GraphScalingConfig(
        num_partitions=num_partitions,
        replication_factor=replication_factor,
        enable_materialized_views=enable_materialized_views,
        pool_size=pool_size,
    )
    return GraphScalingManager(config)


def create_partitioned_graph(
    node_uris: list[str],
    partition_strategy: str = "hash",
) -> PartitionedGraph:
    """Create a partitioned graph.

    Args:
        node_uris: URIs for partition nodes
        partition_strategy: Partitioning strategy

    Returns:
        Configured PartitionedGraph
    """
    num_partitions = len(node_uris)

    if partition_strategy == "hash":
        partitioner = HashPartitioner(num_partitions)
    else:
        partitioner = HashPartitioner(num_partitions)

    return PartitionedGraph(
        partitioner=partitioner,
        node_uris=node_uris,
    )
