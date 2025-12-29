"""
Neo4j Graph Data Science (GDS) Operations for OpenMemory.

Provides advanced graph algorithms using Neo4j's GDS library:
- PageRank: Identify most influential entities
- Community Detection: Find clusters of related memories
- Node Similarity: Find similar entities beyond direct connections
- FastRP Embeddings: Generate graph-based embeddings for ML
- Centrality: Betweenness, closeness, degree centrality

Requires Neo4j GDS plugin to be installed. Set NEO4J_PLUGINS=["graph-data-science"]
in docker-compose.yml or Neo4j configuration.

Usage:
    from app.graph.gds_operations import get_gds_operations

    gds = get_gds_operations()
    if gds:
        # Run PageRank on entities
        rankings = gds.entity_pagerank(user_id="grischadallmer")

        # Detect communities
        communities = gds.detect_memory_communities(user_id="grischadallmer")

        # Find similar entities
        similar = gds.find_similar_entities(user_id="grischadallmer", entity_name="BMG")
"""

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class GDSConfig:
    """Configuration for GDS operations."""

    # PageRank parameters
    pagerank_damping_factor: float = 0.85
    pagerank_max_iterations: int = 20
    pagerank_tolerance: float = 0.0001

    # Community detection parameters
    louvain_max_iterations: int = 10
    louvain_include_intermediate: bool = False

    # Node similarity parameters
    similarity_top_k: int = 10
    similarity_min_similarity: float = 0.5

    # FastRP embedding parameters
    fastrp_embedding_dimension: int = 128
    fastrp_iteration_weights: List[float] = None

    def __post_init__(self):
        if self.fastrp_iteration_weights is None:
            self.fastrp_iteration_weights = [0.0, 1.0, 1.0]


class GDSCypherBuilder:
    """Builds Cypher queries for GDS operations."""

    # =========================================================================
    # Graph Projection Queries
    # =========================================================================

    @staticmethod
    def project_entity_graph_query() -> str:
        """
        Project entity co-mention graph for GDS algorithms.

        Creates an in-memory graph of entities connected by OM_CO_MENTIONED edges.
        """
        return """
        CALL gds.graph.project(
            $graphName,
            {
                OM_Entity: {
                    label: 'OM_Entity',
                    properties: []
                }
            },
            {
                OM_CO_MENTIONED: {
                    type: 'OM_CO_MENTIONED',
                    orientation: 'UNDIRECTED',
                    properties: {
                        weight: {
                            property: 'count',
                            defaultValue: 1.0
                        }
                    }
                }
            }
        )
        YIELD graphName, nodeCount, relationshipCount
        RETURN graphName, nodeCount, relationshipCount
        """

    @staticmethod
    def project_memory_graph_query() -> str:
        """
        Project memory similarity graph for GDS algorithms.

        Creates an in-memory graph of memories connected by OM_SIMILAR edges.
        """
        return """
        CALL gds.graph.project(
            $graphName,
            {
                OM_Memory: {
                    label: 'OM_Memory',
                    properties: []
                }
            },
            {
                OM_SIMILAR: {
                    type: 'OM_SIMILAR',
                    orientation: 'UNDIRECTED',
                    properties: {
                        weight: {
                            property: 'score',
                            defaultValue: 0.5
                        }
                    }
                }
            }
        )
        YIELD graphName, nodeCount, relationshipCount
        RETURN graphName, nodeCount, relationshipCount
        """

    @staticmethod
    def project_full_graph_query() -> str:
        """
        Project full metadata graph including memories, entities, and dimensions.

        Creates a comprehensive in-memory graph for advanced analytics.
        """
        return """
        CALL gds.graph.project(
            $graphName,
            [
                'OM_Memory',
                'OM_Entity',
                'OM_Tag',
                'OM_Category',
                'OM_Scope',
                'OM_ArtifactType',
                'OM_ArtifactRef',
                'OM_Evidence',
                'OM_App'
            ],
            {
                OM_ABOUT: {type: 'OM_ABOUT', orientation: 'UNDIRECTED'},
                OM_TAGGED: {type: 'OM_TAGGED', orientation: 'UNDIRECTED'},
                OM_IN_CATEGORY: {type: 'OM_IN_CATEGORY', orientation: 'UNDIRECTED'},
                OM_IN_SCOPE: {type: 'OM_IN_SCOPE', orientation: 'UNDIRECTED'},
                OM_HAS_ARTIFACT_TYPE: {type: 'OM_HAS_ARTIFACT_TYPE', orientation: 'UNDIRECTED'},
                OM_REFERENCES_ARTIFACT: {type: 'OM_REFERENCES_ARTIFACT', orientation: 'UNDIRECTED'},
                OM_HAS_EVIDENCE: {type: 'OM_HAS_EVIDENCE', orientation: 'UNDIRECTED'},
                OM_WRITTEN_VIA: {type: 'OM_WRITTEN_VIA', orientation: 'UNDIRECTED'},
                OM_SIMILAR: {
                    type: 'OM_SIMILAR',
                    orientation: 'UNDIRECTED',
                    properties: {weight: {property: 'score', defaultValue: 0.5}}
                },
                OM_CO_MENTIONED: {
                    type: 'OM_CO_MENTIONED',
                    orientation: 'UNDIRECTED',
                    properties: {weight: {property: 'count', defaultValue: 1.0}}
                }
            }
        )
        YIELD graphName, nodeCount, relationshipCount
        RETURN graphName, nodeCount, relationshipCount
        """

    @staticmethod
    def drop_graph_query() -> str:
        """Drop an in-memory GDS graph."""
        return """
        CALL gds.graph.drop($graphName, false)
        YIELD graphName
        RETURN graphName
        """

    @staticmethod
    def graph_exists_query() -> str:
        """Check if a GDS graph exists."""
        return """
        CALL gds.graph.exists($graphName)
        YIELD exists
        RETURN exists
        """

    # =========================================================================
    # PageRank Queries
    # =========================================================================

    @staticmethod
    def pagerank_stream_query() -> str:
        """Run PageRank and stream results."""
        return """
        CALL gds.pageRank.stream($graphName, {
            dampingFactor: $dampingFactor,
            maxIterations: $maxIterations,
            tolerance: $tolerance
        })
        YIELD nodeId, score
        WITH gds.util.asNode(nodeId) AS node, score
        RETURN node.name AS name,
               labels(node)[0] AS label,
               score AS pageRankScore
        ORDER BY score DESC
        LIMIT $limit
        """

    @staticmethod
    def pagerank_write_query() -> str:
        """Run PageRank and write results to node properties."""
        return """
        CALL gds.pageRank.write($graphName, {
            dampingFactor: $dampingFactor,
            maxIterations: $maxIterations,
            tolerance: $tolerance,
            writeProperty: 'pageRank'
        })
        YIELD nodePropertiesWritten, ranIterations, didConverge
        RETURN nodePropertiesWritten, ranIterations, didConverge
        """

    # =========================================================================
    # Community Detection Queries
    # =========================================================================

    @staticmethod
    def louvain_stream_query() -> str:
        """Run Louvain community detection and stream results."""
        return """
        CALL gds.louvain.stream($graphName, {
            maxIterations: $maxIterations,
            includeIntermediateCommunities: $includeIntermediate
        })
        YIELD nodeId, communityId, intermediateCommunityIds
        WITH gds.util.asNode(nodeId) AS node, communityId, intermediateCommunityIds
        RETURN node.id AS id,
               node.name AS name,
               labels(node)[0] AS label,
               communityId,
               intermediateCommunityIds
        ORDER BY communityId, name
        """

    @staticmethod
    def louvain_stats_query() -> str:
        """Get Louvain community statistics."""
        return """
        CALL gds.louvain.stats($graphName, {
            maxIterations: $maxIterations
        })
        YIELD communityCount, modularity, modularities
        RETURN communityCount, modularity, modularities
        """

    @staticmethod
    def wcc_stream_query() -> str:
        """Run Weakly Connected Components detection."""
        return """
        CALL gds.wcc.stream($graphName)
        YIELD nodeId, componentId
        WITH gds.util.asNode(nodeId) AS node, componentId
        RETURN node.id AS id,
               node.name AS name,
               labels(node)[0] AS label,
               componentId
        ORDER BY componentId, name
        """

    # =========================================================================
    # Node Similarity Queries
    # =========================================================================

    @staticmethod
    def node_similarity_stream_query() -> str:
        """Find similar nodes based on shared neighbors."""
        return """
        CALL gds.nodeSimilarity.stream($graphName, {
            topK: $topK,
            similarityCutoff: $minSimilarity
        })
        YIELD node1, node2, similarity
        WITH gds.util.asNode(node1) AS n1, gds.util.asNode(node2) AS n2, similarity
        RETURN n1.name AS entity1,
               n2.name AS entity2,
               labels(n1)[0] AS label1,
               labels(n2)[0] AS label2,
               similarity
        ORDER BY similarity DESC
        LIMIT $limit
        """

    @staticmethod
    def knn_stream_query() -> str:
        """K-Nearest Neighbors based on node properties."""
        return """
        CALL gds.knn.stream($graphName, {
            topK: $topK,
            nodeProperties: $nodeProperties,
            similarityCutoff: $minSimilarity
        })
        YIELD node1, node2, similarity
        WITH gds.util.asNode(node1) AS n1, gds.util.asNode(node2) AS n2, similarity
        RETURN n1.name AS name1,
               n2.name AS name2,
               similarity
        ORDER BY similarity DESC
        LIMIT $limit
        """

    # =========================================================================
    # Centrality Queries
    # =========================================================================

    @staticmethod
    def betweenness_stream_query() -> str:
        """Run betweenness centrality."""
        return """
        CALL gds.betweenness.stream($graphName)
        YIELD nodeId, score
        WITH gds.util.asNode(nodeId) AS node, score
        RETURN node.name AS name,
               labels(node)[0] AS label,
               score AS betweenness
        ORDER BY score DESC
        LIMIT $limit
        """

    @staticmethod
    def degree_stream_query() -> str:
        """Run degree centrality."""
        return """
        CALL gds.degree.stream($graphName)
        YIELD nodeId, score
        WITH gds.util.asNode(nodeId) AS node, score
        RETURN node.name AS name,
               labels(node)[0] AS label,
               score AS degree
        ORDER BY score DESC
        LIMIT $limit
        """

    # =========================================================================
    # FastRP Embeddings Queries
    # =========================================================================

    @staticmethod
    def fastrp_stream_query() -> str:
        """Generate FastRP embeddings for nodes."""
        return """
        CALL gds.fastRP.stream($graphName, {
            embeddingDimension: $embeddingDimension,
            iterationWeights: $iterationWeights
        })
        YIELD nodeId, embedding
        WITH gds.util.asNode(nodeId) AS node, embedding
        RETURN node.id AS id,
               node.name AS name,
               labels(node)[0] AS label,
               embedding
        LIMIT $limit
        """

    @staticmethod
    def fastrp_write_query() -> str:
        """Generate FastRP embeddings and write to node properties."""
        return """
        CALL gds.fastRP.write($graphName, {
            embeddingDimension: $embeddingDimension,
            iterationWeights: $iterationWeights,
            writeProperty: 'graphEmbedding'
        })
        YIELD nodePropertiesWritten
        RETURN nodePropertiesWritten
        """


class GDSOperations:
    """
    High-level operations using Neo4j Graph Data Science library.

    Provides methods for running graph algorithms on OpenMemory data:
    - PageRank for entity influence
    - Community detection for memory clustering
    - Node similarity for related entities
    - Graph embeddings for ML features
    """

    def __init__(self, session_factory, config: GDSConfig = None):
        """
        Initialize GDS operations.

        Args:
            session_factory: Callable that returns a Neo4j session context manager
            config: Optional GDS configuration
        """
        self.session_factory = session_factory
        self.config = config or GDSConfig()
        self._gds_available = None

    def is_gds_available(self) -> bool:
        """
        Check if GDS library is available.

        Returns:
            True if GDS is installed and accessible
        """
        if self._gds_available is not None:
            return self._gds_available

        try:
            with self.session_factory() as session:
                result = session.run("RETURN gds.version() AS version")
                for record in result:
                    version = record["version"]
                    logger.info(f"Neo4j GDS version: {version}")
                    self._gds_available = True
                    return True
        except Exception as e:
            logger.warning(f"GDS not available: {e}")
            self._gds_available = False

        return False

    def _ensure_graph_projected(
        self,
        graph_name: str,
        projection_type: str = "entity",
    ) -> bool:
        """
        Ensure an in-memory graph is projected.

        Args:
            graph_name: Name for the projected graph
            projection_type: One of "entity", "memory", or "full"

        Returns:
            True if graph is ready
        """
        try:
            with self.session_factory() as session:
                # Check if graph exists
                result = session.run(
                    GDSCypherBuilder.graph_exists_query(),
                    {"graphName": graph_name}
                )
                for record in result:
                    if record["exists"]:
                        return True

                # Project the graph
                if projection_type == "entity":
                    query = GDSCypherBuilder.project_entity_graph_query()
                elif projection_type == "memory":
                    query = GDSCypherBuilder.project_memory_graph_query()
                else:
                    query = GDSCypherBuilder.project_full_graph_query()

                result = session.run(query, {"graphName": graph_name})
                for record in result:
                    logger.info(
                        f"Projected graph '{graph_name}': "
                        f"{record['nodeCount']} nodes, "
                        f"{record['relationshipCount']} relationships"
                    )
                    return True

                return False

        except Exception as e:
            logger.error(f"Failed to project graph '{graph_name}': {e}")
            return False

    def _drop_graph(self, graph_name: str) -> bool:
        """Drop an in-memory graph."""
        try:
            with self.session_factory() as session:
                session.run(
                    GDSCypherBuilder.drop_graph_query(),
                    {"graphName": graph_name}
                )
                return True
        except Exception as e:
            logger.debug(f"Could not drop graph '{graph_name}': {e}")
            return False

    # =========================================================================
    # PageRank Operations
    # =========================================================================

    def entity_pagerank(
        self,
        user_id: str,
        limit: int = 50,
        write_to_nodes: bool = False,
    ) -> List[Dict[str, Any]]:
        """
        Run PageRank on entity co-mention network.

        Identifies the most influential entities based on how they're
        connected through shared memories.

        Args:
            user_id: String user ID
            limit: Maximum results to return
            write_to_nodes: Whether to write pageRank property to nodes

        Returns:
            List of entities with pageRankScore
        """
        if not self.is_gds_available():
            logger.warning("GDS not available for PageRank")
            return []

        graph_name = f"om_entity_graph_{user_id}"

        try:
            if not self._ensure_graph_projected(graph_name, "entity"):
                return []

            with self.session_factory() as session:
                if write_to_nodes:
                    session.run(
                        GDSCypherBuilder.pagerank_write_query(),
                        {
                            "graphName": graph_name,
                            "dampingFactor": self.config.pagerank_damping_factor,
                            "maxIterations": self.config.pagerank_max_iterations,
                            "tolerance": self.config.pagerank_tolerance,
                        }
                    )

                result = session.run(
                    GDSCypherBuilder.pagerank_stream_query(),
                    {
                        "graphName": graph_name,
                        "dampingFactor": self.config.pagerank_damping_factor,
                        "maxIterations": self.config.pagerank_max_iterations,
                        "tolerance": self.config.pagerank_tolerance,
                        "limit": max(1, min(int(limit or 50), 500)),
                    }
                )

                rankings = []
                for record in result:
                    rankings.append({
                        "name": record["name"],
                        "label": record["label"],
                        "pageRankScore": record["pageRankScore"],
                    })

                return rankings

        except Exception as e:
            logger.error(f"Failed to run entity PageRank: {e}")
            return []

        finally:
            self._drop_graph(graph_name)

    # =========================================================================
    # Community Detection Operations
    # =========================================================================

    def detect_entity_communities(
        self,
        user_id: str,
    ) -> Dict[str, Any]:
        """
        Detect communities in the entity co-mention network using Louvain.

        Groups entities that frequently appear together in memories.

        Args:
            user_id: String user ID

        Returns:
            Dict with communities list and statistics
        """
        if not self.is_gds_available():
            logger.warning("GDS not available for community detection")
            return {"communities": [], "stats": {}}

        graph_name = f"om_entity_graph_{user_id}"

        try:
            if not self._ensure_graph_projected(graph_name, "entity"):
                return {"communities": [], "stats": {}}

            with self.session_factory() as session:
                # Get community statistics
                stats_result = session.run(
                    GDSCypherBuilder.louvain_stats_query(),
                    {
                        "graphName": graph_name,
                        "maxIterations": self.config.louvain_max_iterations,
                    }
                )

                stats = {}
                for record in stats_result:
                    stats = {
                        "communityCount": record["communityCount"],
                        "modularity": record["modularity"],
                        "modularities": record["modularities"],
                    }

                # Get community assignments
                result = session.run(
                    GDSCypherBuilder.louvain_stream_query(),
                    {
                        "graphName": graph_name,
                        "maxIterations": self.config.louvain_max_iterations,
                        "includeIntermediate": self.config.louvain_include_intermediate,
                    }
                )

                # Group by community
                communities_map = {}
                for record in result:
                    community_id = record["communityId"]
                    if community_id not in communities_map:
                        communities_map[community_id] = []

                    communities_map[community_id].append({
                        "id": record["id"],
                        "name": record["name"],
                        "label": record["label"],
                    })

                communities = [
                    {
                        "communityId": cid,
                        "members": members,
                        "size": len(members),
                    }
                    for cid, members in communities_map.items()
                ]

                # Sort by size descending
                communities.sort(key=lambda c: c["size"], reverse=True)

                return {
                    "communities": communities,
                    "stats": stats,
                }

        except Exception as e:
            logger.error(f"Failed to detect entity communities: {e}")
            return {"communities": [], "stats": {}}

        finally:
            self._drop_graph(graph_name)

    def detect_memory_communities(
        self,
        user_id: str,
    ) -> Dict[str, Any]:
        """
        Detect communities in the memory similarity network.

        Groups memories that are semantically similar to each other.

        Args:
            user_id: String user ID

        Returns:
            Dict with communities list and statistics
        """
        if not self.is_gds_available():
            return {"communities": [], "stats": {}}

        graph_name = f"om_memory_graph_{user_id}"

        try:
            if not self._ensure_graph_projected(graph_name, "memory"):
                return {"communities": [], "stats": {}}

            with self.session_factory() as session:
                # Use WCC for memory clusters (Louvain might be too fine-grained)
                result = session.run(
                    GDSCypherBuilder.wcc_stream_query(),
                    {"graphName": graph_name}
                )

                communities_map = {}
                for record in result:
                    component_id = record["componentId"]
                    if component_id not in communities_map:
                        communities_map[component_id] = []

                    communities_map[component_id].append({
                        "id": record["id"],
                        "name": record["name"],
                        "label": record["label"],
                    })

                communities = [
                    {
                        "communityId": cid,
                        "members": members,
                        "size": len(members),
                    }
                    for cid, members in communities_map.items()
                ]

                communities.sort(key=lambda c: c["size"], reverse=True)

                return {
                    "communities": communities,
                    "stats": {"communityCount": len(communities)},
                }

        except Exception as e:
            logger.error(f"Failed to detect memory communities: {e}")
            return {"communities": [], "stats": {}}

        finally:
            self._drop_graph(graph_name)

    # =========================================================================
    # Node Similarity Operations
    # =========================================================================

    def find_similar_entities(
        self,
        user_id: str,
        entity_name: Optional[str] = None,
        limit: int = 50,
    ) -> List[Dict[str, Any]]:
        """
        Find similar entities based on shared memory connections.

        Uses Jaccard similarity on shared neighbors (memories).

        Args:
            user_id: String user ID
            entity_name: Optional specific entity to find similar to
            limit: Maximum results to return

        Returns:
            List of entity pairs with similarity scores
        """
        if not self.is_gds_available():
            return []

        graph_name = f"om_full_graph_{user_id}"

        try:
            if not self._ensure_graph_projected(graph_name, "full"):
                return []

            with self.session_factory() as session:
                result = session.run(
                    GDSCypherBuilder.node_similarity_stream_query(),
                    {
                        "graphName": graph_name,
                        "topK": self.config.similarity_top_k,
                        "minSimilarity": self.config.similarity_min_similarity,
                        "limit": max(1, min(int(limit or 50), 500)),
                    }
                )

                similarities = []
                for record in result:
                    # Filter to entity-entity pairs
                    if record["label1"] == "OM_Entity" and record["label2"] == "OM_Entity":
                        # If entity_name specified, filter to that entity
                        if entity_name:
                            if record["entity1"] != entity_name and record["entity2"] != entity_name:
                                continue

                        similarities.append({
                            "entity1": record["entity1"],
                            "entity2": record["entity2"],
                            "similarity": record["similarity"],
                        })

                return similarities

        except Exception as e:
            logger.error(f"Failed to find similar entities: {e}")
            return []

        finally:
            self._drop_graph(graph_name)

    # =========================================================================
    # Centrality Operations
    # =========================================================================

    def entity_betweenness(
        self,
        user_id: str,
        limit: int = 50,
    ) -> List[Dict[str, Any]]:
        """
        Calculate betweenness centrality for entities.

        Entities with high betweenness act as bridges between
        different clusters of memories.

        Args:
            user_id: String user ID
            limit: Maximum results to return

        Returns:
            List of entities with betweenness scores
        """
        if not self.is_gds_available():
            return []

        graph_name = f"om_entity_graph_{user_id}"

        try:
            if not self._ensure_graph_projected(graph_name, "entity"):
                return []

            with self.session_factory() as session:
                result = session.run(
                    GDSCypherBuilder.betweenness_stream_query(),
                    {
                        "graphName": graph_name,
                        "limit": max(1, min(int(limit or 50), 500)),
                    }
                )

                centralities = []
                for record in result:
                    centralities.append({
                        "name": record["name"],
                        "label": record["label"],
                        "betweenness": record["betweenness"],
                    })

                return centralities

        except Exception as e:
            logger.error(f"Failed to calculate betweenness centrality: {e}")
            return []

        finally:
            self._drop_graph(graph_name)

    # =========================================================================
    # Graph Embedding Operations
    # =========================================================================

    def generate_entity_embeddings(
        self,
        user_id: str,
        write_to_nodes: bool = False,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """
        Generate FastRP graph embeddings for entities.

        These embeddings capture the structural position of entities
        in the co-mention graph and can be used for ML tasks.

        Args:
            user_id: String user ID
            write_to_nodes: Whether to write embeddings to node properties
            limit: Maximum results to return

        Returns:
            List of entities with embeddings
        """
        if not self.is_gds_available():
            return []

        graph_name = f"om_entity_graph_{user_id}"

        try:
            if not self._ensure_graph_projected(graph_name, "entity"):
                return []

            with self.session_factory() as session:
                if write_to_nodes:
                    session.run(
                        GDSCypherBuilder.fastrp_write_query(),
                        {
                            "graphName": graph_name,
                            "embeddingDimension": self.config.fastrp_embedding_dimension,
                            "iterationWeights": self.config.fastrp_iteration_weights,
                        }
                    )

                result = session.run(
                    GDSCypherBuilder.fastrp_stream_query(),
                    {
                        "graphName": graph_name,
                        "embeddingDimension": self.config.fastrp_embedding_dimension,
                        "iterationWeights": self.config.fastrp_iteration_weights,
                        "limit": max(1, min(int(limit or 100), 1000)),
                    }
                )

                embeddings = []
                for record in result:
                    embeddings.append({
                        "id": record["id"],
                        "name": record["name"],
                        "label": record["label"],
                        "embedding": list(record["embedding"]),
                    })

                return embeddings

        except Exception as e:
            logger.error(f"Failed to generate entity embeddings: {e}")
            return []

        finally:
            self._drop_graph(graph_name)


# =============================================================================
# Module-level singleton management
# =============================================================================

_gds_operations = None
_gds_operations_initialized = False


def get_gds_operations() -> Optional[GDSOperations]:
    """
    Get the singleton GDSOperations instance.

    Returns:
        GDSOperations if Neo4j with GDS is configured, None otherwise
    """
    global _gds_operations, _gds_operations_initialized

    if _gds_operations_initialized:
        return _gds_operations

    _gds_operations_initialized = True

    try:
        from app.graph.neo4j_client import get_neo4j_session, is_neo4j_configured

        if not is_neo4j_configured():
            logger.debug("Neo4j not configured, GDS operations disabled")
            return None

        _gds_operations = GDSOperations(get_neo4j_session)

        if _gds_operations.is_gds_available():
            logger.info("GDS operations initialized")
        else:
            logger.info("GDS not available, advanced analytics disabled")

    except Exception as e:
        logger.warning(f"Failed to initialize GDS operations: {e}")
        _gds_operations = None

    return _gds_operations


def reset_gds_operations():
    """Reset the GDS operations instance (for testing or config changes)."""
    global _gds_operations, _gds_operations_initialized
    _gds_operations = None
    _gds_operations_initialized = False
