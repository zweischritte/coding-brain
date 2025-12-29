"""
Code Intelligence Toolkit Factory.

Provides lazy initialization of code tool dependencies with graceful degradation.
This factory centralizes the creation of all code-intel tool instances and their
dependencies, ensuring consistent error handling and service availability checks.

Usage:
    from app.code_toolkit import get_code_toolkit

    toolkit = get_code_toolkit()
    if toolkit.is_available("opensearch"):
        result = toolkit.search_tool.search(input_data)
"""

import logging
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any, Optional, Dict

logger = logging.getLogger(__name__)


@dataclass
class CodeToolkit:
    """Container for code intelligence dependencies and tools.

    This class holds references to all code-intel dependencies and tool instances.
    Use `is_available()` to check if a specific service is available before
    using its associated tools.

    Attributes:
        opensearch_client: OpenSearch client for code search
        neo4j_driver: Neo4j driver for graph operations
        trihybrid_retriever: Tri-hybrid retriever for combined search
        embedding_service: Embedding service for vector operations
        ast_parser: AST parser for code analysis

        search_tool: Search code hybrid tool instance
        explain_tool: Explain code tool instance
        callers_tool: Find callers tool instance
        callees_tool: Find callees tool instance
        impact_tool: Impact analysis tool instance
        adr_tool: ADR automation tool instance
        test_gen_tool: Test generation tool instance
    """

    # Dependencies
    opensearch_client: Optional[Any] = None
    neo4j_driver: Optional[Any] = None
    trihybrid_retriever: Optional[Any] = None
    embedding_service: Optional[Any] = None
    ast_parser: Optional[Any] = None

    # Tool instances
    search_tool: Optional[Any] = None
    explain_tool: Optional[Any] = None
    callers_tool: Optional[Any] = None
    callees_tool: Optional[Any] = None
    impact_tool: Optional[Any] = None
    adr_tool: Optional[Any] = None
    test_gen_tool: Optional[Any] = None

    # Service availability tracking
    _available_services: Dict[str, bool] = field(default_factory=dict)
    _initialization_errors: Dict[str, str] = field(default_factory=dict)

    def is_available(self, service: str) -> bool:
        """Check if a service/dependency is available.

        Args:
            service: Service name (opensearch, neo4j, embedding, etc.)

        Returns:
            True if the service is available and initialized
        """
        return self._available_services.get(service, False)

    def get_error(self, service: str) -> Optional[str]:
        """Get initialization error for a service.

        Args:
            service: Service name

        Returns:
            Error message if initialization failed, None otherwise
        """
        return self._initialization_errors.get(service)

    def get_available_services(self) -> Dict[str, bool]:
        """Get availability status of all services.

        Returns:
            Dict mapping service names to availability status
        """
        return dict(self._available_services)

    def get_missing_sources(self) -> list[str]:
        """Get list of unavailable services.

        Returns:
            List of service names that are not available
        """
        return [
            name for name, available in self._available_services.items()
            if not available
        ]


def _init_opensearch(toolkit: CodeToolkit) -> None:
    """Initialize OpenSearch client."""
    try:
        from retrieval.opensearch import create_opensearch_client

        toolkit.opensearch_client = create_opensearch_client(from_env=True)

        # Test connection
        health = toolkit.opensearch_client.health()
        if health:
            toolkit._available_services["opensearch"] = True
            logger.info("OpenSearch client initialized successfully")
        else:
            toolkit._available_services["opensearch"] = False
            toolkit._initialization_errors["opensearch"] = "Health check failed"
            logger.warning("OpenSearch health check failed")
    except ImportError as e:
        toolkit._available_services["opensearch"] = False
        toolkit._initialization_errors["opensearch"] = f"Import error: {e}"
        logger.warning(f"OpenSearch module not available: {e}")
    except Exception as e:
        toolkit._available_services["opensearch"] = False
        toolkit._initialization_errors["opensearch"] = str(e)
        logger.warning(f"OpenSearch initialization failed: {e}")


def _init_neo4j(toolkit: CodeToolkit) -> None:
    """Initialize Neo4j driver."""
    try:
        from app.graph.neo4j_client import (
            get_neo4j_driver,
            is_neo4j_configured,
            is_neo4j_healthy,
        )

        if not is_neo4j_configured():
            toolkit._available_services["neo4j"] = False
            toolkit._initialization_errors["neo4j"] = "Not configured"
            logger.info("Neo4j not configured - graph features disabled")
            return

        toolkit.neo4j_driver = get_neo4j_driver()

        if toolkit.neo4j_driver and is_neo4j_healthy():
            toolkit._available_services["neo4j"] = True
            logger.info("Neo4j driver initialized successfully")
        else:
            toolkit._available_services["neo4j"] = False
            toolkit._initialization_errors["neo4j"] = "Health check failed"
            logger.warning("Neo4j health check failed")
    except ImportError as e:
        toolkit._available_services["neo4j"] = False
        toolkit._initialization_errors["neo4j"] = f"Import error: {e}"
        logger.warning(f"Neo4j module not available: {e}")
    except Exception as e:
        toolkit._available_services["neo4j"] = False
        toolkit._initialization_errors["neo4j"] = str(e)
        logger.warning(f"Neo4j initialization failed: {e}")


def _init_embedding_service(toolkit: CodeToolkit) -> None:
    """Initialize embedding service."""
    try:
        from retrieval.embedding_pipeline import create_embedding_pipeline

        toolkit.embedding_service = create_embedding_pipeline(
            model_name="nomic-embed-text",
            dimension=768,
            provider="ollama",
        )
        toolkit._available_services["embedding"] = True
        logger.info("Embedding service initialized successfully")
    except ImportError as e:
        toolkit._available_services["embedding"] = False
        toolkit._initialization_errors["embedding"] = f"Import error: {e}"
        logger.warning(f"Embedding module not available: {e}")
    except Exception as e:
        toolkit._available_services["embedding"] = False
        toolkit._initialization_errors["embedding"] = str(e)
        logger.warning(f"Embedding service initialization failed: {e}")


def _init_trihybrid_retriever(toolkit: CodeToolkit) -> None:
    """Initialize tri-hybrid retriever."""
    if not toolkit.is_available("opensearch"):
        toolkit._available_services["trihybrid"] = False
        toolkit._initialization_errors["trihybrid"] = "OpenSearch not available"
        return

    try:
        from retrieval.trihybrid import create_trihybrid_retriever

        toolkit.trihybrid_retriever = create_trihybrid_retriever(
            opensearch_client=toolkit.opensearch_client,
            graph_driver=toolkit.neo4j_driver,  # Optional
        )
        toolkit._available_services["trihybrid"] = True
        logger.info("Tri-hybrid retriever initialized")
    except ImportError as e:
        toolkit._available_services["trihybrid"] = False
        toolkit._initialization_errors["trihybrid"] = f"Import error: {e}"
        logger.warning(f"Tri-hybrid module not available: {e}")
    except Exception as e:
        toolkit._available_services["trihybrid"] = False
        toolkit._initialization_errors["trihybrid"] = str(e)
        logger.warning(f"Tri-hybrid retriever initialization failed: {e}")


def _init_ast_parser(toolkit: CodeToolkit) -> None:
    """Initialize AST parser."""
    try:
        from indexing.ast_parser import create_parser

        toolkit.ast_parser = create_parser()
        toolkit._available_services["ast_parser"] = True
        logger.info("AST parser initialized")
    except ImportError as e:
        toolkit._available_services["ast_parser"] = False
        toolkit._initialization_errors["ast_parser"] = f"Import error: {e}"
        logger.warning(f"AST parser module not available: {e}")
    except Exception as e:
        toolkit._available_services["ast_parser"] = False
        toolkit._initialization_errors["ast_parser"] = str(e)
        logger.warning(f"AST parser initialization failed: {e}")


def _init_search_tool(toolkit: CodeToolkit) -> None:
    """Initialize search code hybrid tool."""
    if not toolkit.is_available("trihybrid") and not toolkit.is_available("opensearch"):
        toolkit._initialization_errors["search_tool"] = "No search backend available"
        logger.warning("Search tool not initialized - no backend available")
        return

    try:
        from tools.search_code_hybrid import create_search_code_hybrid_tool

        toolkit.search_tool = create_search_code_hybrid_tool(
            retriever=toolkit.trihybrid_retriever,
            embedding_service=toolkit.embedding_service,
        )
        logger.info("Search tool initialized")
    except ImportError as e:
        toolkit._initialization_errors["search_tool"] = f"Import error: {e}"
        logger.warning(f"Search tool module not available: {e}")
    except Exception as e:
        toolkit._initialization_errors["search_tool"] = str(e)
        logger.warning(f"Search tool initialization failed: {e}")


def _init_explain_tool(toolkit: CodeToolkit) -> None:
    """Initialize explain code tool."""
    if not toolkit.is_available("neo4j"):
        toolkit._initialization_errors["explain_tool"] = "Neo4j not available"
        return

    try:
        from tools.explain_code import create_explain_code_tool

        toolkit.explain_tool = create_explain_code_tool(
            graph_driver=toolkit.neo4j_driver,
        )
        logger.info("Explain tool initialized")
    except ImportError as e:
        toolkit._initialization_errors["explain_tool"] = f"Import error: {e}"
        logger.warning(f"Explain tool module not available: {e}")
    except Exception as e:
        toolkit._initialization_errors["explain_tool"] = str(e)
        logger.warning(f"Explain tool initialization failed: {e}")


def _init_call_graph_tools(toolkit: CodeToolkit) -> None:
    """Initialize find callers and find callees tools."""
    if not toolkit.is_available("neo4j"):
        toolkit._initialization_errors["callers_tool"] = "Neo4j not available"
        toolkit._initialization_errors["callees_tool"] = "Neo4j not available"
        return

    try:
        from tools.call_graph import (
            create_find_callers_tool,
            create_find_callees_tool,
        )

        toolkit.callers_tool = create_find_callers_tool(
            graph_driver=toolkit.neo4j_driver,
        )
        toolkit.callees_tool = create_find_callees_tool(
            graph_driver=toolkit.neo4j_driver,
        )
        logger.info("Call graph tools initialized")
    except ImportError as e:
        toolkit._initialization_errors["callers_tool"] = f"Import error: {e}"
        toolkit._initialization_errors["callees_tool"] = f"Import error: {e}"
        logger.warning(f"Call graph tools module not available: {e}")
    except Exception as e:
        toolkit._initialization_errors["callers_tool"] = str(e)
        toolkit._initialization_errors["callees_tool"] = str(e)
        logger.warning(f"Call graph tools initialization failed: {e}")


def _init_impact_tool(toolkit: CodeToolkit) -> None:
    """Initialize impact analysis tool."""
    if not toolkit.is_available("neo4j"):
        toolkit._initialization_errors["impact_tool"] = "Neo4j not available"
        return

    try:
        from tools.impact_analysis import create_impact_analysis_tool

        toolkit.impact_tool = create_impact_analysis_tool(
            graph_driver=toolkit.neo4j_driver,
        )
        logger.info("Impact analysis tool initialized")
    except ImportError as e:
        toolkit._initialization_errors["impact_tool"] = f"Import error: {e}"
        logger.warning(f"Impact analysis tool module not available: {e}")
    except Exception as e:
        toolkit._initialization_errors["impact_tool"] = str(e)
        logger.warning(f"Impact analysis tool initialization failed: {e}")


def _init_adr_tool(toolkit: CodeToolkit) -> None:
    """Initialize ADR automation tool."""
    try:
        from tools.adr_automation import create_adr_automation_tool

        toolkit.adr_tool = create_adr_automation_tool(
            graph_driver=toolkit.neo4j_driver,  # Optional
        )
        logger.info("ADR tool initialized")
    except ImportError as e:
        toolkit._initialization_errors["adr_tool"] = f"Import error: {e}"
        logger.warning(f"ADR tool module not available: {e}")
    except Exception as e:
        toolkit._initialization_errors["adr_tool"] = str(e)
        logger.warning(f"ADR tool initialization failed: {e}")


def _init_test_gen_tool(toolkit: CodeToolkit) -> None:
    """Initialize test generation tool."""
    try:
        from tools.test_generation import create_test_generation_tool

        toolkit.test_gen_tool = create_test_generation_tool(
            graph_driver=toolkit.neo4j_driver,  # Optional
            parser=toolkit.ast_parser,  # Optional
        )
        logger.info("Test generation tool initialized")
    except ImportError as e:
        toolkit._initialization_errors["test_gen_tool"] = f"Import error: {e}"
        logger.warning(f"Test generation tool module not available: {e}")
    except Exception as e:
        toolkit._initialization_errors["test_gen_tool"] = str(e)
        logger.warning(f"Test generation tool initialization failed: {e}")


@lru_cache(maxsize=1)
def get_code_toolkit() -> CodeToolkit:
    """Get or create the code intelligence toolkit singleton.

    This function lazily initializes all code-intel dependencies and tools.
    Missing dependencies are logged but don't prevent initialization.
    Use `toolkit.is_available(service)` to check if a specific service
    is available before using its tools.

    Returns:
        CodeToolkit instance with available tools and dependencies.

    Example:
        toolkit = get_code_toolkit()

        if toolkit.is_available("opensearch"):
            result = toolkit.search_tool.search(input_data)
        else:
            # Handle degraded mode
            missing = toolkit.get_missing_sources()
            logger.warning(f"Missing services: {missing}")
    """
    logger.info("Initializing Code Intelligence Toolkit...")
    toolkit = CodeToolkit()

    # Initialize dependencies in order
    _init_opensearch(toolkit)
    _init_neo4j(toolkit)
    _init_embedding_service(toolkit)
    _init_trihybrid_retriever(toolkit)
    _init_ast_parser(toolkit)

    # Initialize tools
    _init_search_tool(toolkit)
    _init_explain_tool(toolkit)
    _init_call_graph_tools(toolkit)
    _init_impact_tool(toolkit)
    _init_adr_tool(toolkit)
    _init_test_gen_tool(toolkit)

    # Log summary
    available = [k for k, v in toolkit._available_services.items() if v]
    unavailable = [k for k, v in toolkit._available_services.items() if not v]

    logger.info(f"Toolkit initialized. Available: {available}")
    if unavailable:
        logger.warning(f"Unavailable services: {unavailable}")

    return toolkit


def reset_toolkit() -> None:
    """Reset the toolkit singleton (mainly for testing).

    This clears the cached toolkit instance, allowing a fresh
    initialization on the next `get_code_toolkit()` call.
    """
    get_code_toolkit.cache_clear()
