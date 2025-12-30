"""Test Generation Tool (FR-008).

This module provides the test generation tool:
- TestGenerationConfig: Configuration for test generation
- SymbolAnalyzer: Analyzes symbols for test generation
- TestPattern: Represents a test pattern extracted from codebase
- PatternMatcher: Matches existing test patterns in the project
- CoverageAnalyzer: Analyzes existing test coverage
- TestTemplate: Template for generating test code
- TestCase: Generated test case structure
- TestSuite: Collection of generated test cases
- TestGenerator: Generates test code from analysis
- TestGenerationTool: Main tool entry point

Features:
- Generate tests for functions and classes
- Apply team test patterns from existing tests
- Generate tests for uncovered code paths
- Support pytest and unittest frameworks
- Generate fixtures and mocks
- Include edge cases and error handling tests
"""

from __future__ import annotations

import logging
import os
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from openmemory.api.indexing.fallback_symbols import extract_python_symbols

logger = logging.getLogger(__name__)


def _find_repo_root(start: Path) -> Optional[Path]:
    """Find a repo root by looking for pyproject.toml."""
    candidate = start
    if candidate.is_file():
        candidate = candidate.parent

    for parent in [candidate] + list(candidate.parents):
        if (parent / "pyproject.toml").exists():
            return parent
    return None


def _suffix_after_repo(path: Path, repo_name: str) -> Optional[Path]:
    """Return path suffix after a repo name marker."""
    marker = f"/{repo_name}/"
    path_str = path.as_posix()
    if marker in path_str:
        return Path(path_str.split(marker, 1)[1])
    return None


def _resolve_source_path(file_path: str) -> Path:
    """Resolve a source path across host/container boundaries."""
    path = Path(file_path)
    if path.exists():
        return path

    candidates: list[Path] = []
    candidate_roots: list[Path] = []

    repo_root = _find_repo_root(Path(__file__).resolve())
    if repo_root:
        candidate_roots.append(repo_root)

    env_root = os.environ.get("OPENMEMORY_REPO_ROOT") or os.environ.get(
        "OPENMEMORY_WORKSPACE_ROOT"
    )
    if env_root:
        candidate_roots.append(Path(env_root))

    usr_src = Path("/usr/src")
    if usr_src.exists() and usr_src.is_dir():
        try:
            for child in usr_src.iterdir():
                if child.is_dir() and (child / "pyproject.toml").exists():
                    candidate_roots.append(child)
        except OSError:
            pass

    for root in candidate_roots:
        if path.is_absolute():
            suffix = _suffix_after_repo(path, root.name)
            if suffix:
                candidates.append(root / suffix)
        else:
            candidates.append(root / path)

    for candidate in candidates:
        if candidate.exists():
            return candidate

    return path


# =============================================================================
# Exceptions
# =============================================================================


class TestGenerationError(Exception):
    """Base exception for test generation errors."""

    pass


class SymbolNotFoundError(TestGenerationError):
    """Raised when a symbol cannot be found."""

    pass


class PatternExtractionError(TestGenerationError):
    """Raised when pattern extraction fails."""

    pass


# =============================================================================
# Configuration
# =============================================================================


@dataclass
class TestGenerationConfig:
    """Configuration for test generation tool.

    Args:
        test_framework: Test framework to use (pytest, unittest)
        include_fixtures: Include fixture generation
        include_mocks: Include mock generation for dependencies
        include_edge_cases: Generate edge case tests
        include_error_cases: Generate error handling tests
        max_tests_per_symbol: Maximum tests to generate per symbol
        use_team_patterns: Apply patterns from existing team tests
    """

    test_framework: str = "pytest"
    include_fixtures: bool = True
    include_mocks: bool = True
    include_edge_cases: bool = True
    include_error_cases: bool = True
    max_tests_per_symbol: int = 10
    use_team_patterns: bool = True


# =============================================================================
# Test Pattern
# =============================================================================


@dataclass
class TestPattern:
    """Represents a test pattern extracted from codebase.

    Args:
        name: Pattern name (e.g., "happy_path", "error_handling")
        description: Human-readable description
        template: Code template for the pattern
        category: Pattern category (fixture, assertion, error, etc.)
    """

    name: str
    description: str
    template: str
    category: str = "general"


# =============================================================================
# Test Case
# =============================================================================


@dataclass
class TestCase:
    """Generated test case structure.

    Args:
        name: Test function name
        description: Test description
        code: Generated test code
        symbol_id: Symbol being tested
        category: Test category (happy_path, edge_case, error, etc.)
    """

    name: str
    description: str
    code: str
    symbol_id: str
    category: str = "general"


# =============================================================================
# Test Suite
# =============================================================================


@dataclass
class TestSuite:
    """Collection of generated test cases.

    Args:
        symbol_id: Symbol being tested
        symbol_name: Name of the symbol
        test_cases: List of generated test cases
        imports: Required imports
        fixtures: Generated fixtures
    """

    symbol_id: str
    symbol_name: str
    test_cases: list[TestCase] = field(default_factory=list)
    imports: list[str] = field(default_factory=list)
    fixtures: list[str] = field(default_factory=list)

    def render(self) -> str:
        """Render the test suite as a complete test file.

        Returns:
            Complete test file content
        """
        lines = []

        # File header
        lines.append(f'"""Tests for {self.symbol_name}."""')
        lines.append("")

        # Imports
        if self.imports:
            for imp in self.imports:
                lines.append(imp)
            lines.append("")

        # Fixtures
        if self.fixtures:
            for fixture in self.fixtures:
                lines.append(fixture)
                lines.append("")

        # Test class header
        class_name = f"Test{self._to_pascal_case(self.symbol_name)}"
        lines.append(f"class {class_name}:")
        lines.append(f'    """Tests for {self.symbol_name}."""')
        lines.append("")

        # Test methods
        for test_case in self.test_cases:
            # Indent the test code
            indented_code = self._indent(test_case.code, 4)
            lines.append(indented_code)
            lines.append("")

        return "\n".join(lines)

    def _to_pascal_case(self, name: str) -> str:
        """Convert a symbol name into a safe PascalCase identifier."""
        sanitized = re.sub(r"[^A-Za-z0-9_]", "_", name or "")
        sanitized = sanitized.strip("_")
        if not sanitized:
            sanitized = "Symbol"
        if sanitized[0].isdigit():
            sanitized = f"Symbol_{sanitized}"
        parts = [part for part in sanitized.split("_") if part]
        return "".join(part.capitalize() for part in parts)

    def _indent(self, code: str, spaces: int) -> str:
        """Indent code block."""
        indent = " " * spaces
        return "\n".join(
            indent + line if line.strip() else line for line in code.split("\n")
        )


# =============================================================================
# Symbol Analyzer
# =============================================================================


class SymbolAnalyzer:
    """Analyzes symbols for test generation."""

    def __init__(self, graph_driver: Any):
        """Initialize with graph driver.

        Args:
            graph_driver: Neo4j driver for CODE_* graph
        """
        self._driver = graph_driver

    def analyze(self, symbol: dict[str, Any]) -> dict[str, Any]:
        """Analyze a symbol for test generation.

        Args:
            symbol: Symbol dictionary

        Returns:
            Analysis result with parameters, return type, dependencies
        """
        signature = symbol.get("signature") or ""
        analysis = {
            "symbol_id": symbol.get("symbol_id", ""),
            "name": symbol.get("name", ""),
            "kind": symbol.get("kind", "function"),
            "signature": signature,
            "docstring": symbol.get("docstring", ""),
            "parameters": [],
            "return_type": None,
            "is_async": False,
            "dependencies": [],
            "has_methods": False,
        }

        # Extract parameters
        params = symbol.get("parameters", [])
        if params:
            analysis["parameters"] = params
        else:
            # Try to parse from signature
            analysis["parameters"] = self._parse_parameters(signature)

        # Extract return type
        analysis["return_type"] = symbol.get(
            "return_type", self._parse_return_type(signature)
        )

        # Check if async
        analysis["is_async"] = symbol.get("is_async", False)
        if not analysis["is_async"]:
            analysis["is_async"] = "async def" in signature

        # Check for class methods
        if symbol.get("kind") == "class":
            analysis["has_methods"] = bool(symbol.get("methods", []))
            analysis["methods"] = symbol.get("methods", [])

        # Get dependencies from call graph
        analysis["dependencies"] = self._get_dependencies(
            symbol.get("symbol_id", "")
        )

        return analysis

    def _parse_parameters(self, signature: str) -> list[dict[str, Any]]:
        """Parse parameters from signature string."""
        params = []
        if not signature:
            return params

        # Match parameters in signature
        match = re.search(r"\((.*?)\)", signature)
        if not match:
            return params

        param_str = match.group(1)
        if not param_str or param_str == "self":
            return params

        # Split parameters
        for param in param_str.split(","):
            param = param.strip()
            if not param or param == "self":
                continue

            # Parse name, type, default
            name = param
            param_type = None
            default = None

            if ":" in param:
                name_part, type_part = param.split(":", 1)
                name = name_part.strip()
                if "=" in type_part:
                    type_part, default_part = type_part.split("=", 1)
                    default = default_part.strip()
                param_type = type_part.strip()
            elif "=" in param:
                name, default = param.split("=", 1)
                name = name.strip()
                default = default.strip()

            params.append(
                {"name": name, "type": param_type, "default": default}
            )

        return params

    def _parse_return_type(self, signature: str) -> Optional[str]:
        """Parse return type from signature string."""
        if not signature:
            return None
        match = re.search(r"->\s*(.+?):", signature)
        if match:
            return match.group(1).strip()
        return None

    def _get_dependencies(self, symbol_id: str) -> list[str]:
        """Get symbol dependencies from call graph."""
        dependencies = []

        try:
            edges = self._driver.get_outgoing_edges(symbol_id)
            for edge in edges:
                edge_type = (
                    edge.edge_type.value
                    if hasattr(edge.edge_type, "value")
                    else str(edge.edge_type)
                )
                if edge_type == "CALLS":
                    dependencies.append(edge.target_id)
        except Exception as e:
            logger.warning(f"Error getting dependencies: {e}")

        return dependencies


# =============================================================================
# Pattern Matcher
# =============================================================================


class PatternMatcher:
    """Extracts and matches test patterns from existing tests."""

    def extract_patterns(self, test_code: str) -> list[TestPattern]:
        """Extract test patterns from existing test code.

        Args:
            test_code: Existing test code to analyze

        Returns:
            List of extracted patterns
        """
        patterns = []

        # Detect fixture patterns
        if "@pytest.fixture" in test_code:
            patterns.append(
                TestPattern(
                    name="pytest_fixture",
                    description="Pytest fixture pattern",
                    template='@pytest.fixture\ndef {name}():\n    """Create {description}."""\n    return {value}',
                    category="fixture",
                )
            )

        # Detect class-based test patterns
        if re.search(r"class Test\w+:", test_code):
            patterns.append(
                TestPattern(
                    name="class_test",
                    description="Class-based test organization",
                    template='class Test{ClassName}:\n    """Tests for {class_name}."""',
                    category="structure",
                )
            )

        # Detect error handling patterns
        if "pytest.raises" in test_code:
            patterns.append(
                TestPattern(
                    name="error_raises",
                    description="pytest.raises error testing",
                    template="with pytest.raises({exception_type}):\n        {code}",
                    category="error",
                )
            )

        # Detect assertion patterns
        if "assert " in test_code:
            patterns.append(
                TestPattern(
                    name="simple_assert",
                    description="Simple assertion pattern",
                    template="assert {expression}",
                    category="assertion",
                )
            )

        # Detect mock patterns
        if "patch" in test_code or "Mock" in test_code:
            patterns.append(
                TestPattern(
                    name="mock_patch",
                    description="Mock/patch pattern",
                    template='@patch("{target}")\ndef test_{name}(self, mock_{var}):\n    mock_{var}.return_value = {value}',
                    category="mock",
                )
            )

        # Detect empty list edge case
        if "[]" in test_code and "empty" in test_code.lower():
            patterns.append(
                TestPattern(
                    name="empty_input",
                    description="Empty input edge case",
                    template="result = {func}([])\nassert result == []",
                    category="edge_case",
                )
            )

        return patterns


# =============================================================================
# Coverage Analyzer
# =============================================================================


class CoverageAnalyzer:
    """Analyzes existing test coverage."""

    def __init__(self, retriever: Any):
        """Initialize with retriever.

        Args:
            retriever: Retriever for finding existing tests
        """
        self._retriever = retriever

    def find_untested(self, symbols: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Find symbols without tests.

        Args:
            symbols: List of symbols to check

        Returns:
            List of untested symbols
        """
        untested = []

        for symbol in symbols:
            name = symbol.get("name", "")

            try:
                # Search for existing tests
                try:
                    from retrieval.trihybrid import TriHybridQuery
                except ImportError:
                    from openmemory.api.retrieval.trihybrid import TriHybridQuery

                query = TriHybridQuery(
                    query_text=f"test_{name}",
                    size=5,
                )
                result = self._retriever.retrieve(query, index_name="tests")

                if not result.hits:
                    untested.append(symbol)
            except Exception as e:
                logger.warning(f"Error checking coverage for {name}: {e}")
                untested.append(symbol)

        return untested

    def find_coverage_gaps(self, symbol: dict[str, Any]) -> list[str]:
        """Find coverage gaps in a tested symbol.

        Args:
            symbol: Symbol to analyze

        Returns:
            List of uncovered scenarios
        """
        gaps = []

        params = symbol.get("parameters", [])

        # Check for missing edge case tests
        for param in params:
            param_type = param.get("type", "")
            if "list" in param_type.lower():
                gaps.append(f"Empty list for {param.get('name')}")
            if "str" in param_type.lower():
                gaps.append(f"Empty string for {param.get('name')}")
            if "int" in param_type.lower() or "float" in param_type.lower():
                gaps.append(f"Zero value for {param.get('name')}")
                gaps.append(f"Negative value for {param.get('name')}")
            if "Optional" in param_type or param.get("default") is not None:
                gaps.append(f"None value for {param.get('name')}")

        return gaps

    def suggest_scenarios(self, symbol: dict[str, Any]) -> list[dict[str, Any]]:
        """Suggest test scenarios for a symbol.

        Args:
            symbol: Symbol to generate scenarios for

        Returns:
            List of test scenarios
        """
        scenarios = []

        name = symbol.get("name", "")
        params = symbol.get("parameters", [])

        # Happy path
        scenarios.append(
            {
                "name": f"test_{name}_success",
                "description": f"Test {name} with valid inputs",
                "category": "happy_path",
            }
        )

        # Edge cases for parameters
        for param in params:
            param_name = param.get("name", "")
            param_type = param.get("type", "")

            if "list" in param_type.lower():
                scenarios.append(
                    {
                        "name": f"test_{name}_empty_{param_name}",
                        "description": f"Test {name} with empty {param_name}",
                        "category": "edge_case",
                    }
                )

            if "Optional" in param_type or param.get("default") is not None:
                scenarios.append(
                    {
                        "name": f"test_{name}_without_{param_name}",
                        "description": f"Test {name} without {param_name}",
                        "category": "edge_case",
                    }
                )

        # Error case
        scenarios.append(
            {
                "name": f"test_{name}_raises_on_invalid_input",
                "description": f"Test {name} raises error on invalid input",
                "category": "error",
            }
        )

        return scenarios


# =============================================================================
# Test Template
# =============================================================================


class TestTemplate:
    """Template for generating test code."""

    def __init__(self, framework: str = "pytest"):
        """Initialize with test framework.

        Args:
            framework: Test framework (pytest or unittest)
        """
        self.framework = framework

    def render_function_test(
        self,
        name: str,
        symbol_name: str,
        assertions: list[str],
        setup: str = "",
        is_async: bool = False,
    ) -> str:
        """Render a function test.

        Args:
            name: Test function name
            symbol_name: Symbol being tested
            assertions: List of assertion statements
            setup: Setup code
            is_async: Whether the function is async

        Returns:
            Generated test code
        """
        async_prefix = "async " if is_async else ""
        await_prefix = "await " if is_async else ""

        lines = []
        lines.append(f'{async_prefix}def {name}(self):')
        lines.append(f'    """Test {symbol_name}."""')

        if setup:
            lines.append(f"    {setup}")

        lines.append(f"    result = {await_prefix}{symbol_name}()")

        for assertion in assertions:
            lines.append(f"    {assertion}")

        return "\n".join(lines)

    def render_fixture(self, name: str, return_value: str) -> str:
        """Render a fixture.

        Args:
            name: Fixture name
            return_value: Value to return

        Returns:
            Generated fixture code
        """
        if self.framework == "pytest":
            return f'''@pytest.fixture
def {name}():
    """Create {name} for testing."""
    return {return_value}'''
        else:
            return f'''def setUp(self):
    """Set up {name}."""
    self.{name} = {return_value}'''

    def render_mock(self, target: str, return_value: str) -> str:
        """Render a mock patch.

        Args:
            target: Module path to mock
            return_value: Value to return from mock

        Returns:
            Generated mock code
        """
        return f'''@patch("{target}")
def test_with_mock(self, mock_target):
    mock_target.return_value = {return_value}'''

    def render_async_test(
        self,
        name: str,
        symbol_name: str,
        assertions: list[str],
        setup: str = "",
    ) -> str:
        """Render an async test.

        Args:
            name: Test function name
            symbol_name: Symbol being tested
            assertions: List of assertion statements
            setup: Setup code

        Returns:
            Generated async test code
        """
        return self.render_function_test(
            name=name,
            symbol_name=symbol_name,
            assertions=assertions,
            setup=setup,
            is_async=True,
        )


# =============================================================================
# Test Generator
# =============================================================================


class TestGenerator:
    """Generates test code from analysis."""

    def __init__(
        self,
        graph_driver: Any,
        retriever: Any,
        config: Optional[TestGenerationConfig] = None,
    ):
        """Initialize generator.

        Args:
            graph_driver: Neo4j driver for CODE_* graph
            retriever: Retriever for finding patterns
            config: Optional configuration
        """
        self.graph_driver = graph_driver
        self.retriever = retriever
        self.config = config or TestGenerationConfig()

        self._analyzer = SymbolAnalyzer(graph_driver)
        self._pattern_matcher = PatternMatcher()
        self._coverage_analyzer = CoverageAnalyzer(retriever)
        self._template = TestTemplate(framework=self.config.test_framework)

    def generate(self, symbol: dict[str, Any]) -> TestSuite:
        """Generate tests for a symbol.

        Args:
            symbol: Symbol to generate tests for

        Returns:
            TestSuite with generated test cases
        """
        # Analyze symbol
        analysis = self._analyzer.analyze(symbol)

        # Get team patterns if enabled
        patterns = []
        if self.config.use_team_patterns:
            patterns = self._get_team_patterns()

        # Generate test cases
        test_cases = []

        # Happy path test
        test_cases.append(self._generate_happy_path_test(analysis))

        # Edge case tests
        if self.config.include_edge_cases:
            edge_cases = self._generate_edge_case_tests(analysis)
            test_cases.extend(edge_cases)

        # Error case tests
        if self.config.include_error_cases:
            error_cases = self._generate_error_case_tests(analysis)
            test_cases.extend(error_cases)

        # Limit to max tests
        test_cases = test_cases[: self.config.max_tests_per_symbol]

        # Generate imports
        imports = self._generate_imports(analysis)

        # Generate fixtures
        fixtures = []
        if self.config.include_fixtures:
            fixtures = self._generate_fixtures(analysis)

        return TestSuite(
            symbol_id=analysis["symbol_id"],
            symbol_name=analysis["name"],
            test_cases=test_cases,
            imports=imports,
            fixtures=fixtures,
        )

    def _get_team_patterns(self) -> list[TestPattern]:
        """Get test patterns from existing team tests."""
        patterns = []

        try:
            from retrieval.trihybrid import TriHybridQuery
        except ImportError:
            from openmemory.api.retrieval.trihybrid import TriHybridQuery

        try:
            query = TriHybridQuery(
                query_text="def test_",
                size=5,
            )
            result = self.retriever.retrieve(query, index_name="tests")

            for hit in result.hits:
                content = hit.source.get("content", "")
                if content:
                    extracted = self._pattern_matcher.extract_patterns(content)
                    patterns.extend(extracted)
        except Exception as e:
            logger.warning(f"Error getting team patterns: {e}")

        return patterns

    def _generate_happy_path_test(self, analysis: dict[str, Any]) -> TestCase:
        """Generate happy path test."""
        name = analysis["name"]
        is_async = analysis.get("is_async", False)

        test_name = f"test_{name}_success"
        description = f"Test {name} with valid inputs"

        if is_async:
            code = self._template.render_async_test(
                name=test_name,
                symbol_name=name,
                assertions=["assert result is not None"],
            )
        else:
            code = self._template.render_function_test(
                name=test_name,
                symbol_name=name,
                assertions=["assert result is not None"],
            )

        return TestCase(
            name=test_name,
            description=description,
            code=code,
            symbol_id=analysis["symbol_id"],
            category="happy_path",
        )

    def _generate_edge_case_tests(self, analysis: dict[str, Any]) -> list[TestCase]:
        """Generate edge case tests."""
        test_cases = []
        name = analysis["name"]
        params = analysis.get("parameters", [])

        for param in params:
            param_name = param.get("name", "")
            param_type = param.get("type", "") or ""

            # Empty list test
            if "list" in param_type.lower():
                test_name = f"test_{name}_empty_{param_name}"
                code = self._template.render_function_test(
                    name=test_name,
                    symbol_name=name,
                    assertions=["assert result == [] or result is not None"],
                    setup=f"# Test with empty {param_name}",
                )
                test_cases.append(
                    TestCase(
                        name=test_name,
                        description=f"Test {name} with empty {param_name}",
                        code=code,
                        symbol_id=analysis["symbol_id"],
                        category="edge_case",
                    )
                )

            # None value test for optionals
            if "Optional" in param_type or param.get("default") is not None:
                test_name = f"test_{name}_none_{param_name}"
                code = self._template.render_function_test(
                    name=test_name,
                    symbol_name=name,
                    assertions=["assert result is not None or result is None"],
                    setup=f"# Test with None {param_name}",
                )
                test_cases.append(
                    TestCase(
                        name=test_name,
                        description=f"Test {name} with None {param_name}",
                        code=code,
                        symbol_id=analysis["symbol_id"],
                        category="edge_case",
                    )
                )

        return test_cases

    def _generate_error_case_tests(self, analysis: dict[str, Any]) -> list[TestCase]:
        """Generate error case tests."""
        test_cases = []
        name = analysis["name"]

        # Invalid input test
        test_name = f"test_{name}_raises_on_invalid_input"
        code = f'''def {test_name}(self):
    """Test {name} raises error on invalid input."""
    with pytest.raises((ValueError, TypeError)):
        {name}(None)'''

        test_cases.append(
            TestCase(
                name=test_name,
                description=f"Test {name} raises error on invalid input",
                code=code,
                symbol_id=analysis["symbol_id"],
                category="error",
            )
        )

        return test_cases

    def _generate_imports(self, analysis: dict[str, Any]) -> list[str]:
        """Generate import statements."""
        imports = []

        if self.config.test_framework == "pytest":
            imports.append("import pytest")
        else:
            imports.append("import unittest")

        if self.config.include_mocks:
            imports.append("from unittest.mock import Mock, patch")

        # Import the symbol being tested
        # This is a simplified import - real implementation would resolve the module
        name = analysis["name"]
        file_path = analysis.get("file_path", "")

        if file_path:
            # Convert file path to module path
            module = file_path.replace("/", ".").replace(".py", "")
            module = module.lstrip(".")
            imports.append(f"from {module} import {name}")

        return imports

    def _generate_fixtures(self, analysis: dict[str, Any]) -> list[str]:
        """Generate fixtures for test dependencies."""
        fixtures = []
        params = analysis.get("parameters", [])

        for param in params:
            param_name = param.get("name", "")
            param_type = param.get("type", "") or ""

            if "list" in param_type.lower():
                fixture = self._template.render_fixture(
                    name=f"sample_{param_name}",
                    return_value="[]",
                )
                fixtures.append(fixture)
            elif "dict" in param_type.lower():
                fixture = self._template.render_fixture(
                    name=f"sample_{param_name}",
                    return_value="{}",
                )
                fixtures.append(fixture)
            elif "str" in param_type.lower():
                fixture = self._template.render_fixture(
                    name=f"sample_{param_name}",
                    return_value='"test_value"',
                )
                fixtures.append(fixture)

        return fixtures


# =============================================================================
# Main Tool
# =============================================================================


class TestGenerationTool:
    """MCP tool for test generation.

    This tool analyzes code symbols and generates test cases,
    applying team patterns and generating appropriate fixtures.
    """

    def __init__(
        self,
        graph_driver: Any,
        retriever: Any,
        parser: Any,
        config: Optional[TestGenerationConfig] = None,
    ):
        """Initialize test generation tool.

        Args:
            graph_driver: Neo4j driver for CODE_* graph
            retriever: TriHybridRetriever for patterns
            parser: ASTParser for symbol extraction
            config: Optional configuration
        """
        self.graph_driver = graph_driver
        self.retriever = retriever
        self.parser = parser
        self.config = config or TestGenerationConfig()

        self._generator = TestGenerator(
            graph_driver=graph_driver,
            retriever=retriever,
            config=self.config,
        )
        self._analyzer = SymbolAnalyzer(graph_driver)

    def generate_for_symbol(self, symbol_id: str) -> TestSuite:
        """Generate tests for a symbol by ID.

        Args:
            symbol_id: SCIP symbol ID

        Returns:
            TestSuite with generated tests
        """
        # Get symbol from graph
        try:
            node = self.graph_driver.get_node(symbol_id)
            if node is None:
                raise SymbolNotFoundError(f"Symbol not found: {symbol_id}")

            symbol = dict(node.properties)
            symbol["symbol_id"] = symbol_id

        except Exception as e:
            logger.warning(f"Error getting symbol: {e}")
            # Create minimal symbol for generation
            symbol = {
                "symbol_id": symbol_id,
                "name": symbol_id.split("/")[-1].rstrip("."),
                "kind": "function",
                "signature": "",
            }

        return self._generator.generate(symbol)

    def generate_for_file(self, file_path: str) -> TestSuite:
        """Generate tests for all symbols in a file.

        Args:
            file_path: Path to source file

        Returns:
            Combined TestSuite for all symbols
        """
        try:
            resolved_path = _resolve_source_path(file_path)
            parser = self.parser
            if parser is None:
                try:
                    from indexing.ast_parser import create_parser
                except ImportError:
                    from openmemory.api.indexing.ast_parser import create_parser

                try:
                    parser = create_parser()
                    self.parser = parser
                except Exception as exc:
                    logger.warning(f"AST parser unavailable, falling back: {exc}")

            symbols = []
            if parser is not None:
                result = parser.parse_file(resolved_path)
                symbols = result.symbols

            if not symbols and resolved_path.suffix == ".py":
                content = resolved_path.read_text(errors="ignore")
                symbols = extract_python_symbols(content)
            if not symbols:
                return TestSuite(
                    symbol_id="",
                    symbol_name=file_path,
                    test_cases=[],
                )

            all_test_cases = []
            all_imports = set()
            all_fixtures = []

            for symbol in symbols:
                symbol_type = getattr(symbol, "symbol_type", None)
                kind = (
                    symbol_type.value if symbol_type else getattr(symbol, "kind", "function")
                )
                if kind not in {"function", "class", "method"}:
                    continue
                if not getattr(symbol, "name", None):
                    continue
                symbol_id = (
                    getattr(symbol, "symbol_id", "")
                    or f"{resolved_path}:{symbol.name}"
                )
                symbol_dict = {
                    "symbol_id": symbol_id,
                    "name": symbol.name,
                    "kind": kind,
                    "signature": getattr(symbol, "signature", ""),
                    "docstring": getattr(symbol, "docstring", ""),
                    "file_path": str(resolved_path),
                }

                suite = self._generator.generate(symbol_dict)
                all_test_cases.extend(suite.test_cases)
                all_imports.update(suite.imports)
                all_fixtures.extend(suite.fixtures)

            return TestSuite(
                symbol_id=str(resolved_path),
                symbol_name=resolved_path.name,
                test_cases=all_test_cases,
                imports=list(all_imports),
                fixtures=all_fixtures,
            )

        except Exception as e:
            logger.warning(f"Error parsing file: {e}")
            return TestSuite(
                symbol_id="",
                symbol_name=file_path,
                test_cases=[],
            )

    def get_mcp_schema(self) -> dict[str, Any]:
        """Get MCP tool schema.

        Returns:
            MCP schema dictionary
        """
        return {
            "name": "generate_tests",
            "description": "Generates test cases for a code symbol or file",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "symbol_id": {
                        "type": "string",
                        "description": "SCIP symbol ID to generate tests for",
                    },
                    "file_path": {
                        "type": "string",
                        "description": "File path to generate tests for",
                    },
                    "framework": {
                        "type": "string",
                        "enum": ["pytest", "unittest"],
                        "default": "pytest",
                        "description": "Test framework to use",
                    },
                    "include_edge_cases": {
                        "type": "boolean",
                        "default": True,
                        "description": "Include edge case tests",
                    },
                    "include_error_cases": {
                        "type": "boolean",
                        "default": True,
                        "description": "Include error handling tests",
                    },
                },
            },
        }

    def execute(self, input_data: dict[str, Any]) -> dict[str, Any]:
        """Execute tool via MCP interface.

        Args:
            input_data: MCP input data

        Returns:
            Result dictionary
        """
        symbol_id = input_data.get("symbol_id")
        file_path = input_data.get("file_path")
        framework = input_data.get("framework", self.config.test_framework)
        include_edge_cases = input_data.get(
            "include_edge_cases", self.config.include_edge_cases
        )
        include_error_cases = input_data.get(
            "include_error_cases", self.config.include_error_cases
        )

        # Update config
        self.config.test_framework = framework
        self.config.include_edge_cases = include_edge_cases
        self.config.include_error_cases = include_error_cases
        self._generator.config = self.config
        self._generator._template = TestTemplate(framework=framework)

        if symbol_id:
            suite = self.generate_for_symbol(symbol_id)
        elif file_path:
            suite = self.generate_for_file(file_path)
        else:
            return {"error": "Either symbol_id or file_path is required"}

        return {
            "symbol_id": suite.symbol_id,
            "symbol_name": suite.symbol_name,
            "test_cases": [
                {
                    "name": tc.name,
                    "description": tc.description,
                    "category": tc.category,
                    "code": tc.code,
                }
                for tc in suite.test_cases
            ],
            "file_content": suite.render(),
        }


# =============================================================================
# Factory Function
# =============================================================================


def create_test_generation_tool(
    graph_driver: Any,
    retriever: Any,
    parser: Any,
    config: Optional[TestGenerationConfig] = None,
) -> TestGenerationTool:
    """Create a test generation tool.

    Args:
        graph_driver: Neo4j driver for CODE_* graph
        retriever: TriHybridRetriever for patterns
        parser: ASTParser for symbol extraction
        config: Optional configuration

    Returns:
        Configured TestGenerationTool
    """
    return TestGenerationTool(
        graph_driver=graph_driver,
        retriever=retriever,
        parser=parser,
        config=config,
    )
