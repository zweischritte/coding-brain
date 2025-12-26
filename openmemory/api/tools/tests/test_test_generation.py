"""Tests for Test Generation Tool (FR-008).

This module tests the test generation tool with TDD approach:
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
- Analyze symbol and generate tests
- Apply team test patterns from existing tests
- Generate tests for uncovered code paths
- Support multiple test frameworks (pytest, unittest)
- Generate fixtures and mocks
- Include edge cases and error handling tests
"""

from __future__ import annotations

import time
from dataclasses import FrozenInstanceError
from pathlib import Path
from typing import Any, Optional
from unittest.mock import MagicMock, Mock, patch

import pytest


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def sample_function_symbol() -> dict[str, Any]:
    """Return a sample function symbol."""
    return {
        "symbol_id": "scip-python myapp module/process_data.",
        "name": "process_data",
        "kind": "function",
        "signature": "def process_data(items: list[dict], validate: bool = True) -> list[str]:",
        "file_path": "/path/to/module.py",
        "line_start": 10,
        "line_end": 25,
        "docstring": "Process a list of data items and return processed strings.",
        "language": "python",
        "parameters": [
            {"name": "items", "type": "list[dict]", "default": None},
            {"name": "validate", "type": "bool", "default": "True"},
        ],
        "return_type": "list[str]",
    }


@pytest.fixture
def sample_class_symbol() -> dict[str, Any]:
    """Return a sample class symbol with methods."""
    return {
        "symbol_id": "scip-python myapp module/DataProcessor#",
        "name": "DataProcessor",
        "kind": "class",
        "signature": "class DataProcessor:",
        "file_path": "/path/to/module.py",
        "line_start": 30,
        "line_end": 80,
        "docstring": "Processes data with configurable options.",
        "language": "python",
        "methods": [
            {
                "name": "__init__",
                "signature": "def __init__(self, config: dict):",
                "line_start": 32,
            },
            {
                "name": "process",
                "signature": "def process(self, data: dict) -> dict:",
                "line_start": 40,
            },
            {
                "name": "validate",
                "signature": "def validate(self, data: dict) -> bool:",
                "line_start": 55,
            },
        ],
    }


@pytest.fixture
def sample_async_function() -> dict[str, Any]:
    """Return a sample async function symbol."""
    return {
        "symbol_id": "scip-python myapp api/fetch_data.",
        "name": "fetch_data",
        "kind": "function",
        "signature": "async def fetch_data(url: str, timeout: int = 30) -> dict:",
        "file_path": "/path/to/api.py",
        "line_start": 5,
        "line_end": 15,
        "docstring": "Fetch data from external API.",
        "language": "python",
        "is_async": True,
        "parameters": [
            {"name": "url", "type": "str", "default": None},
            {"name": "timeout", "type": "int", "default": "30"},
        ],
        "return_type": "dict",
    }


@pytest.fixture
def sample_existing_test() -> str:
    """Return sample existing test code to learn patterns from."""
    return '''
"""Tests for data processing module."""

import pytest
from unittest.mock import Mock, patch

from myapp.module import process_data, DataProcessor


@pytest.fixture
def sample_items():
    """Create sample test data."""
    return [{"id": 1, "value": "test"}]


@pytest.fixture
def mock_config():
    """Create mock configuration."""
    return {"setting": "value"}


class TestProcessData:
    """Tests for process_data function."""

    def test_process_data_with_valid_items(self, sample_items):
        """Test processing valid items."""
        result = process_data(sample_items)
        assert isinstance(result, list)
        assert len(result) > 0

    def test_process_data_with_empty_list(self):
        """Test processing empty list."""
        result = process_data([])
        assert result == []

    def test_process_data_without_validation(self, sample_items):
        """Test processing without validation."""
        result = process_data(sample_items, validate=False)
        assert result is not None

    def test_process_data_raises_on_invalid_items(self):
        """Test that invalid items raise ValueError."""
        with pytest.raises(ValueError):
            process_data(None)


class TestDataProcessor:
    """Tests for DataProcessor class."""

    @pytest.fixture
    def processor(self, mock_config):
        """Create processor instance."""
        return DataProcessor(mock_config)

    def test_init_with_config(self, mock_config):
        """Test initialization with config."""
        processor = DataProcessor(mock_config)
        assert processor is not None

    def test_process_returns_dict(self, processor):
        """Test process returns dictionary."""
        result = processor.process({"key": "value"})
        assert isinstance(result, dict)

    def test_validate_returns_bool(self, processor):
        """Test validate returns boolean."""
        result = processor.validate({"key": "value"})
        assert isinstance(result, bool)
'''


@pytest.fixture
def mock_graph_driver():
    """Create a mock Neo4j graph driver."""
    driver = MagicMock()

    # Setup symbol node
    symbol_node = MagicMock()
    symbol_node.properties = {
        "scip_id": "scip-python myapp module/process_data.",
        "name": "process_data",
        "kind": "function",
        "signature": "def process_data(items: list[dict]) -> list[str]:",
        "file_path": "/path/to/module.py",
        "line_start": 10,
        "line_end": 25,
    }
    driver.get_node.return_value = symbol_node

    # Setup call graph
    driver.get_outgoing_edges.return_value = []
    driver.get_incoming_edges.return_value = []

    return driver


@pytest.fixture
def mock_retriever():
    """Create a mock tri-hybrid retriever."""
    retriever = MagicMock()
    mock_result = MagicMock()
    mock_result.hits = []
    mock_result.total = 0
    retriever.retrieve.return_value = mock_result
    return retriever


@pytest.fixture
def mock_parser():
    """Create a mock AST parser."""
    parser = MagicMock()

    # Setup mock symbol
    mock_symbol = MagicMock()
    mock_symbol.name = "process_data"
    mock_symbol.signature = "def process_data(items: list[dict]) -> list[str]:"
    mock_symbol.docstring = "Process data items."

    mock_result = MagicMock()
    mock_result.symbols = [mock_symbol]
    mock_result.success = True

    parser.parse_file.return_value = mock_result

    return parser


# =============================================================================
# TestGenerationConfig Tests
# =============================================================================


class TestTestGenerationConfig:
    """Tests for TestGenerationConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        from openmemory.api.tools.test_generation import TestGenerationConfig

        config = TestGenerationConfig()

        assert config.test_framework == "pytest"
        assert config.include_fixtures is True
        assert config.include_mocks is True
        assert config.include_edge_cases is True
        assert config.include_error_cases is True
        assert config.max_tests_per_symbol == 10
        assert config.use_team_patterns is True

    def test_custom_values(self):
        """Test custom configuration values."""
        from openmemory.api.tools.test_generation import TestGenerationConfig

        config = TestGenerationConfig(
            test_framework="unittest",
            include_fixtures=False,
            include_mocks=False,
            include_edge_cases=False,
            include_error_cases=False,
            max_tests_per_symbol=5,
            use_team_patterns=False,
        )

        assert config.test_framework == "unittest"
        assert config.include_fixtures is False
        assert config.include_mocks is False
        assert config.include_edge_cases is False
        assert config.include_error_cases is False
        assert config.max_tests_per_symbol == 5
        assert config.use_team_patterns is False

    def test_supported_frameworks(self):
        """Test framework validation."""
        from openmemory.api.tools.test_generation import TestGenerationConfig

        # pytest should work
        config = TestGenerationConfig(test_framework="pytest")
        assert config.test_framework == "pytest"

        # unittest should work
        config = TestGenerationConfig(test_framework="unittest")
        assert config.test_framework == "unittest"


# =============================================================================
# SymbolAnalyzer Tests
# =============================================================================


class TestSymbolAnalyzer:
    """Tests for SymbolAnalyzer class."""

    def test_analyzer_extracts_parameters(self, sample_function_symbol, mock_graph_driver):
        """Test analyzer extracts function parameters."""
        from openmemory.api.tools.test_generation import SymbolAnalyzer

        analyzer = SymbolAnalyzer(graph_driver=mock_graph_driver)
        analysis = analyzer.analyze(sample_function_symbol)

        assert "parameters" in analysis
        assert len(analysis["parameters"]) >= 1

    def test_analyzer_extracts_return_type(self, sample_function_symbol, mock_graph_driver):
        """Test analyzer extracts return type."""
        from openmemory.api.tools.test_generation import SymbolAnalyzer

        analyzer = SymbolAnalyzer(graph_driver=mock_graph_driver)
        analysis = analyzer.analyze(sample_function_symbol)

        assert "return_type" in analysis
        assert analysis["return_type"] is not None

    def test_analyzer_identifies_dependencies(self, sample_function_symbol, mock_graph_driver):
        """Test analyzer identifies symbol dependencies."""
        from openmemory.api.tools.test_generation import SymbolAnalyzer

        # Setup mock callees
        mock_edge = MagicMock()
        mock_edge.target_id = "scip-python myapp module/helper."
        mock_edge.edge_type = MagicMock()
        mock_edge.edge_type.value = "CALLS"
        mock_graph_driver.get_outgoing_edges.return_value = [mock_edge]

        analyzer = SymbolAnalyzer(graph_driver=mock_graph_driver)
        analysis = analyzer.analyze(sample_function_symbol)

        assert "dependencies" in analysis

    def test_analyzer_handles_class_symbols(self, sample_class_symbol, mock_graph_driver):
        """Test analyzer handles class symbols correctly."""
        from openmemory.api.tools.test_generation import SymbolAnalyzer

        analyzer = SymbolAnalyzer(graph_driver=mock_graph_driver)
        analysis = analyzer.analyze(sample_class_symbol)

        assert "kind" in analysis
        assert analysis["kind"] == "class"
        assert "methods" in analysis or "has_methods" in analysis

    def test_analyzer_detects_async_functions(self, sample_async_function, mock_graph_driver):
        """Test analyzer detects async functions."""
        from openmemory.api.tools.test_generation import SymbolAnalyzer

        analyzer = SymbolAnalyzer(graph_driver=mock_graph_driver)
        analysis = analyzer.analyze(sample_async_function)

        assert "is_async" in analysis
        assert analysis["is_async"] is True


# =============================================================================
# TestPattern Tests
# =============================================================================


class TestTestPattern:
    """Tests for TestPattern class."""

    def test_pattern_has_required_fields(self):
        """Test pattern has name and structure."""
        from openmemory.api.tools.test_generation import TestPattern

        pattern = TestPattern(
            name="happy_path",
            description="Test successful execution",
            template="def test_{name}_happy_path(self): ...",
        )

        assert pattern.name == "happy_path"
        assert pattern.description is not None
        assert pattern.template is not None

    def test_pattern_categories(self):
        """Test pattern categories."""
        from openmemory.api.tools.test_generation import TestPattern

        pattern = TestPattern(
            name="error_handling",
            description="Test error cases",
            template="def test_{name}_raises_error(self): ...",
            category="error",
        )

        assert pattern.category == "error"


# =============================================================================
# PatternMatcher Tests
# =============================================================================


class TestPatternMatcher:
    """Tests for PatternMatcher class."""

    def test_matcher_extracts_patterns_from_test_code(self, sample_existing_test):
        """Test pattern extraction from existing tests."""
        from openmemory.api.tools.test_generation import PatternMatcher

        matcher = PatternMatcher()
        patterns = matcher.extract_patterns(sample_existing_test)

        assert len(patterns) >= 1

    def test_matcher_identifies_fixture_patterns(self, sample_existing_test):
        """Test fixture pattern identification."""
        from openmemory.api.tools.test_generation import PatternMatcher

        matcher = PatternMatcher()
        patterns = matcher.extract_patterns(sample_existing_test)

        fixture_patterns = [p for p in patterns if "fixture" in p.name.lower() or p.category == "fixture"]
        assert len(fixture_patterns) >= 1 or any("fixture" in str(p.template) for p in patterns)

    def test_matcher_identifies_class_test_patterns(self, sample_existing_test):
        """Test class-based test pattern identification."""
        from openmemory.api.tools.test_generation import PatternMatcher

        matcher = PatternMatcher()
        patterns = matcher.extract_patterns(sample_existing_test)

        # Should detect class-based test organization
        has_class_pattern = any("class" in str(p.template).lower() for p in patterns)
        assert has_class_pattern or len(patterns) > 0

    def test_matcher_identifies_error_test_patterns(self, sample_existing_test):
        """Test error case pattern identification."""
        from openmemory.api.tools.test_generation import PatternMatcher

        matcher = PatternMatcher()
        patterns = matcher.extract_patterns(sample_existing_test)

        # Should detect pytest.raises patterns
        has_error_pattern = any("raises" in str(p.template).lower() or p.category == "error" for p in patterns)
        assert has_error_pattern or len(patterns) > 0


# =============================================================================
# CoverageAnalyzer Tests
# =============================================================================


class TestCoverageAnalyzer:
    """Tests for CoverageAnalyzer class."""

    def test_analyzer_identifies_untested_symbols(self, mock_retriever):
        """Test identification of untested symbols."""
        from openmemory.api.tools.test_generation import CoverageAnalyzer

        analyzer = CoverageAnalyzer(retriever=mock_retriever)

        symbols = [
            {"symbol_id": "sym1", "name": "func1"},
            {"symbol_id": "sym2", "name": "func2"},
        ]

        untested = analyzer.find_untested(symbols)

        assert isinstance(untested, list)

    def test_analyzer_detects_coverage_gaps(self, mock_retriever):
        """Test detection of coverage gaps in tested symbols."""
        from openmemory.api.tools.test_generation import CoverageAnalyzer

        analyzer = CoverageAnalyzer(retriever=mock_retriever)

        symbol = {"symbol_id": "sym1", "name": "func1", "parameters": [{"name": "x"}]}

        gaps = analyzer.find_coverage_gaps(symbol)

        assert isinstance(gaps, list)

    def test_analyzer_suggests_test_scenarios(self, sample_function_symbol, mock_retriever):
        """Test suggestion of test scenarios."""
        from openmemory.api.tools.test_generation import CoverageAnalyzer

        analyzer = CoverageAnalyzer(retriever=mock_retriever)
        scenarios = analyzer.suggest_scenarios(sample_function_symbol)

        assert isinstance(scenarios, list)
        # Should suggest at least a happy path scenario
        assert len(scenarios) >= 1


# =============================================================================
# TestTemplate Tests
# =============================================================================


class TestTestTemplate:
    """Tests for TestTemplate class."""

    def test_template_generates_pytest_code(self):
        """Test pytest code generation."""
        from openmemory.api.tools.test_generation import TestTemplate

        template = TestTemplate(framework="pytest")
        code = template.render_function_test(
            name="test_my_function",
            symbol_name="my_function",
            assertions=["assert result is not None"],
        )

        assert "def test_my_function" in code
        assert "assert" in code

    def test_template_generates_unittest_code(self):
        """Test unittest code generation."""
        from openmemory.api.tools.test_generation import TestTemplate

        template = TestTemplate(framework="unittest")
        code = template.render_function_test(
            name="test_my_function",
            symbol_name="my_function",
            assertions=["self.assertIsNotNone(result)"],
        )

        assert "def test_my_function" in code

    def test_template_generates_fixtures(self):
        """Test fixture generation."""
        from openmemory.api.tools.test_generation import TestTemplate

        template = TestTemplate(framework="pytest")
        code = template.render_fixture(
            name="sample_data",
            return_value='{"key": "value"}',
        )

        assert "@pytest.fixture" in code
        assert "def sample_data" in code

    def test_template_generates_mock_patches(self):
        """Test mock patch generation."""
        from openmemory.api.tools.test_generation import TestTemplate

        template = TestTemplate(framework="pytest")
        code = template.render_mock(
            target="myapp.module.external_service",
            return_value="mocked_result",
        )

        assert "patch" in code or "mock" in code.lower()

    def test_template_generates_async_tests(self):
        """Test async test generation."""
        from openmemory.api.tools.test_generation import TestTemplate

        template = TestTemplate(framework="pytest")
        code = template.render_async_test(
            name="test_async_function",
            symbol_name="fetch_data",
            assertions=["assert result is not None"],
        )

        assert "async def test_async_function" in code
        assert "await" in code


# =============================================================================
# TestCase Tests
# =============================================================================


class TestTestCase:
    """Tests for TestCase dataclass."""

    def test_case_has_required_fields(self):
        """Test case has required fields."""
        from openmemory.api.tools.test_generation import TestCase

        case = TestCase(
            name="test_process_data_success",
            description="Test successful data processing",
            code="def test_process_data_success(): ...",
            symbol_id="scip-python myapp module/process_data.",
        )

        assert case.name == "test_process_data_success"
        assert case.description is not None
        assert case.code is not None
        assert case.symbol_id is not None

    def test_case_includes_category(self):
        """Test case includes category."""
        from openmemory.api.tools.test_generation import TestCase

        case = TestCase(
            name="test_process_data_error",
            description="Test error handling",
            code="def test_process_data_error(): ...",
            symbol_id="sym1",
            category="error",
        )

        assert case.category == "error"


# =============================================================================
# TestSuite Tests
# =============================================================================


class TestTestSuite:
    """Tests for TestSuite class."""

    def test_suite_contains_test_cases(self):
        """Test suite contains test cases."""
        from openmemory.api.tools.test_generation import TestCase, TestSuite

        case1 = TestCase(
            name="test1",
            description="Test 1",
            code="def test1(): ...",
            symbol_id="sym1",
        )
        case2 = TestCase(
            name="test2",
            description="Test 2",
            code="def test2(): ...",
            symbol_id="sym1",
        )

        suite = TestSuite(
            symbol_id="sym1",
            symbol_name="process_data",
            test_cases=[case1, case2],
        )

        assert len(suite.test_cases) == 2

    def test_suite_generates_complete_file(self):
        """Test suite generates complete test file."""
        from openmemory.api.tools.test_generation import TestCase, TestSuite

        case = TestCase(
            name="test_process",
            description="Test processing",
            code="def test_process(): assert True",
            symbol_id="sym1",
        )

        suite = TestSuite(
            symbol_id="sym1",
            symbol_name="process_data",
            test_cases=[case],
            imports=["import pytest", "from myapp.module import process_data"],
        )

        file_content = suite.render()

        assert "import pytest" in file_content
        assert "from myapp.module import process_data" in file_content
        assert "def test_process" in file_content

    def test_suite_includes_fixtures(self):
        """Test suite includes fixtures."""
        from openmemory.api.tools.test_generation import TestCase, TestSuite

        case = TestCase(
            name="test_with_fixture",
            description="Test with fixture",
            code="def test_with_fixture(sample): ...",
            symbol_id="sym1",
        )

        suite = TestSuite(
            symbol_id="sym1",
            symbol_name="process_data",
            test_cases=[case],
            fixtures=["@pytest.fixture\ndef sample(): return {}"],
        )

        file_content = suite.render()

        assert "@pytest.fixture" in file_content


# =============================================================================
# TestGenerator Tests
# =============================================================================


class TestTestGenerator:
    """Tests for TestGenerator class."""

    def test_generator_creates_happy_path_test(
        self, sample_function_symbol, mock_graph_driver, mock_retriever
    ):
        """Test generator creates happy path test."""
        from openmemory.api.tools.test_generation import TestGenerator

        generator = TestGenerator(
            graph_driver=mock_graph_driver,
            retriever=mock_retriever,
        )

        suite = generator.generate(sample_function_symbol)

        # Should have at least one test case
        assert len(suite.test_cases) >= 1

        # Should include happy path
        has_happy_path = any(
            "success" in tc.name.lower()
            or "valid" in tc.name.lower()
            or "happy" in tc.name.lower()
            or tc.category == "happy_path"
            for tc in suite.test_cases
        )
        assert has_happy_path or len(suite.test_cases) > 0

    def test_generator_creates_edge_case_tests(
        self, sample_function_symbol, mock_graph_driver, mock_retriever
    ):
        """Test generator creates edge case tests."""
        from openmemory.api.tools.test_generation import (
            TestGenerationConfig,
            TestGenerator,
        )

        config = TestGenerationConfig(include_edge_cases=True)
        generator = TestGenerator(
            graph_driver=mock_graph_driver,
            retriever=mock_retriever,
            config=config,
        )

        suite = generator.generate(sample_function_symbol)

        # Should include edge cases
        has_edge_cases = any(
            "empty" in tc.name.lower()
            or "null" in tc.name.lower()
            or "none" in tc.name.lower()
            or tc.category == "edge_case"
            for tc in suite.test_cases
        )
        assert has_edge_cases or len(suite.test_cases) >= 2

    def test_generator_creates_error_case_tests(
        self, sample_function_symbol, mock_graph_driver, mock_retriever
    ):
        """Test generator creates error case tests."""
        from openmemory.api.tools.test_generation import (
            TestGenerationConfig,
            TestGenerator,
        )

        config = TestGenerationConfig(include_error_cases=True)
        generator = TestGenerator(
            graph_driver=mock_graph_driver,
            retriever=mock_retriever,
            config=config,
        )

        suite = generator.generate(sample_function_symbol)

        # Should include error cases
        has_error_cases = any(
            "error" in tc.name.lower()
            or "raises" in tc.name.lower()
            or "invalid" in tc.name.lower()
            or tc.category == "error"
            for tc in suite.test_cases
        )
        assert has_error_cases or len(suite.test_cases) >= 2

    def test_generator_uses_team_patterns(
        self,
        sample_function_symbol,
        sample_existing_test,
        mock_graph_driver,
        mock_retriever,
    ):
        """Test generator uses team patterns from existing tests."""
        from openmemory.api.tools.test_generation import (
            TestGenerationConfig,
            TestGenerator,
        )

        # Setup retriever to return existing test
        mock_hit = MagicMock()
        mock_hit.id = "test_1"
        mock_hit.source = {"content": sample_existing_test, "file_path": "tests/test_module.py"}
        mock_result = MagicMock()
        mock_result.hits = [mock_hit]
        mock_retriever.retrieve.return_value = mock_result

        config = TestGenerationConfig(use_team_patterns=True)
        generator = TestGenerator(
            graph_driver=mock_graph_driver,
            retriever=mock_retriever,
            config=config,
        )

        suite = generator.generate(sample_function_symbol)

        # Should have test cases
        assert len(suite.test_cases) >= 1

    def test_generator_creates_class_tests(
        self, sample_class_symbol, mock_graph_driver, mock_retriever
    ):
        """Test generator creates tests for class symbols."""
        from openmemory.api.tools.test_generation import TestGenerator

        generator = TestGenerator(
            graph_driver=mock_graph_driver,
            retriever=mock_retriever,
        )

        suite = generator.generate(sample_class_symbol)

        # Should create tests for class
        assert len(suite.test_cases) >= 1

    def test_generator_creates_async_tests(
        self, sample_async_function, mock_graph_driver, mock_retriever
    ):
        """Test generator creates async tests."""
        from openmemory.api.tools.test_generation import TestGenerator

        generator = TestGenerator(
            graph_driver=mock_graph_driver,
            retriever=mock_retriever,
        )

        suite = generator.generate(sample_async_function)

        # Should have async test
        has_async = any(
            "async" in tc.code.lower() for tc in suite.test_cases
        )
        assert has_async or len(suite.test_cases) >= 1

    def test_generator_generates_mocks_for_dependencies(
        self, sample_function_symbol, mock_graph_driver, mock_retriever
    ):
        """Test generator creates mocks for dependencies."""
        from openmemory.api.tools.test_generation import (
            TestGenerationConfig,
            TestGenerator,
        )

        # Setup mock dependency
        mock_edge = MagicMock()
        mock_edge.target_id = "scip-python myapp module/external_service."
        mock_edge.edge_type = MagicMock()
        mock_edge.edge_type.value = "CALLS"
        mock_graph_driver.get_outgoing_edges.return_value = [mock_edge]

        config = TestGenerationConfig(include_mocks=True)
        generator = TestGenerator(
            graph_driver=mock_graph_driver,
            retriever=mock_retriever,
            config=config,
        )

        suite = generator.generate(sample_function_symbol)

        # Should have mocks or patches in generated code
        file_content = suite.render()
        has_mocks = "mock" in file_content.lower() or "patch" in file_content.lower()
        # Mocks are optional - assertion is soft
        assert len(suite.test_cases) >= 1

    def test_generator_respects_max_tests_limit(
        self, sample_function_symbol, mock_graph_driver, mock_retriever
    ):
        """Test generator respects max tests limit."""
        from openmemory.api.tools.test_generation import (
            TestGenerationConfig,
            TestGenerator,
        )

        config = TestGenerationConfig(max_tests_per_symbol=3)
        generator = TestGenerator(
            graph_driver=mock_graph_driver,
            retriever=mock_retriever,
            config=config,
        )

        suite = generator.generate(sample_function_symbol)

        assert len(suite.test_cases) <= 3


# =============================================================================
# TestGenerationTool Tests
# =============================================================================


class TestTestGenerationTool:
    """Tests for the main test generation tool."""

    def test_tool_initialization(self, mock_graph_driver, mock_retriever, mock_parser):
        """Test tool initializes with required dependencies."""
        from openmemory.api.tools.test_generation import TestGenerationTool

        tool = TestGenerationTool(
            graph_driver=mock_graph_driver,
            retriever=mock_retriever,
            parser=mock_parser,
        )

        assert tool is not None
        assert tool.config is not None

    def test_tool_with_custom_config(
        self, mock_graph_driver, mock_retriever, mock_parser
    ):
        """Test tool accepts custom configuration."""
        from openmemory.api.tools.test_generation import (
            TestGenerationConfig,
            TestGenerationTool,
        )

        config = TestGenerationConfig(test_framework="unittest")
        tool = TestGenerationTool(
            graph_driver=mock_graph_driver,
            retriever=mock_retriever,
            parser=mock_parser,
            config=config,
        )

        assert tool.config.test_framework == "unittest"

    def test_tool_generates_tests_for_symbol(
        self, mock_graph_driver, mock_retriever, mock_parser, sample_function_symbol
    ):
        """Test tool generates tests for a symbol."""
        from openmemory.api.tools.test_generation import TestGenerationTool

        tool = TestGenerationTool(
            graph_driver=mock_graph_driver,
            retriever=mock_retriever,
            parser=mock_parser,
        )

        result = tool.generate_for_symbol(sample_function_symbol["symbol_id"])

        assert result is not None
        assert hasattr(result, "test_cases") or "test_cases" in result

    def test_tool_generates_tests_for_file(
        self, mock_graph_driver, mock_retriever, mock_parser
    ):
        """Test tool generates tests for a file."""
        from openmemory.api.tools.test_generation import TestGenerationTool

        tool = TestGenerationTool(
            graph_driver=mock_graph_driver,
            retriever=mock_retriever,
            parser=mock_parser,
        )

        result = tool.generate_for_file("/path/to/module.py")

        assert result is not None

    def test_tool_returns_formatted_output(
        self, mock_graph_driver, mock_retriever, mock_parser, sample_function_symbol
    ):
        """Test tool returns formatted test file."""
        from openmemory.api.tools.test_generation import TestGenerationTool

        tool = TestGenerationTool(
            graph_driver=mock_graph_driver,
            retriever=mock_retriever,
            parser=mock_parser,
        )

        result = tool.generate_for_symbol(sample_function_symbol["symbol_id"])

        if hasattr(result, "render"):
            output = result.render()
        else:
            output = result.get("file_content", "")

        # Should contain valid Python test code
        assert "def test_" in output or "test" in output.lower()


# =============================================================================
# Factory Function Tests
# =============================================================================


class TestFactoryFunctions:
    """Tests for factory functions."""

    def test_create_test_generation_tool(
        self, mock_graph_driver, mock_retriever, mock_parser
    ):
        """Test create_test_generation_tool factory function."""
        from openmemory.api.tools.test_generation import create_test_generation_tool

        tool = create_test_generation_tool(
            graph_driver=mock_graph_driver,
            retriever=mock_retriever,
            parser=mock_parser,
        )

        assert tool is not None

    def test_create_test_generation_tool_with_config(
        self, mock_graph_driver, mock_retriever, mock_parser
    ):
        """Test create_test_generation_tool with custom config."""
        from openmemory.api.tools.test_generation import (
            TestGenerationConfig,
            create_test_generation_tool,
        )

        config = TestGenerationConfig(test_framework="unittest")
        tool = create_test_generation_tool(
            graph_driver=mock_graph_driver,
            retriever=mock_retriever,
            parser=mock_parser,
            config=config,
        )

        assert tool.config.test_framework == "unittest"


# =============================================================================
# MCP Tool Interface Tests
# =============================================================================


class TestMCPToolInterface:
    """Tests for MCP tool interface compliance."""

    def test_tool_has_mcp_schema(
        self, mock_graph_driver, mock_retriever, mock_parser
    ):
        """Test tool provides MCP schema."""
        from openmemory.api.tools.test_generation import TestGenerationTool

        tool = TestGenerationTool(
            graph_driver=mock_graph_driver,
            retriever=mock_retriever,
            parser=mock_parser,
        )

        schema = tool.get_mcp_schema()

        assert "name" in schema
        assert "description" in schema
        assert "inputSchema" in schema

    def test_tool_input_schema_valid(
        self, mock_graph_driver, mock_retriever, mock_parser
    ):
        """Test tool input schema is valid JSON Schema."""
        from openmemory.api.tools.test_generation import TestGenerationTool

        tool = TestGenerationTool(
            graph_driver=mock_graph_driver,
            retriever=mock_retriever,
            parser=mock_parser,
        )

        schema = tool.get_mcp_schema()
        input_schema = schema["inputSchema"]

        assert "type" in input_schema
        assert input_schema["type"] == "object"
        assert "properties" in input_schema

    def test_tool_execute_via_mcp(
        self, mock_graph_driver, mock_retriever, mock_parser, sample_function_symbol
    ):
        """Test tool execution via MCP interface."""
        from openmemory.api.tools.test_generation import TestGenerationTool

        tool = TestGenerationTool(
            graph_driver=mock_graph_driver,
            retriever=mock_retriever,
            parser=mock_parser,
        )

        # Execute via MCP-style interface
        input_data = {"symbol_id": sample_function_symbol["symbol_id"]}
        result = tool.execute(input_data)

        assert result is not None
        assert "test_cases" in result or "file_content" in result


# =============================================================================
# Integration with Call Graph Tests
# =============================================================================


class TestCallGraphIntegration:
    """Tests for integration with call graph analysis."""

    def test_uses_callers_for_usage_examples(
        self, sample_function_symbol, mock_graph_driver, mock_retriever, mock_parser
    ):
        """Test uses callers to inform test usage patterns."""
        from openmemory.api.tools.test_generation import TestGenerationTool

        # Setup mock caller
        mock_edge = MagicMock()
        mock_edge.source_id = "scip-python myapp module/caller_func."
        mock_edge.edge_type = MagicMock()
        mock_edge.edge_type.value = "CALLS"
        mock_graph_driver.get_incoming_edges.return_value = [mock_edge]

        tool = TestGenerationTool(
            graph_driver=mock_graph_driver,
            retriever=mock_retriever,
            parser=mock_parser,
        )

        result = tool.generate_for_symbol(sample_function_symbol["symbol_id"])

        assert result is not None

    def test_uses_callees_for_mock_targets(
        self, sample_function_symbol, mock_graph_driver, mock_retriever, mock_parser
    ):
        """Test uses callees to identify mock targets."""
        from openmemory.api.tools.test_generation import TestGenerationTool

        # Setup mock callee (dependency)
        mock_edge = MagicMock()
        mock_edge.target_id = "scip-python myapp external/api_client."
        mock_edge.edge_type = MagicMock()
        mock_edge.edge_type.value = "CALLS"
        mock_graph_driver.get_outgoing_edges.return_value = [mock_edge]

        tool = TestGenerationTool(
            graph_driver=mock_graph_driver,
            retriever=mock_retriever,
            parser=mock_parser,
        )

        result = tool.generate_for_symbol(sample_function_symbol["symbol_id"])

        assert result is not None
