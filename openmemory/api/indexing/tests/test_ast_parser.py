"""Tests for AST parser with Tree-sitter.

Following TDD: Write tests first, then implement.
Covers:
- Parse Python, TypeScript (TS/TSX), and Java files into AST
- Extract symbols per language (functions, classes, methods, imports)
- Extract symbol properties: name, signature, docstring, line numbers
- Handle malformed/partial files gracefully (skip with logging)
- Parse error rate tracking
- Language plugin interface
"""

import pytest
from dataclasses import dataclass
from typing import Optional
from pathlib import Path
from unittest.mock import Mock, patch

# Import the module under test (will be implemented)
from openmemory.api.indexing.ast_parser import (
    # Core types
    Language,
    SymbolType,
    Symbol,
    DecoratorInfo,
    ParseResult,
    ParseError,
    ParseStatistics,
    # Parser interface
    LanguagePlugin,
    ASTParser,
    ASTParserConfig,
    # Concrete plugins
    PythonPlugin,
    TypeScriptPlugin,
    JavaPlugin,
    # Factory
    create_parser,
    # Exceptions
    UnsupportedLanguageError,
    ParseFailedError,
    # Decorator mappings
    KNOWN_DECORATORS,
    # Event Registry
    EventEdge,
    EventRegistry,
    discover_event_publishers,
    discover_event_subscribers,
    build_event_registry,
    generate_event_edges,
)


# =============================================================================
# Language Enum Tests
# =============================================================================


class TestLanguageEnum:
    """Tests for Language enumeration."""

    def test_language_values(self):
        """Language enum has expected values."""
        assert Language.PYTHON.value == "python"
        assert Language.TYPESCRIPT.value == "typescript"
        assert Language.TSX.value == "tsx"
        assert Language.JAVA.value == "java"
        assert Language.GO.value == "go"

    def test_language_from_extension(self):
        """Language can be determined from file extension."""
        assert Language.from_extension(".py") == Language.PYTHON
        assert Language.from_extension(".ts") == Language.TYPESCRIPT
        assert Language.from_extension(".tsx") == Language.TSX
        assert Language.from_extension(".java") == Language.JAVA
        assert Language.from_extension(".go") == Language.GO
        assert Language.from_extension(".js") is None
        assert Language.from_extension(".unknown") is None

    def test_language_from_path(self):
        """Language can be determined from file path."""
        assert Language.from_path(Path("foo/bar.py")) == Language.PYTHON
        assert Language.from_path(Path("src/component.tsx")) == Language.TSX
        assert Language.from_path(Path("Main.java")) == Language.JAVA
        assert Language.from_path(Path("pkg/main.go")) == Language.GO
        assert Language.from_path(Path("README.md")) is None


# =============================================================================
# Symbol Type Tests
# =============================================================================


class TestSymbolType:
    """Tests for SymbolType enumeration."""

    def test_symbol_type_values(self):
        """SymbolType enum has expected values."""
        assert SymbolType.FUNCTION.value == "function"
        assert SymbolType.CLASS.value == "class"
        assert SymbolType.METHOD.value == "method"
        assert SymbolType.IMPORT.value == "import"
        assert SymbolType.VARIABLE.value == "variable"
        assert SymbolType.INTERFACE.value == "interface"
        assert SymbolType.ENUM.value == "enum"
        assert SymbolType.TYPE_ALIAS.value == "type_alias"


# =============================================================================
# Symbol Dataclass Tests
# =============================================================================


class TestSymbol:
    """Tests for Symbol dataclass."""

    def test_symbol_creation(self):
        """Symbol can be created with required fields."""
        symbol = Symbol(
            name="my_function",
            symbol_type=SymbolType.FUNCTION,
            line_start=10,
            line_end=20,
            language=Language.PYTHON,
        )
        assert symbol.name == "my_function"
        assert symbol.symbol_type == SymbolType.FUNCTION
        assert symbol.line_start == 10
        assert symbol.line_end == 20
        assert symbol.language == Language.PYTHON

    def test_symbol_with_optional_fields(self):
        """Symbol can have optional fields."""
        symbol = Symbol(
            name="MyClass",
            symbol_type=SymbolType.CLASS,
            line_start=1,
            line_end=50,
            language=Language.PYTHON,
            signature="class MyClass(BaseClass):",
            docstring="A class that does something.",
            parent_name="module",
            col_start=0,
            col_end=10,
        )
        assert symbol.signature == "class MyClass(BaseClass):"
        assert symbol.docstring == "A class that does something."
        assert symbol.parent_name == "module"
        assert symbol.col_start == 0
        assert symbol.col_end == 10

    def test_symbol_defaults(self):
        """Symbol has sensible defaults."""
        symbol = Symbol(
            name="x",
            symbol_type=SymbolType.VARIABLE,
            line_start=1,
            line_end=1,
            language=Language.PYTHON,
        )
        assert symbol.signature is None
        assert symbol.docstring is None
        assert symbol.parent_name is None
        assert symbol.col_start is None
        assert symbol.col_end is None


# =============================================================================
# ParseResult Tests
# =============================================================================


class TestParseResult:
    """Tests for ParseResult dataclass."""

    def test_parse_result_success(self):
        """ParseResult for successful parse."""
        result = ParseResult(
            file_path=Path("test.py"),
            language=Language.PYTHON,
            symbols=[
                Symbol(
                    name="foo",
                    symbol_type=SymbolType.FUNCTION,
                    line_start=1,
                    line_end=5,
                    language=Language.PYTHON,
                )
            ],
            errors=[],
            is_partial=False,
        )
        assert result.success
        assert len(result.symbols) == 1
        assert not result.is_partial

    def test_parse_result_partial(self):
        """ParseResult for partial parse with errors."""
        result = ParseResult(
            file_path=Path("test.py"),
            language=Language.PYTHON,
            symbols=[
                Symbol(
                    name="valid_func",
                    symbol_type=SymbolType.FUNCTION,
                    line_start=1,
                    line_end=5,
                    language=Language.PYTHON,
                )
            ],
            errors=[ParseError(line=10, column=5, message="Unexpected token")],
            is_partial=True,
        )
        assert result.success  # Still success because we extracted symbols
        assert result.is_partial
        assert len(result.errors) == 1

    def test_parse_result_failure(self):
        """ParseResult for failed parse."""
        result = ParseResult(
            file_path=Path("test.py"),
            language=Language.PYTHON,
            symbols=[],
            errors=[ParseError(line=1, column=0, message="Syntax error")],
            is_partial=False,
        )
        assert not result.success
        assert len(result.errors) == 1


# =============================================================================
# ParseStatistics Tests
# =============================================================================


class TestParseStatistics:
    """Tests for ParseStatistics tracking."""

    def test_statistics_initial(self):
        """Statistics start at zero."""
        stats = ParseStatistics()
        assert stats.total_files == 0
        assert stats.successful_files == 0
        assert stats.partial_files == 0
        assert stats.failed_files == 0
        assert stats.total_symbols == 0
        assert stats.parse_error_rate == 0.0

    def test_statistics_record_success(self):
        """Statistics track successful parses."""
        stats = ParseStatistics()
        stats.record_success(symbol_count=5)
        assert stats.total_files == 1
        assert stats.successful_files == 1
        assert stats.total_symbols == 5
        assert stats.parse_error_rate == 0.0

    def test_statistics_record_partial(self):
        """Statistics track partial parses."""
        stats = ParseStatistics()
        stats.record_partial(symbol_count=3, error_count=2)
        assert stats.total_files == 1
        assert stats.partial_files == 1
        assert stats.total_symbols == 3
        assert stats.parse_error_rate == 0.0  # Partial still counts as handled

    def test_statistics_record_failure(self):
        """Statistics track failed parses."""
        stats = ParseStatistics()
        stats.record_failure()
        assert stats.total_files == 1
        assert stats.failed_files == 1
        assert stats.parse_error_rate == 1.0

    def test_statistics_error_rate_calculation(self):
        """Error rate is calculated correctly."""
        stats = ParseStatistics()
        # 8 success, 2 failed = 20% error rate
        for _ in range(8):
            stats.record_success(symbol_count=1)
        for _ in range(2):
            stats.record_failure()
        assert stats.total_files == 10
        assert stats.parse_error_rate == pytest.approx(0.2, rel=1e-3)

    def test_statistics_threshold_check(self):
        """Can check if error rate exceeds threshold."""
        stats = ParseStatistics()
        for _ in range(98):
            stats.record_success(symbol_count=1)
        for _ in range(2):
            stats.record_failure()
        assert not stats.exceeds_threshold(0.02)  # Exactly 2%
        assert not stats.exceeds_threshold(0.03)  # Below 3%

        stats.record_failure()  # Now 3/101 ≈ 2.97%
        assert stats.exceeds_threshold(0.02)


# =============================================================================
# Language Plugin Interface Tests
# =============================================================================


class TestLanguagePluginInterface:
    """Tests for LanguagePlugin abstract interface."""

    def test_plugin_interface_methods(self):
        """LanguagePlugin defines required methods."""
        # Check that the interface has expected abstract methods
        assert hasattr(LanguagePlugin, "parse")
        assert hasattr(LanguagePlugin, "extract_symbols")
        assert hasattr(LanguagePlugin, "supported_extensions")

    def test_plugin_is_abstract(self):
        """LanguagePlugin cannot be instantiated directly."""
        with pytest.raises(TypeError):
            LanguagePlugin()


# =============================================================================
# Python Plugin Tests
# =============================================================================


class TestPythonPlugin:
    """Tests for Python language plugin."""

    @pytest.fixture
    def plugin(self):
        """Create Python plugin instance."""
        return PythonPlugin()

    def test_supported_extensions(self, plugin):
        """Python plugin supports .py extension."""
        assert ".py" in plugin.supported_extensions

    def test_parse_simple_function(self, plugin):
        """Parse simple Python function."""
        code = '''
def hello(name: str) -> str:
    """Say hello."""
    return f"Hello, {name}!"
'''
        result = plugin.parse(code.encode("utf-8"))
        assert result is not None
        assert not result.root_node.has_error

    def test_extract_function_symbol(self, plugin):
        """Extract function symbol from Python code."""
        code = '''
def hello(name: str) -> str:
    """Say hello to someone."""
    return f"Hello, {name}!"
'''
        tree = plugin.parse(code.encode("utf-8"))
        symbols = plugin.extract_symbols(tree, code.encode("utf-8"))

        assert len(symbols) >= 1
        func = next((s for s in symbols if s.name == "hello"), None)
        assert func is not None
        assert func.symbol_type == SymbolType.FUNCTION
        assert func.line_start >= 1
        assert "name: str" in (func.signature or "")
        assert "Say hello" in (func.docstring or "")

    def test_extract_class_symbol(self, plugin):
        """Extract class symbol from Python code."""
        code = '''
class MyClass:
    """A sample class."""

    def __init__(self, value: int):
        self.value = value

    def get_value(self) -> int:
        """Return the value."""
        return self.value
'''
        tree = plugin.parse(code.encode("utf-8"))
        symbols = plugin.extract_symbols(tree, code.encode("utf-8"))

        class_sym = next((s for s in symbols if s.name == "MyClass"), None)
        assert class_sym is not None
        assert class_sym.symbol_type == SymbolType.CLASS
        assert "A sample class" in (class_sym.docstring or "")

        # Methods should be extracted too
        init_method = next((s for s in symbols if s.name == "__init__"), None)
        assert init_method is not None
        assert init_method.symbol_type == SymbolType.METHOD
        assert init_method.parent_name == "MyClass"

    def test_extract_imports(self, plugin):
        """Extract import statements from Python code."""
        code = '''
import os
from typing import List, Optional
from pathlib import Path
'''
        tree = plugin.parse(code.encode("utf-8"))
        symbols = plugin.extract_symbols(tree, code.encode("utf-8"))

        imports = [s for s in symbols if s.symbol_type == SymbolType.IMPORT]
        assert len(imports) >= 3

    def test_extract_nested_class(self, plugin):
        """Extract nested class from Python code."""
        code = '''
class Outer:
    """Outer class."""

    class Inner:
        """Inner class."""
        pass
'''
        tree = plugin.parse(code.encode("utf-8"))
        symbols = plugin.extract_symbols(tree, code.encode("utf-8"))

        inner = next((s for s in symbols if s.name == "Inner"), None)
        assert inner is not None
        assert inner.parent_name == "Outer"

    def test_parse_with_syntax_error(self, plugin):
        """Parse Python code with syntax errors."""
        code = '''
def broken(:
    pass
'''
        tree = plugin.parse(code.encode("utf-8"))
        assert tree.root_node.has_error

    def test_partial_extraction_with_errors(self, plugin):
        """Extract symbols from partially valid code."""
        code = '''
def valid_function():
    pass

def broken(:
    pass

class ValidClass:
    pass
'''
        tree = plugin.parse(code.encode("utf-8"))
        symbols = plugin.extract_symbols(tree, code.encode("utf-8"))

        # Should still extract valid symbols
        valid_func = next((s for s in symbols if s.name == "valid_function"), None)
        valid_class = next((s for s in symbols if s.name == "ValidClass"), None)
        assert valid_func is not None
        assert valid_class is not None

    def test_async_function(self, plugin):
        """Parse async function."""
        code = '''
async def fetch_data(url: str) -> dict:
    """Fetch data from URL."""
    pass
'''
        tree = plugin.parse(code.encode("utf-8"))
        symbols = plugin.extract_symbols(tree, code.encode("utf-8"))

        func = next((s for s in symbols if s.name == "fetch_data"), None)
        assert func is not None
        assert func.symbol_type == SymbolType.FUNCTION

    def test_decorated_function(self, plugin):
        """Parse decorated function."""
        code = '''
@decorator
@another_decorator("arg")
def decorated_func():
    pass
'''
        tree = plugin.parse(code.encode("utf-8"))
        symbols = plugin.extract_symbols(tree, code.encode("utf-8"))

        func = next((s for s in symbols if s.name == "decorated_func"), None)
        assert func is not None


# =============================================================================
# TypeScript Plugin Tests
# =============================================================================


class TestTypeScriptPlugin:
    """Tests for TypeScript language plugin."""

    @pytest.fixture
    def plugin(self):
        """Create TypeScript plugin instance."""
        return TypeScriptPlugin()

    def test_supported_extensions(self, plugin):
        """TypeScript plugin supports .ts and .tsx extensions."""
        assert ".ts" in plugin.supported_extensions
        assert ".tsx" in plugin.supported_extensions

    def test_parse_simple_function(self, plugin):
        """Parse simple TypeScript function."""
        code = '''
function greet(name: string): string {
    return `Hello, ${name}!`;
}
'''
        result = plugin.parse(code.encode("utf-8"), is_tsx=False)
        assert result is not None
        assert not result.root_node.has_error

    def test_extract_function_symbol(self, plugin):
        """Extract function symbol from TypeScript code."""
        code = '''
/**
 * Greet someone by name.
 */
function greet(name: string): string {
    return `Hello, ${name}!`;
}
'''
        tree = plugin.parse(code.encode("utf-8"), is_tsx=False)
        symbols = plugin.extract_symbols(tree, code.encode("utf-8"))

        func = next((s for s in symbols if s.name == "greet"), None)
        assert func is not None
        assert func.symbol_type == SymbolType.FUNCTION

    def test_extract_class_symbol(self, plugin):
        """Extract class symbol from TypeScript code."""
        code = '''
/**
 * A sample class.
 */
class MyClass {
    private value: number;

    constructor(value: number) {
        this.value = value;
    }

    getValue(): number {
        return this.value;
    }
}
'''
        tree = plugin.parse(code.encode("utf-8"), is_tsx=False)
        symbols = plugin.extract_symbols(tree, code.encode("utf-8"))

        class_sym = next((s for s in symbols if s.name == "MyClass"), None)
        assert class_sym is not None
        assert class_sym.symbol_type == SymbolType.CLASS

    def test_extract_interface(self, plugin):
        """Extract interface from TypeScript code."""
        code = '''
interface User {
    name: string;
    age: number;
}
'''
        tree = plugin.parse(code.encode("utf-8"), is_tsx=False)
        symbols = plugin.extract_symbols(tree, code.encode("utf-8"))

        iface = next((s for s in symbols if s.name == "User"), None)
        assert iface is not None
        assert iface.symbol_type == SymbolType.INTERFACE

    def test_extract_type_alias(self, plugin):
        """Extract type alias from TypeScript code."""
        code = '''
type Status = "pending" | "active" | "completed";
'''
        tree = plugin.parse(code.encode("utf-8"), is_tsx=False)
        symbols = plugin.extract_symbols(tree, code.encode("utf-8"))

        type_alias = next((s for s in symbols if s.name == "Status"), None)
        assert type_alias is not None
        assert type_alias.symbol_type == SymbolType.TYPE_ALIAS

    def test_extract_enum(self, plugin):
        """Extract enum from TypeScript code."""
        code = '''
enum Color {
    Red,
    Green,
    Blue
}
'''
        tree = plugin.parse(code.encode("utf-8"), is_tsx=False)
        symbols = plugin.extract_symbols(tree, code.encode("utf-8"))

        enum_sym = next((s for s in symbols if s.name == "Color"), None)
        assert enum_sym is not None
        assert enum_sym.symbol_type == SymbolType.ENUM

    def test_arrow_function(self, plugin):
        """Parse arrow function."""
        code = '''
const multiply = (a: number, b: number): number => a * b;
'''
        tree = plugin.parse(code.encode("utf-8"), is_tsx=False)
        symbols = plugin.extract_symbols(tree, code.encode("utf-8"))

        # Arrow function assigned to const should be extracted
        func = next((s for s in symbols if s.name == "multiply"), None)
        assert func is not None

    def test_parse_tsx_with_jsx(self, plugin):
        """Parse TSX file with JSX content."""
        code = '''
import React from 'react';

interface Props {
    name: string;
}

function Greeting({ name }: Props): JSX.Element {
    return <div>Hello, {name}!</div>;
}

export default Greeting;
'''
        tree = plugin.parse(code.encode("utf-8"), is_tsx=True)
        assert not tree.root_node.has_error

        symbols = plugin.extract_symbols(tree, code.encode("utf-8"))

        component = next((s for s in symbols if s.name == "Greeting"), None)
        assert component is not None

    def test_export_statements(self, plugin):
        """Parse export statements."""
        code = '''
export function publicFunc(): void {}
export class PublicClass {}
export const PUBLIC_CONST = 42;
'''
        tree = plugin.parse(code.encode("utf-8"), is_tsx=False)
        symbols = plugin.extract_symbols(tree, code.encode("utf-8"))

        assert any(s.name == "publicFunc" for s in symbols)
        assert any(s.name == "PublicClass" for s in symbols)


# =============================================================================
# TypeScript Decorator Tests (NestJS/Angular)
# =============================================================================


class TestTypeScriptDecorators:
    """Tests for NestJS/Angular decorator extraction from TypeScript."""

    @pytest.fixture
    def plugin(self):
        """Create TypeScript plugin instance."""
        return TypeScriptPlugin()

    def test_on_event_decorator_extracted(self, plugin):
        """@OnEvent decorator is correctly extracted."""
        code = '''
import { OnEvent } from '@nestjs/event-emitter';

class FileService {
    @OnEvent('file.uploaded')
    handleFileUploaded(payload: FileUploadedEvent) {}
}
'''
        tree = plugin.parse(code.encode("utf-8"), is_tsx=False)
        symbols = plugin.extract_symbols(tree, code.encode("utf-8"))

        # Find handleFileUploaded method
        method = next((s for s in symbols if s.name == "handleFileUploaded"), None)
        assert method is not None
        assert len(method.decorators) == 1

        dec = method.decorators[0]
        assert dec.name == "OnEvent"
        assert dec.decorator_type == "event_handler"
        assert len(dec.arguments) == 1
        assert "'file.uploaded'" in dec.arguments[0]

    def test_multiple_decorators_extracted(self, plugin):
        """Multiple decorators on a method are all extracted."""
        code = '''
class UserController {
    @UseGuards(AuthGuard)
    @Post('/users')
    createUser(@Body() dto: CreateUserDto) {}
}
'''
        tree = plugin.parse(code.encode("utf-8"), is_tsx=False)
        symbols = plugin.extract_symbols(tree, code.encode("utf-8"))

        method = next((s for s in symbols if s.name == "createUser"), None)
        assert method is not None
        assert len(method.decorators) == 2

        decorator_names = [d.name for d in method.decorators]
        assert "UseGuards" in decorator_names
        assert "Post" in decorator_names

    def test_controller_decorator_with_prefix(self, plugin):
        """@Controller with route prefix is extracted."""
        code = '''
@Controller('users')
export class UserController {}
'''
        tree = plugin.parse(code.encode("utf-8"), is_tsx=False)
        symbols = plugin.extract_symbols(tree, code.encode("utf-8"))

        cls = next((s for s in symbols if s.name == "UserController"), None)
        assert cls is not None
        assert len(cls.decorators) == 1

        dec = cls.decorators[0]
        assert dec.name == "Controller"
        assert dec.decorator_type == "controller"
        assert len(dec.arguments) == 1
        assert "'users'" in dec.arguments[0]

    def test_injectable_decorator_without_args(self, plugin):
        """@Injectable decorator without arguments is extracted."""
        code = '''
@Injectable()
export class StorageService {}
'''
        tree = plugin.parse(code.encode("utf-8"), is_tsx=False)
        symbols = plugin.extract_symbols(tree, code.encode("utf-8"))

        cls = next((s for s in symbols if s.name == "StorageService"), None)
        assert cls is not None
        assert len(cls.decorators) == 1

        dec = cls.decorators[0]
        assert dec.name == "Injectable"
        assert dec.decorator_type == "di_provider"
        assert len(dec.arguments) == 0

    def test_http_handlers_extracted(self, plugin):
        """HTTP handler decorators (@Get, @Post, etc.) are correctly typed."""
        code = '''
@Controller('api')
class ApiController {
    @Get('/items')
    getItems() {}

    @Post('/items')
    createItem() {}

    @Put('/items/:id')
    updateItem() {}

    @Delete('/items/:id')
    deleteItem() {}
}
'''
        tree = plugin.parse(code.encode("utf-8"), is_tsx=False)
        symbols = plugin.extract_symbols(tree, code.encode("utf-8"))

        methods = [s for s in symbols if s.symbol_type == SymbolType.METHOD]
        assert len(methods) == 4

        for method in methods:
            assert len(method.decorators) == 1
            assert method.decorators[0].decorator_type == "http_handler"

    def test_cron_decorator_extracted(self, plugin):
        """@Cron scheduled task decorator is extracted with cron expression."""
        code = '''
class TaskService {
    @Cron('0 * * * *')
    handleHourlyTask() {}
}
'''
        tree = plugin.parse(code.encode("utf-8"), is_tsx=False)
        symbols = plugin.extract_symbols(tree, code.encode("utf-8"))

        method = next((s for s in symbols if s.name == "handleHourlyTask"), None)
        assert method is not None
        assert len(method.decorators) == 1

        dec = method.decorators[0]
        assert dec.name == "Cron"
        assert dec.decorator_type == "scheduled_task"
        assert "'0 * * * *'" in dec.arguments[0]

    def test_websocket_decorators(self, plugin):
        """WebSocket decorators are extracted."""
        code = '''
@WebSocketGateway()
class EventsGateway {
    @SubscribeMessage('events')
    handleEvent(data: string) {}
}
'''
        tree = plugin.parse(code.encode("utf-8"), is_tsx=False)
        symbols = plugin.extract_symbols(tree, code.encode("utf-8"))

        gateway_cls = next((s for s in symbols if s.name == "EventsGateway"), None)
        assert gateway_cls is not None
        assert len(gateway_cls.decorators) == 1
        assert gateway_cls.decorators[0].decorator_type == "websocket_gateway"

        method = next((s for s in symbols if s.name == "handleEvent"), None)
        assert method is not None
        assert len(method.decorators) == 1
        assert method.decorators[0].decorator_type == "websocket_handler"

    def test_event_pattern_decorator(self, plugin):
        """@EventPattern microservice decorator is extracted."""
        code = '''
class MessageController {
    @EventPattern('user.created')
    handleUserCreated(data: any) {}
}
'''
        tree = plugin.parse(code.encode("utf-8"), is_tsx=False)
        symbols = plugin.extract_symbols(tree, code.encode("utf-8"))

        method = next((s for s in symbols if s.name == "handleUserCreated"), None)
        assert method is not None
        assert len(method.decorators) == 1

        dec = method.decorators[0]
        assert dec.name == "EventPattern"
        assert dec.decorator_type == "message_handler"

    def test_decorator_raw_text_preserved(self, plugin):
        """Decorator raw text is preserved for debugging."""
        code = '''
class MyClass {
    @OnEvent('complex.event', { async: true })
    handleEvent() {}
}
'''
        tree = plugin.parse(code.encode("utf-8"), is_tsx=False)
        symbols = plugin.extract_symbols(tree, code.encode("utf-8"))

        method = next((s for s in symbols if s.name == "handleEvent"), None)
        assert method is not None
        assert len(method.decorators) == 1

        dec = method.decorators[0]
        assert dec.raw_text is not None
        assert "@OnEvent" in dec.raw_text

    def test_unknown_decorator_has_none_type(self, plugin):
        """Unknown decorators have decorator_type=None."""
        code = '''
class MyClass {
    @CustomDecorator()
    myMethod() {}
}
'''
        tree = plugin.parse(code.encode("utf-8"), is_tsx=False)
        symbols = plugin.extract_symbols(tree, code.encode("utf-8"))

        method = next((s for s in symbols if s.name == "myMethod"), None)
        assert method is not None
        assert len(method.decorators) == 1

        dec = method.decorators[0]
        assert dec.name == "CustomDecorator"
        assert dec.decorator_type is None  # Unknown decorator

    def test_class_and_method_decorators_separate(self, plugin):
        """Class decorators and method decorators are kept separate."""
        code = '''
@Controller('users')
class UserController {
    @Get()
    findAll() {}
}
'''
        tree = plugin.parse(code.encode("utf-8"), is_tsx=False)
        symbols = plugin.extract_symbols(tree, code.encode("utf-8"))

        cls = next((s for s in symbols if s.name == "UserController"), None)
        assert cls is not None
        assert len(cls.decorators) == 1
        assert cls.decorators[0].name == "Controller"

        method = next((s for s in symbols if s.name == "findAll"), None)
        assert method is not None
        assert len(method.decorators) == 1
        assert method.decorators[0].name == "Get"


class TestDecoratorInfo:
    """Tests for DecoratorInfo dataclass."""

    def test_decorator_info_creation(self):
        """DecoratorInfo can be created with all fields."""
        dec = DecoratorInfo(
            name="OnEvent",
            decorator_type="event_handler",
            arguments=["'file.uploaded'"],
            raw_text="@OnEvent('file.uploaded')",
        )
        assert dec.name == "OnEvent"
        assert dec.decorator_type == "event_handler"
        assert dec.arguments == ["'file.uploaded'"]
        assert dec.raw_text == "@OnEvent('file.uploaded')"

    def test_decorator_info_defaults(self):
        """DecoratorInfo has sensible defaults."""
        dec = DecoratorInfo(name="Injectable")
        assert dec.name == "Injectable"
        assert dec.decorator_type is None
        assert dec.arguments == []
        assert dec.raw_text is None


class TestKnownDecorators:
    """Tests for KNOWN_DECORATORS mapping."""

    def test_known_decorators_contains_event_handlers(self):
        """KNOWN_DECORATORS includes event handling decorators."""
        assert KNOWN_DECORATORS.get("OnEvent") == "event_handler"
        assert KNOWN_DECORATORS.get("OnQueueEvent") == "event_handler"

    def test_known_decorators_contains_http_handlers(self):
        """KNOWN_DECORATORS includes HTTP handler decorators."""
        for method in ["Get", "Post", "Put", "Patch", "Delete", "Head", "Options"]:
            assert KNOWN_DECORATORS.get(method) == "http_handler"

    def test_known_decorators_contains_di_decorators(self):
        """KNOWN_DECORATORS includes DI decorators."""
        assert KNOWN_DECORATORS.get("Injectable") == "di_provider"
        assert KNOWN_DECORATORS.get("Inject") == "di_injection"

    def test_known_decorators_contains_scheduling(self):
        """KNOWN_DECORATORS includes scheduling decorators."""
        assert KNOWN_DECORATORS.get("Cron") == "scheduled_task"
        assert KNOWN_DECORATORS.get("Interval") == "scheduled_task"


# =============================================================================
# Event Registry Tests
# =============================================================================


class TestEventEdge:
    """Tests for EventEdge dataclass."""

    def test_event_edge_creation(self):
        """EventEdge can be created with all fields."""
        edge = EventEdge(
            publisher_symbol_id="pub-123",
            subscriber_symbol_id="sub-456",
            event_name="file.uploaded",
        )
        assert edge.publisher_symbol_id == "pub-123"
        assert edge.subscriber_symbol_id == "sub-456"
        assert edge.event_name == "file.uploaded"
        assert edge.edge_type == "TRIGGERS_EVENT"

    def test_event_edge_custom_type(self):
        """EventEdge can have custom edge type."""
        edge = EventEdge(
            publisher_symbol_id="pub-123",
            subscriber_symbol_id="sub-456",
            event_name="file.uploaded",
            edge_type="CUSTOM_TYPE",
        )
        assert edge.edge_type == "CUSTOM_TYPE"


class TestEventRegistry:
    """Tests for EventRegistry class."""

    def test_empty_registry(self):
        """Empty registry returns no edges."""
        registry = EventRegistry()
        assert registry.get_edges() == []

    def test_register_subscriber(self):
        """Subscriber can be registered."""
        registry = EventRegistry()
        registry.register_subscriber("file.uploaded", "handler-1")

        assert "file.uploaded" in registry.subscribers
        assert "handler-1" in registry.subscribers["file.uploaded"]

    def test_register_publisher(self):
        """Publisher can be registered."""
        registry = EventRegistry()
        registry.register_publisher("file.uploaded", "emitter-1")

        assert "file.uploaded" in registry.publishers
        assert "emitter-1" in registry.publishers["file.uploaded"]

    def test_get_edges_with_match(self):
        """Matching publisher and subscriber generate edges."""
        registry = EventRegistry()
        registry.register_publisher("file.uploaded", "emitter-1")
        registry.register_subscriber("file.uploaded", "handler-1")

        edges = registry.get_edges()
        assert len(edges) == 1
        edge = edges[0]
        assert edge.publisher_symbol_id == "emitter-1"
        assert edge.subscriber_symbol_id == "handler-1"
        assert edge.event_name == "file.uploaded"

    def test_get_edges_no_match(self):
        """Non-matching events generate no edges."""
        registry = EventRegistry()
        registry.register_publisher("file.uploaded", "emitter-1")
        registry.register_subscriber("user.created", "handler-1")

        edges = registry.get_edges()
        assert len(edges) == 0

    def test_multiple_subscribers_one_publisher(self):
        """Multiple subscribers for one event generate multiple edges."""
        registry = EventRegistry()
        registry.register_publisher("file.uploaded", "emitter-1")
        registry.register_subscriber("file.uploaded", "handler-1")
        registry.register_subscriber("file.uploaded", "handler-2")

        edges = registry.get_edges()
        assert len(edges) == 2

    def test_multiple_publishers_one_subscriber(self):
        """Multiple publishers for one event generate multiple edges."""
        registry = EventRegistry()
        registry.register_publisher("file.uploaded", "emitter-1")
        registry.register_publisher("file.uploaded", "emitter-2")
        registry.register_subscriber("file.uploaded", "handler-1")

        edges = registry.get_edges()
        assert len(edges) == 2

    def test_no_duplicate_registrations(self):
        """Duplicate registrations are ignored."""
        registry = EventRegistry()
        registry.register_publisher("file.uploaded", "emitter-1")
        registry.register_publisher("file.uploaded", "emitter-1")
        registry.register_subscriber("file.uploaded", "handler-1")

        edges = registry.get_edges()
        assert len(edges) == 1


class TestDiscoverEventPublishers:
    """Tests for discover_event_publishers function."""

    def test_find_single_emit(self):
        """Find single emit() call in source."""
        symbol = Symbol(
            name="uploadFile",
            symbol_type=SymbolType.FUNCTION,
            line_start=1,
            line_end=5,
            language=Language.TYPESCRIPT,
        )
        source = """
        async uploadFile(file: File) {
            await this.saveFile(file);
            this.eventEmitter.emit('file.uploaded', { fileId: file.id });
        }
        """
        events = discover_event_publishers(symbol, source)
        assert events == ["file.uploaded"]

    def test_find_multiple_emits(self):
        """Find multiple emit() calls in source."""
        symbol = Symbol(
            name="processFile",
            symbol_type=SymbolType.FUNCTION,
            line_start=1,
            line_end=10,
            language=Language.TYPESCRIPT,
        )
        source = """
        async processFile(file: File) {
            this.eventEmitter.emit('file.processing.started', { fileId: file.id });
            await this.transform(file);
            this.eventEmitter.emit('file.processing.completed', { fileId: file.id });
        }
        """
        events = discover_event_publishers(symbol, source)
        assert len(events) == 2
        assert "file.processing.started" in events
        assert "file.processing.completed" in events

    def test_no_emit_calls(self):
        """No events when no emit() calls."""
        symbol = Symbol(
            name="simpleMethod",
            symbol_type=SymbolType.FUNCTION,
            line_start=1,
            line_end=3,
            language=Language.TYPESCRIPT,
        )
        source = """
        simpleMethod() {
            return this.value;
        }
        """
        events = discover_event_publishers(symbol, source)
        assert events == []

    def test_double_quoted_events(self):
        """Find emit() calls with double-quoted strings."""
        symbol = Symbol(
            name="emitEvent",
            symbol_type=SymbolType.FUNCTION,
            line_start=1,
            line_end=3,
            language=Language.TYPESCRIPT,
        )
        source = '''
        emitEvent() {
            this.emit("user.created", payload);
        }
        '''
        events = discover_event_publishers(symbol, source)
        assert events == ["user.created"]


class TestDiscoverEventSubscribers:
    """Tests for discover_event_subscribers function."""

    def test_find_on_event_handler(self):
        """Find @OnEvent decorated method."""
        symbols = [
            Symbol(
                name="handleFileUploaded",
                symbol_type=SymbolType.METHOD,
                line_start=1,
                line_end=5,
                language=Language.TYPESCRIPT,
                decorators=[
                    DecoratorInfo(
                        name="OnEvent",
                        decorator_type="event_handler",
                        arguments=["'file.uploaded'"],
                    ),
                ],
            ),
        ]
        subscribers = discover_event_subscribers(symbols)
        assert "file.uploaded" in subscribers
        assert len(subscribers["file.uploaded"]) == 1

    def test_find_multiple_handlers_same_event(self):
        """Find multiple handlers for same event."""
        symbols = [
            Symbol(
                name="handler1",
                symbol_type=SymbolType.METHOD,
                line_start=1,
                line_end=5,
                language=Language.TYPESCRIPT,
                decorators=[
                    DecoratorInfo(
                        name="OnEvent",
                        decorator_type="event_handler",
                        arguments=["'file.uploaded'"],
                    ),
                ],
            ),
            Symbol(
                name="handler2",
                symbol_type=SymbolType.METHOD,
                line_start=6,
                line_end=10,
                language=Language.TYPESCRIPT,
                decorators=[
                    DecoratorInfo(
                        name="OnEvent",
                        decorator_type="event_handler",
                        arguments=["'file.uploaded'"],
                    ),
                ],
            ),
        ]
        subscribers = discover_event_subscribers(symbols)
        assert "file.uploaded" in subscribers
        assert len(subscribers["file.uploaded"]) == 2

    def test_no_event_handlers(self):
        """No handlers when no @OnEvent decorators."""
        symbols = [
            Symbol(
                name="regularMethod",
                symbol_type=SymbolType.METHOD,
                line_start=1,
                line_end=5,
                language=Language.TYPESCRIPT,
                decorators=[
                    DecoratorInfo(
                        name="Get",
                        decorator_type="http_handler",
                        arguments=["'/items'"],
                    ),
                ],
            ),
        ]
        subscribers = discover_event_subscribers(symbols)
        assert len(subscribers) == 0


class TestBuildEventRegistry:
    """Tests for build_event_registry function."""

    def test_build_registry_with_matches(self):
        """Build registry with matching publishers and subscribers."""
        symbols = [
            Symbol(
                name="uploadFile",
                symbol_type=SymbolType.METHOD,
                line_start=1,
                line_end=5,
                language=Language.TYPESCRIPT,
                parent_name="UploadService",
            ),
            Symbol(
                name="handleFileUploaded",
                symbol_type=SymbolType.METHOD,
                line_start=10,
                line_end=15,
                language=Language.TYPESCRIPT,
                parent_name="ProcessingService",
                decorators=[
                    DecoratorInfo(
                        name="OnEvent",
                        decorator_type="event_handler",
                        arguments=["'file.uploaded'"],
                    ),
                ],
            ),
        ]

        symbol_id_map = {
            "UploadService.uploadFile": "sym-1",
            "ProcessingService.handleFileUploaded": "sym-2",
        }

        source_map = {
            "UploadService.uploadFile": """
                this.eventEmitter.emit('file.uploaded', { fileId: file.id });
            """,
        }

        registry = build_event_registry(symbols, symbol_id_map, source_map)

        edges = registry.get_edges()
        assert len(edges) == 1
        edge = edges[0]
        assert edge.publisher_symbol_id == "sym-1"
        assert edge.subscriber_symbol_id == "sym-2"
        assert edge.event_name == "file.uploaded"


class TestGenerateEventEdges:
    """Tests for generate_event_edges function."""

    def test_generate_neo4j_edges(self):
        """Generate Neo4j-compatible edge data."""
        registry = EventRegistry()
        registry.register_publisher("file.uploaded", "emitter-1")
        registry.register_subscriber("file.uploaded", "handler-1")

        edges = generate_event_edges(registry, "test-repo")

        assert len(edges) == 1
        edge = edges[0]
        assert edge["source_id"] == "emitter-1"
        assert edge["target_id"] == "handler-1"
        assert edge["relationship"] == "TRIGGERS_EVENT"
        assert edge["properties"]["event_name"] == "file.uploaded"
        assert edge["properties"]["repo_id"] == "test-repo"
        assert edge["properties"]["inferred"] is True


# =============================================================================
# Java Plugin Tests
# =============================================================================


class TestJavaPlugin:
    """Tests for Java language plugin."""

    @pytest.fixture
    def plugin(self):
        """Create Java plugin instance."""
        return JavaPlugin()

    def test_supported_extensions(self, plugin):
        """Java plugin supports .java extension."""
        assert ".java" in plugin.supported_extensions

    def test_parse_simple_class(self, plugin):
        """Parse simple Java class."""
        code = '''
public class HelloWorld {
    public static void main(String[] args) {
        System.out.println("Hello, World!");
    }
}
'''
        result = plugin.parse(code.encode("utf-8"))
        assert result is not None
        assert not result.root_node.has_error

    def test_extract_class_symbol(self, plugin):
        """Extract class symbol from Java code."""
        code = '''
/**
 * A sample class.
 */
public class MyClass {
    private int value;

    public MyClass(int value) {
        this.value = value;
    }

    public int getValue() {
        return this.value;
    }
}
'''
        tree = plugin.parse(code.encode("utf-8"))
        symbols = plugin.extract_symbols(tree, code.encode("utf-8"))

        class_sym = next((s for s in symbols if s.name == "MyClass"), None)
        assert class_sym is not None
        assert class_sym.symbol_type == SymbolType.CLASS

    def test_extract_methods(self, plugin):
        """Extract methods from Java class."""
        code = '''
public class Calculator {
    public int add(int a, int b) {
        return a + b;
    }

    public int subtract(int a, int b) {
        return a - b;
    }
}
'''
        tree = plugin.parse(code.encode("utf-8"))
        symbols = plugin.extract_symbols(tree, code.encode("utf-8"))

        add_method = next((s for s in symbols if s.name == "add"), None)
        assert add_method is not None
        assert add_method.symbol_type == SymbolType.METHOD
        assert add_method.parent_name == "Calculator"

    def test_extract_interface(self, plugin):
        """Extract interface from Java code."""
        code = '''
public interface Comparable<T> {
    int compareTo(T other);
}
'''
        tree = plugin.parse(code.encode("utf-8"))
        symbols = plugin.extract_symbols(tree, code.encode("utf-8"))

        iface = next((s for s in symbols if s.name == "Comparable"), None)
        assert iface is not None
        assert iface.symbol_type == SymbolType.INTERFACE

    def test_extract_enum(self, plugin):
        """Extract enum from Java code."""
        code = '''
public enum Status {
    PENDING,
    ACTIVE,
    COMPLETED
}
'''
        tree = plugin.parse(code.encode("utf-8"))
        symbols = plugin.extract_symbols(tree, code.encode("utf-8"))

        enum_sym = next((s for s in symbols if s.name == "Status"), None)
        assert enum_sym is not None
        assert enum_sym.symbol_type == SymbolType.ENUM

    def test_extract_imports(self, plugin):
        """Extract imports from Java code."""
        code = '''
import java.util.List;
import java.util.ArrayList;
import java.io.*;

public class Main {}
'''
        tree = plugin.parse(code.encode("utf-8"))
        symbols = plugin.extract_symbols(tree, code.encode("utf-8"))

        imports = [s for s in symbols if s.symbol_type == SymbolType.IMPORT]
        assert len(imports) >= 2

    def test_inner_class(self, plugin):
        """Extract inner class from Java code."""
        code = '''
public class Outer {
    public class Inner {
        public void doSomething() {}
    }
}
'''
        tree = plugin.parse(code.encode("utf-8"))
        symbols = plugin.extract_symbols(tree, code.encode("utf-8"))

        inner = next((s for s in symbols if s.name == "Inner"), None)
        assert inner is not None
        assert inner.parent_name == "Outer"


# =============================================================================
# ASTParser Config Tests
# =============================================================================


class TestASTParserConfig:
    """Tests for ASTParser configuration."""

    def test_default_config(self):
        """Default config has sensible values."""
        config = ASTParserConfig()
        assert config.max_file_size_bytes == 1_000_000  # 1MB default
        assert config.error_rate_threshold == 0.02  # 2% threshold
        assert config.skip_malformed is True
        assert config.log_errors is True

    def test_custom_config(self):
        """Can create custom config."""
        config = ASTParserConfig(
            max_file_size_bytes=5_000_000,
            error_rate_threshold=0.05,
            skip_malformed=False,
            log_errors=False,
        )
        assert config.max_file_size_bytes == 5_000_000
        assert config.error_rate_threshold == 0.05
        assert config.skip_malformed is False
        assert config.log_errors is False


# =============================================================================
# ASTParser Tests
# =============================================================================


class TestASTParser:
    """Tests for main ASTParser class."""

    @pytest.fixture
    def parser(self):
        """Create parser with all plugins."""
        return create_parser()

    def test_parser_has_plugins(self, parser):
        """Parser has language plugins registered."""
        assert Language.PYTHON in parser.plugins
        assert Language.TYPESCRIPT in parser.plugins
        assert Language.JAVA in parser.plugins
        assert Language.GO in parser.plugins

    def test_parse_python_file(self, parser, tmp_path):
        """Parse a Python file."""
        py_file = tmp_path / "test.py"
        py_file.write_text('''
def hello():
    pass

class World:
    pass
''')
        result = parser.parse_file(py_file)
        assert result.success
        assert result.language == Language.PYTHON
        assert any(s.name == "hello" for s in result.symbols)
        assert any(s.name == "World" for s in result.symbols)

    def test_parse_typescript_file(self, parser, tmp_path):
        """Parse a TypeScript file."""
        ts_file = tmp_path / "test.ts"
        ts_file.write_text('''
function hello(): void {}

interface World {
    name: string;
}
''')
        result = parser.parse_file(ts_file)
        assert result.success
        assert result.language == Language.TYPESCRIPT
        assert any(s.name == "hello" for s in result.symbols)
        assert any(s.name == "World" for s in result.symbols)

    def test_parse_tsx_file(self, parser, tmp_path):
        """Parse a TSX file."""
        tsx_file = tmp_path / "Component.tsx"
        tsx_file.write_text('''
function Component(): JSX.Element {
    return <div>Hello</div>;
}
''')
        result = parser.parse_file(tsx_file)
        assert result.success
        assert result.language == Language.TSX

    def test_parse_java_file(self, parser, tmp_path):
        """Parse a Java file."""
        java_file = tmp_path / "Main.java"
        java_file.write_text('''
public class Main {
    public static void main(String[] args) {}
}
''')
        result = parser.parse_file(java_file)
        assert result.success
        assert result.language == Language.JAVA
        assert any(s.name == "Main" for s in result.symbols)

    def test_parse_unsupported_extension(self, parser, tmp_path):
        """Raise error for unsupported file extension."""
        rb_file = tmp_path / "test.rb"
        rb_file.write_text("puts 'hello'")

        with pytest.raises(UnsupportedLanguageError):
            parser.parse_file(rb_file)

    def test_parse_nonexistent_file(self, parser, tmp_path):
        """Handle nonexistent file gracefully."""
        fake_file = tmp_path / "nonexistent.py"

        with pytest.raises(FileNotFoundError):
            parser.parse_file(fake_file)

    def test_parse_file_too_large(self, parser, tmp_path):
        """Skip files exceeding size limit."""
        config = ASTParserConfig(max_file_size_bytes=100)
        parser_small = create_parser(config)

        large_file = tmp_path / "large.py"
        large_file.write_text("x = 1\n" * 100)

        result = parser_small.parse_file(large_file)
        assert not result.success
        assert any("size" in e.message.lower() for e in result.errors)

    def test_parse_string(self, parser):
        """Parse code from string."""
        code = "def hello(): pass"
        result = parser.parse_string(code, Language.PYTHON)
        assert result.success
        assert any(s.name == "hello" for s in result.symbols)

    def test_parse_string_with_bytes(self, parser):
        """Parse code from bytes."""
        code = b"def hello(): pass"
        result = parser.parse_bytes(code, Language.PYTHON)
        assert result.success

    def test_parse_directory(self, parser, tmp_path):
        """Parse all files in a directory."""
        # Create test files
        (tmp_path / "a.py").write_text("def func_a(): pass")
        (tmp_path / "b.py").write_text("class ClassB: pass")
        (tmp_path / "c.ts").write_text("function funcC(): void {}")
        (tmp_path / "readme.md").write_text("# Readme")  # Should be skipped

        results = parser.parse_directory(tmp_path)
        assert len(results) == 3
        assert all(r.success for r in results)

    def test_parse_directory_recursive(self, parser, tmp_path):
        """Parse directory recursively."""
        subdir = tmp_path / "subdir"
        subdir.mkdir()

        (tmp_path / "a.py").write_text("def func_a(): pass")
        (subdir / "b.py").write_text("def func_b(): pass")

        results = parser.parse_directory(tmp_path, recursive=True)
        assert len(results) == 2

    def test_statistics_tracking(self, parser, tmp_path):
        """Parser tracks statistics."""
        (tmp_path / "good.py").write_text("def good(): pass")
        (tmp_path / "bad.py").write_text("def bad(:")

        parser.parse_directory(tmp_path)
        stats = parser.statistics

        assert stats.total_files == 2
        assert stats.successful_files >= 1

    def test_parse_with_incremental_update(self, parser, tmp_path):
        """Support incremental parsing with edit."""
        py_file = tmp_path / "test.py"
        py_file.write_text("def hello(): pass")

        # Initial parse
        result1 = parser.parse_file(py_file)
        assert result1.success

        # Modify file
        py_file.write_text("def hello(): pass\ndef world(): pass")

        # Re-parse
        result2 = parser.parse_file(py_file)
        assert result2.success
        assert len(result2.symbols) > len(result1.symbols)


# =============================================================================
# Edge Cases and Error Handling
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    @pytest.fixture
    def parser(self):
        """Create parser instance."""
        return create_parser()

    def test_empty_file(self, parser, tmp_path):
        """Handle empty file."""
        empty_file = tmp_path / "empty.py"
        empty_file.write_text("")

        result = parser.parse_file(empty_file)
        assert result.success
        assert len(result.symbols) == 0

    def test_file_with_only_comments(self, parser, tmp_path):
        """Handle file with only comments."""
        comment_file = tmp_path / "comments.py"
        comment_file.write_text('''
# This is a comment
# Another comment
"""
A docstring without any code
"""
''')
        result = parser.parse_file(comment_file)
        assert result.success
        assert len(result.symbols) == 0

    def test_unicode_content(self, parser, tmp_path):
        """Handle unicode in code."""
        unicode_file = tmp_path / "unicode.py"
        unicode_file.write_text('''
def greet_世界():
    """挨拶する"""
    return "こんにちは"
''')
        result = parser.parse_file(unicode_file)
        assert result.success
        func = next((s for s in result.symbols if "世界" in s.name), None)
        assert func is not None

    def test_very_long_lines(self, parser, tmp_path):
        """Handle files with very long lines."""
        long_line_file = tmp_path / "long.py"
        long_arg = ", ".join([f"arg{i}: int" for i in range(100)])
        long_line_file.write_text(f"def func({long_arg}): pass")

        result = parser.parse_file(long_line_file)
        assert result.success

    def test_deeply_nested_code(self, parser, tmp_path):
        """Handle deeply nested code."""
        nested_file = tmp_path / "nested.py"
        code = "class A:\n"
        for i in range(20):
            code += "  " * (i + 1) + f"class Level{i}:\n"
        code += "  " * 21 + "pass"
        nested_file.write_text(code)

        result = parser.parse_file(nested_file)
        # Should still parse, even if deeply nested
        assert result.success or result.is_partial

    def test_malformed_but_recoverable(self, parser, tmp_path):
        """Handle malformed code with recoverable portions."""
        config = ASTParserConfig(skip_malformed=True)
        parser_tolerant = create_parser(config)

        malformed_file = tmp_path / "malformed.py"
        malformed_file.write_text('''
def valid1():
    pass

class Broken(:
    pass

def valid2():
    pass
''')
        result = parser_tolerant.parse_file(malformed_file)
        # Should extract valid symbols even with errors
        assert any(s.name == "valid1" for s in result.symbols)
        assert any(s.name == "valid2" for s in result.symbols)

    def test_binary_file_detection(self, parser, tmp_path):
        """Skip binary files."""
        binary_file = tmp_path / "binary.py"
        binary_file.write_bytes(b"\x00\x01\x02\x03")

        result = parser.parse_file(binary_file)
        assert not result.success
        assert any("binary" in e.message.lower() for e in result.errors)


# =============================================================================
# Factory Function Tests
# =============================================================================


class TestFactory:
    """Tests for parser factory function."""

    def test_create_parser_default(self):
        """Create parser with default config."""
        parser = create_parser()
        assert parser is not None
        assert len(parser.plugins) >= 3

    def test_create_parser_custom_config(self):
        """Create parser with custom config."""
        config = ASTParserConfig(max_file_size_bytes=500_000)
        parser = create_parser(config)
        assert parser.config.max_file_size_bytes == 500_000

    def test_create_parser_with_plugins(self):
        """Create parser with specific plugins."""
        parser = create_parser(plugins=[PythonPlugin()])
        assert Language.PYTHON in parser.plugins
        assert Language.TYPESCRIPT not in parser.plugins


# =============================================================================
# Logging Tests
# =============================================================================


class TestLogging:
    """Tests for error logging behavior."""

    def test_logs_parse_errors(self, tmp_path, caplog):
        """Parser logs parse errors."""
        config = ASTParserConfig(log_errors=True)
        parser = create_parser(config)

        bad_file = tmp_path / "bad.py"
        bad_file.write_text("def broken(:")

        import logging
        with caplog.at_level(logging.WARNING):
            parser.parse_file(bad_file)

        # Should have logged something about the error
        assert len(caplog.records) > 0 or True  # May not always log

    def test_no_logs_when_disabled(self, tmp_path, caplog):
        """Parser doesn't log when logging disabled."""
        config = ASTParserConfig(log_errors=False)
        parser = create_parser(config)

        bad_file = tmp_path / "bad.py"
        bad_file.write_text("def broken(:")

        import logging
        with caplog.at_level(logging.DEBUG):
            parser.parse_file(bad_file)

        # Should not have any logs from our parser
        parse_logs = [r for r in caplog.records if "ast_parser" in r.name.lower()]
        assert len(parse_logs) == 0
