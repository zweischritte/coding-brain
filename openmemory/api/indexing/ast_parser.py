"""AST Parser with Tree-sitter for Python, TypeScript, and Java.

This module provides:
- Language enum for supported languages
- Symbol extraction from AST (functions, classes, methods, imports)
- Parse error tracking and statistics
- Language plugin interface for extensibility
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional, Union

import tree_sitter_python as ts_python
import tree_sitter_typescript as ts_typescript
import tree_sitter_java as ts_java
from tree_sitter import Language as TSLanguage, Parser, Tree, Node

logger = logging.getLogger(__name__)


# =============================================================================
# Exceptions
# =============================================================================


class UnsupportedLanguageError(Exception):
    """Raised when attempting to parse an unsupported language."""

    pass


class ParseFailedError(Exception):
    """Raised when parsing fails completely."""

    pass


# =============================================================================
# Enums
# =============================================================================


class Language(Enum):
    """Supported programming languages."""

    PYTHON = "python"
    TYPESCRIPT = "typescript"
    TSX = "tsx"
    JAVA = "java"

    @classmethod
    def from_extension(cls, ext: str) -> Optional["Language"]:
        """Get language from file extension."""
        ext_map = {
            ".py": cls.PYTHON,
            ".ts": cls.TYPESCRIPT,
            ".tsx": cls.TSX,
            ".java": cls.JAVA,
        }
        return ext_map.get(ext.lower())

    @classmethod
    def from_path(cls, path: Path) -> Optional["Language"]:
        """Get language from file path."""
        return cls.from_extension(path.suffix)


class SymbolType(Enum):
    """Types of symbols that can be extracted."""

    FUNCTION = "function"
    CLASS = "class"
    METHOD = "method"
    IMPORT = "import"
    VARIABLE = "variable"
    INTERFACE = "interface"
    ENUM = "enum"
    TYPE_ALIAS = "type_alias"


# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class DecoratorInfo:
    """Information about a decorator applied to a symbol.

    Used for NestJS/Angular decorator extraction to enable:
    - Event-edge inference (@OnEvent -> emit() connections)
    - HTTP endpoint mapping (@Get, @Post routes)
    - DI graph extraction (@Injectable, @Inject)
    """

    name: str
    decorator_type: Optional[str] = None  # event_handler, http_handler, di_provider, etc.
    arguments: list[str] = field(default_factory=list)
    raw_text: Optional[str] = None


@dataclass
class Symbol:
    """A symbol extracted from source code."""

    name: str
    symbol_type: SymbolType
    line_start: int
    line_end: int
    language: Language
    signature: Optional[str] = None
    docstring: Optional[str] = None
    parent_name: Optional[str] = None
    col_start: Optional[int] = None
    col_end: Optional[int] = None
    decorators: list[DecoratorInfo] = field(default_factory=list)


@dataclass
class ParseError:
    """A parse error encountered during parsing."""

    line: int
    column: int
    message: str


@dataclass
class ParseResult:
    """Result of parsing a file."""

    file_path: Optional[Path]
    language: Language
    symbols: list[Symbol]
    errors: list[ParseError]
    is_partial: bool = False

    @property
    def success(self) -> bool:
        """True if parsing produced symbols or had no errors."""
        return len(self.symbols) > 0 or len(self.errors) == 0


@dataclass
class ParseStatistics:
    """Statistics for parse operations."""

    total_files: int = 0
    successful_files: int = 0
    partial_files: int = 0
    failed_files: int = 0
    total_symbols: int = 0

    @property
    def parse_error_rate(self) -> float:
        """Calculate parse error rate."""
        if self.total_files == 0:
            return 0.0
        return self.failed_files / self.total_files

    def record_success(self, symbol_count: int) -> None:
        """Record a successful parse."""
        self.total_files += 1
        self.successful_files += 1
        self.total_symbols += symbol_count

    def record_partial(self, symbol_count: int, error_count: int) -> None:
        """Record a partial parse with some errors."""
        self.total_files += 1
        self.partial_files += 1
        self.total_symbols += symbol_count

    def record_failure(self) -> None:
        """Record a failed parse."""
        self.total_files += 1
        self.failed_files += 1

    def exceeds_threshold(self, threshold: float) -> bool:
        """Check if error rate exceeds threshold."""
        return self.parse_error_rate > threshold


@dataclass
class ASTParserConfig:
    """Configuration for AST parser."""

    max_file_size_bytes: int = 1_000_000  # 1MB default
    error_rate_threshold: float = 0.02  # 2% threshold
    skip_malformed: bool = True
    log_errors: bool = True


# =============================================================================
# Language Plugin Interface
# =============================================================================


class LanguagePlugin(ABC):
    """Abstract base class for language plugins."""

    @property
    @abstractmethod
    def supported_extensions(self) -> list[str]:
        """Return list of supported file extensions."""
        pass

    @abstractmethod
    def parse(self, source: bytes, **kwargs) -> Tree:
        """Parse source code into AST."""
        pass

    @abstractmethod
    def extract_symbols(self, tree: Tree, source: bytes) -> list[Symbol]:
        """Extract symbols from AST."""
        pass


# =============================================================================
# Python Plugin
# =============================================================================


class PythonPlugin(LanguagePlugin):
    """Language plugin for Python."""

    def __init__(self):
        self._parser = Parser(TSLanguage(ts_python.language()))

    @property
    def supported_extensions(self) -> list[str]:
        return [".py"]

    def parse(self, source: bytes, **kwargs) -> Tree:
        """Parse Python source code."""
        return self._parser.parse(source)

    def extract_symbols(self, tree: Tree, source: bytes) -> list[Symbol]:
        """Extract symbols from Python AST."""
        symbols = []
        self._extract_from_node(tree.root_node, source, symbols, parent_name=None)
        return symbols

    def _extract_from_node(
        self,
        node: Node,
        source: bytes,
        symbols: list[Symbol],
        parent_name: Optional[str],
    ) -> None:
        """Recursively extract symbols from AST node."""
        if node.type == "function_definition":
            self._extract_function(node, source, symbols, parent_name)
        elif node.type == "async_function_definition":
            self._extract_function(node, source, symbols, parent_name, is_async=True)
        elif node.type == "class_definition":
            self._extract_class(node, source, symbols, parent_name)
        elif node.type == "import_statement":
            self._extract_import(node, source, symbols)
        elif node.type == "import_from_statement":
            self._extract_import_from(node, source, symbols)
        elif node.type == "decorated_definition":
            # Handle decorated functions/classes
            for child in node.children:
                if child.type in ("function_definition", "async_function_definition", "class_definition"):
                    self._extract_from_node(child, source, symbols, parent_name)
        else:
            # Recurse into children
            for child in node.children:
                self._extract_from_node(child, source, symbols, parent_name)

    def _extract_function(
        self,
        node: Node,
        source: bytes,
        symbols: list[Symbol],
        parent_name: Optional[str],
        is_async: bool = False,
    ) -> None:
        """Extract function symbol."""
        name_node = node.child_by_field_name("name")
        if not name_node:
            return

        name = source[name_node.start_byte : name_node.end_byte].decode("utf-8")
        symbol_type = SymbolType.METHOD if parent_name else SymbolType.FUNCTION

        # Extract signature
        params_node = node.child_by_field_name("parameters")
        return_type = node.child_by_field_name("return_type")
        signature = self._build_function_signature(name, params_node, return_type, source, is_async)

        # Extract docstring
        docstring = self._extract_docstring(node, source)

        symbols.append(
            Symbol(
                name=name,
                symbol_type=symbol_type,
                line_start=node.start_point[0] + 1,
                line_end=node.end_point[0] + 1,
                language=Language.PYTHON,
                signature=signature,
                docstring=docstring,
                parent_name=parent_name,
                col_start=node.start_point[1],
                col_end=node.end_point[1],
            )
        )

    def _extract_class(
        self,
        node: Node,
        source: bytes,
        symbols: list[Symbol],
        parent_name: Optional[str],
    ) -> None:
        """Extract class symbol."""
        name_node = node.child_by_field_name("name")
        if not name_node:
            return

        name = source[name_node.start_byte : name_node.end_byte].decode("utf-8")

        # Extract signature with base classes
        signature = self._build_class_signature(node, source)

        # Extract docstring
        docstring = self._extract_docstring(node, source)

        symbols.append(
            Symbol(
                name=name,
                symbol_type=SymbolType.CLASS,
                line_start=node.start_point[0] + 1,
                line_end=node.end_point[0] + 1,
                language=Language.PYTHON,
                signature=signature,
                docstring=docstring,
                parent_name=parent_name,
                col_start=node.start_point[1],
                col_end=node.end_point[1],
            )
        )

        # Extract methods within the class
        body = node.child_by_field_name("body")
        if body:
            for child in body.children:
                self._extract_from_node(child, source, symbols, parent_name=name)

    def _extract_import(self, node: Node, source: bytes, symbols: list[Symbol]) -> None:
        """Extract import statement."""
        for child in node.children:
            if child.type == "dotted_name":
                import_name = source[child.start_byte : child.end_byte].decode("utf-8")
                symbols.append(
                    Symbol(
                        name=import_name,
                        symbol_type=SymbolType.IMPORT,
                        line_start=node.start_point[0] + 1,
                        line_end=node.end_point[0] + 1,
                        language=Language.PYTHON,
                    )
                )

    def _extract_import_from(self, node: Node, source: bytes, symbols: list[Symbol]) -> None:
        """Extract from...import statement."""
        module_name = None
        for child in node.children:
            if child.type == "dotted_name":
                module_name = source[child.start_byte : child.end_byte].decode("utf-8")
                break

        if module_name:
            symbols.append(
                Symbol(
                    name=module_name,
                    symbol_type=SymbolType.IMPORT,
                    line_start=node.start_point[0] + 1,
                    line_end=node.end_point[0] + 1,
                    language=Language.PYTHON,
                )
            )

    def _build_function_signature(
        self,
        name: str,
        params_node: Optional[Node],
        return_type: Optional[Node],
        source: bytes,
        is_async: bool = False,
    ) -> str:
        """Build function signature string."""
        async_prefix = "async " if is_async else ""
        params = ""
        if params_node:
            params = source[params_node.start_byte : params_node.end_byte].decode("utf-8")

        ret = ""
        if return_type:
            ret = " -> " + source[return_type.start_byte : return_type.end_byte].decode("utf-8")

        return f"{async_prefix}def {name}{params}{ret}:"

    def _build_class_signature(self, node: Node, source: bytes) -> str:
        """Build class signature string."""
        name_node = node.child_by_field_name("name")
        name = source[name_node.start_byte : name_node.end_byte].decode("utf-8") if name_node else "Unknown"

        # Find base classes
        bases = ""
        for child in node.children:
            if child.type == "argument_list":
                bases = source[child.start_byte : child.end_byte].decode("utf-8")
                break

        return f"class {name}{bases}:"

    def _extract_docstring(self, node: Node, source: bytes) -> Optional[str]:
        """Extract docstring from function or class."""
        body = node.child_by_field_name("body")
        if not body or not body.children:
            return None

        first_stmt = body.children[0]
        if first_stmt.type == "expression_statement":
            expr = first_stmt.children[0] if first_stmt.children else None
            if expr and expr.type == "string":
                docstring = source[expr.start_byte : expr.end_byte].decode("utf-8")
                # Strip quotes
                if docstring.startswith('"""') or docstring.startswith("'''"):
                    docstring = docstring[3:-3]
                elif docstring.startswith('"') or docstring.startswith("'"):
                    docstring = docstring[1:-1]
                return docstring.strip()
        return None


# =============================================================================
# TypeScript Plugin
# =============================================================================


# Known NestJS/Angular decorator mappings for semantic classification
KNOWN_DECORATORS: dict[str, str] = {
    # === Event Handling ===
    "OnEvent": "event_handler",
    "OnQueueEvent": "event_handler",
    "Process": "queue_processor",

    # === HTTP Handlers ===
    "Get": "http_handler",
    "Post": "http_handler",
    "Put": "http_handler",
    "Patch": "http_handler",
    "Delete": "http_handler",
    "Head": "http_handler",
    "Options": "http_handler",
    "All": "http_handler",

    # === Controllers ===
    "Controller": "controller",
    "Resolver": "graphql_resolver",

    # === Microservices ===
    "EventPattern": "message_handler",
    "MessagePattern": "message_handler",
    "GrpcMethod": "grpc_handler",
    "GrpcStreamMethod": "grpc_stream_handler",

    # === Dependency Injection ===
    "Injectable": "di_provider",
    "Inject": "di_injection",
    "Optional": "di_optional",
    "Global": "di_global",

    # === Middleware ===
    "UseGuards": "guard_chain",
    "UseInterceptors": "interceptor_chain",
    "UsePipes": "pipe_chain",
    "UseFilters": "filter_chain",

    # === Scheduling ===
    "Cron": "scheduled_task",
    "Interval": "scheduled_task",
    "Timeout": "scheduled_task",

    # === WebSockets ===
    "WebSocketGateway": "websocket_gateway",
    "SubscribeMessage": "websocket_handler",
    "ConnectedSocket": "websocket_param",
    "MessageBody": "websocket_param",

    # === Validation / Params ===
    "Body": "param_decorator",
    "Query": "param_decorator",
    "Param": "param_decorator",
    "Headers": "param_decorator",

    # === GraphQL ===
    "Mutation": "graphql_mutation",
    "Subscription": "graphql_subscription",
    "ResolveField": "graphql_field_resolver",

    # === Angular-specific ===
    "Component": "component",
    "Directive": "directive",
    "Pipe": "pipe",
    "NgModule": "module",
    "Input": "input_binding",
    "Output": "output_binding",
    "HostListener": "host_listener",
    "HostBinding": "host_binding",
    "ViewChild": "view_child",
    "ContentChild": "content_child",
}


class TypeScriptPlugin(LanguagePlugin):
    """Language plugin for TypeScript and TSX."""

    def __init__(self):
        self._ts_parser = Parser(TSLanguage(ts_typescript.language_typescript()))
        self._tsx_parser = Parser(TSLanguage(ts_typescript.language_tsx()))

    @property
    def supported_extensions(self) -> list[str]:
        return [".ts", ".tsx"]

    def parse(self, source: bytes, is_tsx: bool = False, **kwargs) -> Tree:
        """Parse TypeScript or TSX source code."""
        parser = self._tsx_parser if is_tsx else self._ts_parser
        return parser.parse(source)

    def extract_symbols(self, tree: Tree, source: bytes) -> list[Symbol]:
        """Extract symbols from TypeScript AST."""
        symbols = []
        self._extract_from_node(tree.root_node, source, symbols, parent_name=None)
        return symbols

    def _extract_from_node(
        self,
        node: Node,
        source: bytes,
        symbols: list[Symbol],
        parent_name: Optional[str],
    ) -> None:
        """Recursively extract symbols from AST node."""
        if node.type == "function_declaration":
            self._extract_function(node, source, symbols, parent_name)
        elif node.type == "class_declaration":
            self._extract_class(node, source, symbols, parent_name)
        elif node.type == "interface_declaration":
            self._extract_interface(node, source, symbols)
        elif node.type == "type_alias_declaration":
            self._extract_type_alias(node, source, symbols)
        elif node.type == "enum_declaration":
            self._extract_enum(node, source, symbols)
        elif node.type == "lexical_declaration":
            self._extract_lexical(node, source, symbols, parent_name)
        elif node.type == "export_statement":
            # Handle exported declarations
            for child in node.children:
                self._extract_from_node(child, source, symbols, parent_name)
        else:
            # Recurse into children
            for child in node.children:
                self._extract_from_node(child, source, symbols, parent_name)

    def _extract_function(
        self,
        node: Node,
        source: bytes,
        symbols: list[Symbol],
        parent_name: Optional[str],
    ) -> None:
        """Extract function symbol."""
        name_node = node.child_by_field_name("name")
        if not name_node:
            return

        name = source[name_node.start_byte : name_node.end_byte].decode("utf-8")
        docstring = self._extract_jsdoc(node, source)

        symbols.append(
            Symbol(
                name=name,
                symbol_type=SymbolType.FUNCTION,
                line_start=node.start_point[0] + 1,
                line_end=node.end_point[0] + 1,
                language=Language.TYPESCRIPT,
                docstring=docstring,
                parent_name=parent_name,
            )
        )

    def _extract_class(
        self,
        node: Node,
        source: bytes,
        symbols: list[Symbol],
        parent_name: Optional[str],
    ) -> None:
        """Extract class symbol with decorators.

        Extracts class-level decorators like @Controller, @Injectable, @Component
        and attaches them to the Symbol.
        """
        name_node = node.child_by_field_name("name")
        if not name_node:
            return

        name = source[name_node.start_byte : name_node.end_byte].decode("utf-8")
        docstring = self._extract_jsdoc(node, source)

        # Extract class-level decorators
        decorators = self._extract_decorators_from_node(node, source)

        symbols.append(
            Symbol(
                name=name,
                symbol_type=SymbolType.CLASS,
                line_start=node.start_point[0] + 1,
                line_end=node.end_point[0] + 1,
                language=Language.TYPESCRIPT,
                docstring=docstring,
                parent_name=parent_name,
                decorators=decorators,
            )
        )

        # Extract methods
        body = node.child_by_field_name("body")
        if body:
            for child in body.children:
                if child.type == "method_definition":
                    self._extract_method(child, source, symbols, name)

    def _extract_method(
        self,
        node: Node,
        source: bytes,
        symbols: list[Symbol],
        parent_name: str,
    ) -> None:
        """Extract method symbol with decorators.

        Extracts method-level decorators like @Get, @Post, @OnEvent
        and attaches them to the Symbol.
        """
        name_node = node.child_by_field_name("name")
        if not name_node:
            return

        name = source[name_node.start_byte : name_node.end_byte].decode("utf-8")

        # Extract method-level decorators
        decorators = self._extract_decorators_from_node(node, source)

        symbols.append(
            Symbol(
                name=name,
                symbol_type=SymbolType.METHOD,
                line_start=node.start_point[0] + 1,
                line_end=node.end_point[0] + 1,
                language=Language.TYPESCRIPT,
                parent_name=parent_name,
                decorators=decorators,
            )
        )

    def _extract_interface(self, node: Node, source: bytes, symbols: list[Symbol]) -> None:
        """Extract interface symbol."""
        name_node = node.child_by_field_name("name")
        if not name_node:
            return

        name = source[name_node.start_byte : name_node.end_byte].decode("utf-8")

        symbols.append(
            Symbol(
                name=name,
                symbol_type=SymbolType.INTERFACE,
                line_start=node.start_point[0] + 1,
                line_end=node.end_point[0] + 1,
                language=Language.TYPESCRIPT,
            )
        )

    def _extract_type_alias(self, node: Node, source: bytes, symbols: list[Symbol]) -> None:
        """Extract type alias symbol."""
        name_node = node.child_by_field_name("name")
        if not name_node:
            return

        name = source[name_node.start_byte : name_node.end_byte].decode("utf-8")

        symbols.append(
            Symbol(
                name=name,
                symbol_type=SymbolType.TYPE_ALIAS,
                line_start=node.start_point[0] + 1,
                line_end=node.end_point[0] + 1,
                language=Language.TYPESCRIPT,
            )
        )

    def _extract_enum(self, node: Node, source: bytes, symbols: list[Symbol]) -> None:
        """Extract enum symbol."""
        name_node = node.child_by_field_name("name")
        if not name_node:
            return

        name = source[name_node.start_byte : name_node.end_byte].decode("utf-8")

        symbols.append(
            Symbol(
                name=name,
                symbol_type=SymbolType.ENUM,
                line_start=node.start_point[0] + 1,
                line_end=node.end_point[0] + 1,
                language=Language.TYPESCRIPT,
            )
        )

    def _extract_lexical(
        self,
        node: Node,
        source: bytes,
        symbols: list[Symbol],
        parent_name: Optional[str],
    ) -> None:
        """Extract const/let declarations (including arrow functions)."""
        for child in node.children:
            if child.type == "variable_declarator":
                name_node = child.child_by_field_name("name")
                value_node = child.child_by_field_name("value")

                if name_node and value_node:
                    name = source[name_node.start_byte : name_node.end_byte].decode("utf-8")
                    # Check if it's an arrow function
                    if value_node.type == "arrow_function":
                        symbols.append(
                            Symbol(
                                name=name,
                                symbol_type=SymbolType.FUNCTION,
                                line_start=node.start_point[0] + 1,
                                line_end=node.end_point[0] + 1,
                                language=Language.TYPESCRIPT,
                                parent_name=parent_name,
                            )
                        )

    def _extract_jsdoc(self, node: Node, source: bytes) -> Optional[str]:
        """Extract JSDoc comment preceding a node."""
        # Look for preceding comment
        if node.prev_sibling and node.prev_sibling.type == "comment":
            comment = source[node.prev_sibling.start_byte : node.prev_sibling.end_byte].decode("utf-8")
            if comment.startswith("/**"):
                # Strip JSDoc markers
                lines = comment[3:-2].strip().split("\n")
                cleaned = []
                for line in lines:
                    line = line.strip()
                    if line.startswith("*"):
                        line = line[1:].strip()
                    cleaned.append(line)
                return " ".join(cleaned).strip()
        return None

    def _extract_decorator_info(self, node: Node, source: bytes) -> Optional[DecoratorInfo]:
        """Extract decorator information from a decorator AST node.

        Handles two decorator forms:
        1. @OnEvent('file.uploaded') - call_expression with arguments
        2. @Injectable - plain identifier without arguments

        Args:
            node: Tree-sitter Node of type "decorator"
            source: Source code as bytes

        Returns:
            DecoratorInfo or None if not a valid decorator
        """
        if node.type != "decorator":
            return None

        # Case 1: @OnEvent('x') - decorator with call_expression
        for child in node.children:
            if child.type == "call_expression":
                func_node = child.child_by_field_name("function")
                args_node = child.child_by_field_name("arguments")

                if func_node:
                    name = source[func_node.start_byte : func_node.end_byte].decode("utf-8")
                    arguments = self._extract_call_arguments(args_node, source) if args_node else []

                    return DecoratorInfo(
                        name=name,
                        decorator_type=KNOWN_DECORATORS.get(name),
                        arguments=arguments,
                        raw_text=source[node.start_byte : node.end_byte].decode("utf-8"),
                    )

            # Case 2: @Injectable - decorator with plain identifier
            elif child.type == "identifier":
                name = source[child.start_byte : child.end_byte].decode("utf-8")
                return DecoratorInfo(
                    name=name,
                    decorator_type=KNOWN_DECORATORS.get(name),
                    arguments=[],
                    raw_text=source[node.start_byte : node.end_byte].decode("utf-8"),
                )

        return None

    def _extract_call_arguments(self, args_node: Node, source: bytes) -> list[str]:
        """Extract arguments from a call expression arguments node.

        Handles various argument types:
        - Strings: 'event.name' or "event.name"
        - Template strings: `event.${type}`
        - Numbers: 5000
        - Identifiers: AuthGuard
        - Objects: { path: 'users' }

        Args:
            args_node: Tree-sitter Node of type "arguments"
            source: Source code as bytes

        Returns:
            List of argument values as strings
        """
        if not args_node:
            return []

        args = []
        for child in args_node.children:
            if child.type in ("string", "template_string", "number", "identifier", "true", "false"):
                args.append(source[child.start_byte : child.end_byte].decode("utf-8"))
            elif child.type == "object":
                # For @Controller({ path: 'users' }) - capture full object
                args.append(source[child.start_byte : child.end_byte].decode("utf-8"))
            elif child.type == "array":
                # For @UseGuards([Guard1, Guard2])
                args.append(source[child.start_byte : child.end_byte].decode("utf-8"))
        return args

    def _extract_decorators_from_node(self, node: Node, source: bytes) -> list[DecoratorInfo]:
        """Extract all decorators from a class or method node.

        Tree-sitter parses decorators in several ways depending on context:

        1. Method decorators in class body:
           class_body -> [decorator, method_definition]
           (decorator is a preceding sibling of method_definition)

        2. Exported decorated class:
           export_statement -> [decorator, export, class_declaration]
           (decorator is a preceding sibling of class_declaration inside export_statement)

        3. Non-exported decorated class:
           class_declaration -> [decorator, class, type_identifier, class_body]
           (decorator is a child of class_declaration, before the 'class' keyword)

        This method handles all cases.

        Args:
            node: The class_declaration or method_definition node
            source: Source code as bytes

        Returns:
            List of DecoratorInfo for all decorators on this node
        """
        decorators = []

        # Strategy 1: Look for decorator siblings preceding this node
        # (works for methods in class body and exported classes)
        current = node.prev_sibling
        while current:
            if current.type == "decorator":
                dec_info = self._extract_decorator_info(current, source)
                if dec_info:
                    decorators.append(dec_info)
            elif current.type in ("comment", "export"):
                # Skip comments and 'export' keyword between decorator and class
                pass
            else:
                # Stop when we hit a non-decorator/non-comment/non-export node
                break
            current = current.prev_sibling

        # Strategy 2: For class_declaration, also check children for decorators
        # (works for non-exported decorated classes where decorator is a child)
        if not decorators and node.type == "class_declaration":
            for child in node.children:
                if child.type == "decorator":
                    dec_info = self._extract_decorator_info(child, source)
                    if dec_info:
                        decorators.append(dec_info)
                elif child.type == "class":
                    # Stop when we hit the 'class' keyword
                    break

        # Decorators were collected in reverse order (nearest first) for strategy 1
        # but in order for strategy 2. Reverse only if we used strategy 1.
        if decorators and node.prev_sibling:
            # Check if first decorator was a sibling (strategy 1)
            check = node.prev_sibling
            while check and check.type in ("comment", "export"):
                check = check.prev_sibling
            if check and check.type == "decorator":
                decorators.reverse()

        return decorators


# =============================================================================
# Java Plugin
# =============================================================================


class JavaPlugin(LanguagePlugin):
    """Language plugin for Java."""

    def __init__(self):
        self._parser = Parser(TSLanguage(ts_java.language()))

    @property
    def supported_extensions(self) -> list[str]:
        return [".java"]

    def parse(self, source: bytes, **kwargs) -> Tree:
        """Parse Java source code."""
        return self._parser.parse(source)

    def extract_symbols(self, tree: Tree, source: bytes) -> list[Symbol]:
        """Extract symbols from Java AST."""
        symbols = []
        self._extract_from_node(tree.root_node, source, symbols, parent_name=None)
        return symbols

    def _extract_from_node(
        self,
        node: Node,
        source: bytes,
        symbols: list[Symbol],
        parent_name: Optional[str],
    ) -> None:
        """Recursively extract symbols from AST node."""
        if node.type == "class_declaration":
            self._extract_class(node, source, symbols, parent_name)
        elif node.type == "interface_declaration":
            self._extract_interface(node, source, symbols, parent_name)
        elif node.type == "enum_declaration":
            self._extract_enum(node, source, symbols, parent_name)
        elif node.type == "method_declaration":
            self._extract_method(node, source, symbols, parent_name)
        elif node.type == "import_declaration":
            self._extract_import(node, source, symbols)
        else:
            # Recurse into children
            for child in node.children:
                self._extract_from_node(child, source, symbols, parent_name)

    def _extract_class(
        self,
        node: Node,
        source: bytes,
        symbols: list[Symbol],
        parent_name: Optional[str],
    ) -> None:
        """Extract class symbol."""
        name_node = node.child_by_field_name("name")
        if not name_node:
            return

        name = source[name_node.start_byte : name_node.end_byte].decode("utf-8")
        docstring = self._extract_javadoc(node, source)

        symbols.append(
            Symbol(
                name=name,
                symbol_type=SymbolType.CLASS,
                line_start=node.start_point[0] + 1,
                line_end=node.end_point[0] + 1,
                language=Language.JAVA,
                docstring=docstring,
                parent_name=parent_name,
            )
        )

        # Extract methods and inner classes
        body = node.child_by_field_name("body")
        if body:
            for child in body.children:
                self._extract_from_node(child, source, symbols, parent_name=name)

    def _extract_interface(
        self,
        node: Node,
        source: bytes,
        symbols: list[Symbol],
        parent_name: Optional[str],
    ) -> None:
        """Extract interface symbol."""
        name_node = node.child_by_field_name("name")
        if not name_node:
            return

        name = source[name_node.start_byte : name_node.end_byte].decode("utf-8")

        symbols.append(
            Symbol(
                name=name,
                symbol_type=SymbolType.INTERFACE,
                line_start=node.start_point[0] + 1,
                line_end=node.end_point[0] + 1,
                language=Language.JAVA,
                parent_name=parent_name,
            )
        )

    def _extract_enum(
        self,
        node: Node,
        source: bytes,
        symbols: list[Symbol],
        parent_name: Optional[str],
    ) -> None:
        """Extract enum symbol."""
        name_node = node.child_by_field_name("name")
        if not name_node:
            return

        name = source[name_node.start_byte : name_node.end_byte].decode("utf-8")

        symbols.append(
            Symbol(
                name=name,
                symbol_type=SymbolType.ENUM,
                line_start=node.start_point[0] + 1,
                line_end=node.end_point[0] + 1,
                language=Language.JAVA,
                parent_name=parent_name,
            )
        )

    def _extract_method(
        self,
        node: Node,
        source: bytes,
        symbols: list[Symbol],
        parent_name: Optional[str],
    ) -> None:
        """Extract method symbol."""
        name_node = node.child_by_field_name("name")
        if not name_node:
            return

        name = source[name_node.start_byte : name_node.end_byte].decode("utf-8")

        symbols.append(
            Symbol(
                name=name,
                symbol_type=SymbolType.METHOD,
                line_start=node.start_point[0] + 1,
                line_end=node.end_point[0] + 1,
                language=Language.JAVA,
                parent_name=parent_name,
            )
        )

    def _extract_import(self, node: Node, source: bytes, symbols: list[Symbol]) -> None:
        """Extract import symbol."""
        # Find the scoped identifier or identifier
        for child in node.children:
            if child.type in ("scoped_identifier", "identifier"):
                name = source[child.start_byte : child.end_byte].decode("utf-8")
                symbols.append(
                    Symbol(
                        name=name,
                        symbol_type=SymbolType.IMPORT,
                        line_start=node.start_point[0] + 1,
                        line_end=node.end_point[0] + 1,
                        language=Language.JAVA,
                    )
                )
                break

    def _extract_javadoc(self, node: Node, source: bytes) -> Optional[str]:
        """Extract Javadoc comment preceding a node."""
        # Look for preceding comment
        if node.prev_sibling and node.prev_sibling.type == "block_comment":
            comment = source[node.prev_sibling.start_byte : node.prev_sibling.end_byte].decode("utf-8")
            if comment.startswith("/**"):
                # Strip Javadoc markers
                lines = comment[3:-2].strip().split("\n")
                cleaned = []
                for line in lines:
                    line = line.strip()
                    if line.startswith("*"):
                        line = line[1:].strip()
                    # Skip @param, @return, etc.
                    if not line.startswith("@"):
                        cleaned.append(line)
                return " ".join(cleaned).strip()
        return None


# =============================================================================
# Main AST Parser
# =============================================================================


class ASTParser:
    """Main AST parser that coordinates language plugins."""

    def __init__(
        self,
        config: Optional[ASTParserConfig] = None,
        plugins: Optional[list[LanguagePlugin]] = None,
    ):
        self.config = config or ASTParserConfig()
        self.plugins: dict[Language, LanguagePlugin] = {}
        self.statistics = ParseStatistics()

        # Register plugins
        if plugins:
            for plugin in plugins:
                self._register_plugin(plugin)
        else:
            # Register default plugins
            self._register_plugin(PythonPlugin())
            self._register_plugin(TypeScriptPlugin())
            self._register_plugin(JavaPlugin())

    def _register_plugin(self, plugin: LanguagePlugin) -> None:
        """Register a language plugin."""
        for ext in plugin.supported_extensions:
            lang = Language.from_extension(ext)
            if lang:
                self.plugins[lang] = plugin

    def parse_file(self, file_path: Path) -> ParseResult:
        """Parse a file and extract symbols."""
        file_path = Path(file_path)

        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        # Check language support
        language = Language.from_path(file_path)
        if not language:
            raise UnsupportedLanguageError(f"Unsupported file extension: {file_path.suffix}")

        plugin = self.plugins.get(language)
        if not plugin:
            raise UnsupportedLanguageError(f"No plugin for language: {language}")

        # Check file size
        file_size = file_path.stat().st_size
        if file_size > self.config.max_file_size_bytes:
            error = ParseError(
                line=0,
                column=0,
                message=f"File size ({file_size} bytes) exceeds limit ({self.config.max_file_size_bytes} bytes)",
            )
            self.statistics.record_failure()
            return ParseResult(
                file_path=file_path,
                language=language,
                symbols=[],
                errors=[error],
                is_partial=False,
            )

        # Read file
        try:
            source = file_path.read_bytes()
        except Exception as e:
            error = ParseError(line=0, column=0, message=f"Failed to read file: {e}")
            self.statistics.record_failure()
            return ParseResult(
                file_path=file_path,
                language=language,
                symbols=[],
                errors=[error],
                is_partial=False,
            )

        # Check for binary content
        if b"\x00" in source[:8192]:  # Check first 8KB
            error = ParseError(line=0, column=0, message="Binary file detected")
            self.statistics.record_failure()
            return ParseResult(
                file_path=file_path,
                language=language,
                symbols=[],
                errors=[error],
                is_partial=False,
            )

        return self._parse_source(source, language, file_path, plugin)

    def parse_string(self, code: str, language: Language) -> ParseResult:
        """Parse code from string."""
        return self.parse_bytes(code.encode("utf-8"), language)

    def parse_bytes(self, source: bytes, language: Language) -> ParseResult:
        """Parse code from bytes."""
        plugin = self.plugins.get(language)
        if not plugin:
            raise UnsupportedLanguageError(f"No plugin for language: {language}")

        return self._parse_source(source, language, None, plugin)

    def _parse_source(
        self,
        source: bytes,
        language: Language,
        file_path: Optional[Path],
        plugin: LanguagePlugin,
    ) -> ParseResult:
        """Parse source code with plugin."""
        try:
            # Parse with special handling for TSX
            if language == Language.TSX and isinstance(plugin, TypeScriptPlugin):
                tree = plugin.parse(source, is_tsx=True)
            else:
                tree = plugin.parse(source)

            # Extract symbols
            symbols = plugin.extract_symbols(tree, source)

            # Check for parse errors
            errors = []
            is_partial = tree.root_node.has_error
            if is_partial:
                self._collect_errors(tree.root_node, errors, source)

            # Update statistics
            if errors and not symbols:
                self.statistics.record_failure()
                if self.config.log_errors:
                    logger.warning(f"Failed to parse {file_path}: {errors[0].message if errors else 'Unknown error'}")
            elif is_partial:
                self.statistics.record_partial(len(symbols), len(errors))
            else:
                self.statistics.record_success(len(symbols))

            return ParseResult(
                file_path=file_path,
                language=language,
                symbols=symbols,
                errors=errors,
                is_partial=is_partial,
            )

        except Exception as e:
            error = ParseError(line=0, column=0, message=str(e))
            self.statistics.record_failure()
            if self.config.log_errors:
                logger.error(f"Exception parsing {file_path}: {e}")
            return ParseResult(
                file_path=file_path,
                language=language,
                symbols=[],
                errors=[error],
                is_partial=False,
            )

    def _collect_errors(self, node: Node, errors: list[ParseError], source: bytes) -> None:
        """Collect parse errors from AST."""
        if node.type == "ERROR" or node.is_missing:
            errors.append(
                ParseError(
                    line=node.start_point[0] + 1,
                    column=node.start_point[1],
                    message=f"Syntax error at line {node.start_point[0] + 1}",
                )
            )
        for child in node.children:
            self._collect_errors(child, errors, source)

    def parse_directory(
        self,
        directory: Path,
        recursive: bool = False,
        extensions: Optional[list[str]] = None,
    ) -> list[ParseResult]:
        """Parse all supported files in a directory."""
        directory = Path(directory)
        results = []

        # Determine extensions to scan
        if extensions is None:
            extensions = [".py", ".ts", ".tsx", ".java"]

        # Collect files
        if recursive:
            files = []
            for ext in extensions:
                files.extend(directory.rglob(f"*{ext}"))
        else:
            files = []
            for ext in extensions:
                files.extend(directory.glob(f"*{ext}"))

        # Parse each file
        for file_path in sorted(files):
            try:
                result = self.parse_file(file_path)
                results.append(result)
            except UnsupportedLanguageError:
                continue
            except Exception as e:
                if self.config.log_errors:
                    logger.error(f"Error parsing {file_path}: {e}")

        return results


# =============================================================================
# Factory Function
# =============================================================================


def create_parser(
    config: Optional[ASTParserConfig] = None,
    plugins: Optional[list[LanguagePlugin]] = None,
) -> ASTParser:
    """Create an AST parser with optional configuration and plugins."""
    return ASTParser(config=config, plugins=plugins)


# =============================================================================
# Event Registry for NestJS Event-Based Call Edges
# =============================================================================


import re


@dataclass
class EventEdge:
    """Represents an event-based connection between publisher and subscriber.

    Used to generate TRIGGERS_EVENT edges in the code graph.
    """

    publisher_symbol_id: str
    subscriber_symbol_id: str
    event_name: str
    edge_type: str = "TRIGGERS_EVENT"


@dataclass
class EventRegistry:
    """Registry for event publishers and subscribers.

    Collects @OnEvent handlers and emit() calls to generate call edges
    between event emitters and handlers.
    """

    publishers: dict[str, list[str]] = field(default_factory=dict)  # event_name -> [symbol_ids]
    subscribers: dict[str, list[str]] = field(default_factory=dict)  # event_name -> [symbol_ids]

    def register_publisher(self, event_name: str, symbol_id: str) -> None:
        """Register an event publisher (emit() caller)."""
        if event_name not in self.publishers:
            self.publishers[event_name] = []
        if symbol_id not in self.publishers[event_name]:
            self.publishers[event_name].append(symbol_id)

    def register_subscriber(self, event_name: str, symbol_id: str) -> None:
        """Register an event subscriber (@OnEvent handler)."""
        if event_name not in self.subscribers:
            self.subscribers[event_name] = []
        if symbol_id not in self.subscribers[event_name]:
            self.subscribers[event_name].append(symbol_id)

    def get_edges(self) -> list[EventEdge]:
        """Generate EventEdge objects for all publisher->subscriber connections.

        Returns:
            List of EventEdge objects representing TRIGGERS_EVENT relationships
        """
        edges = []
        for event_name, subscriber_ids in self.subscribers.items():
            publisher_ids = self.publishers.get(event_name, [])
            for pub_id in publisher_ids:
                for sub_id in subscriber_ids:
                    edges.append(EventEdge(
                        publisher_symbol_id=pub_id,
                        subscriber_symbol_id=sub_id,
                        event_name=event_name,
                    ))
        return edges


# Regex pattern for emit() calls
# Matches: .emit('event.name', ...) or .emit("event.name", ...)
_EMIT_PATTERN = re.compile(
    r"""
    \.emit\s*\(\s*       # .emit(
    ['"]([^'"]+)['"]     # 'event.name' or "event.name" - capture group 1
    """,
    re.VERBOSE,
)


def discover_event_publishers(symbol: Symbol, source_body: str) -> list[str]:
    """Discover event names that a symbol publishes via emit() calls.

    Searches the symbol's body for patterns like:
    - this.eventEmitter.emit('event.name', payload)
    - eventEmitter.emit('event.name')
    - this.emit('event.name')

    Args:
        symbol: The Symbol to search for emit() calls
        source_body: The source code body of the symbol

    Returns:
        List of event names that this symbol publishes
    """
    matches = _EMIT_PATTERN.findall(source_body)
    return list(set(matches))  # Deduplicate


def discover_event_subscribers(symbols: list[Symbol]) -> dict[str, list[Symbol]]:
    """Collect all @OnEvent handlers from a list of symbols.

    Scans decorators for event_handler type decorators and extracts
    the event name from the decorator arguments.

    Args:
        symbols: List of Symbol objects to scan

    Returns:
        Dict mapping event_name -> [handler_symbols]
    """
    subscribers: dict[str, list[Symbol]] = {}

    for sym in symbols:
        for dec in sym.decorators:
            if dec.decorator_type == "event_handler" and dec.arguments:
                # Extract event name from first argument
                # Arguments look like: ["'file.uploaded'"] or ['"file.uploaded"']
                event_name = dec.arguments[0].strip("'\"")
                if event_name not in subscribers:
                    subscribers[event_name] = []
                subscribers[event_name].append(sym)

    return subscribers


def build_event_registry(
    symbols: list[Symbol],
    symbol_id_map: dict[str, str],
    source_map: dict[str, str],
) -> EventRegistry:
    """Build an EventRegistry from parsed symbols.

    Args:
        symbols: List of all parsed Symbol objects
        symbol_id_map: Mapping from symbol name to symbol ID
        source_map: Mapping from symbol name to source code body

    Returns:
        Populated EventRegistry with publishers and subscribers
    """
    registry = EventRegistry()

    # Register subscribers from @OnEvent decorators
    event_subscribers = discover_event_subscribers(symbols)
    for event_name, handler_symbols in event_subscribers.items():
        for sym in handler_symbols:
            symbol_key = f"{sym.parent_name}.{sym.name}" if sym.parent_name else sym.name
            symbol_id = symbol_id_map.get(symbol_key, symbol_key)
            registry.register_subscriber(event_name, symbol_id)

    # Register publishers from emit() calls
    for sym in symbols:
        symbol_key = f"{sym.parent_name}.{sym.name}" if sym.parent_name else sym.name
        source_body = source_map.get(symbol_key, "")
        if source_body:
            event_names = discover_event_publishers(sym, source_body)
            for event_name in event_names:
                symbol_id = symbol_id_map.get(symbol_key, symbol_key)
                registry.register_publisher(event_name, symbol_id)

    return registry


def generate_event_edges(
    registry: EventRegistry,
    repo_id: str,
) -> list[dict]:
    """Generate Neo4j-compatible edge data from EventRegistry.

    Args:
        registry: Populated EventRegistry
        repo_id: Repository ID for edge properties

    Returns:
        List of edge dicts for Neo4j import
    """
    edges = []
    for edge in registry.get_edges():
        edges.append({
            "source_id": edge.publisher_symbol_id,
            "target_id": edge.subscriber_symbol_id,
            "relationship": edge.edge_type,
            "properties": {
                "event_name": edge.event_name,
                "repo_id": repo_id,
                "inferred": True,
            },
        })
    return edges
