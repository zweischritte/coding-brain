"""Fallback symbol extraction helpers."""

from __future__ import annotations

import ast
import re
from typing import Iterable

from openmemory.api.indexing.ast_parser import Language, Symbol, SymbolType


def extract_python_symbols(source: str) -> list[Symbol]:
    """Extract Python symbols using the built-in AST as a fallback."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return _extract_python_symbols_regex(source)

    symbols: list[Symbol] = []
    class_stack: list[str] = []

    def add_symbol(
        name: str,
        symbol_type: SymbolType,
        node: ast.AST,
        parent_name: str | None = None,
        signature: str | None = None,
        docstring: str | None = None,
    ) -> None:
        line_start = getattr(node, "lineno", 0) or 0
        line_end = getattr(node, "end_lineno", line_start) or line_start
        col_start = getattr(node, "col_offset", None)
        col_end = getattr(node, "end_col_offset", None)

        symbols.append(
            Symbol(
                name=name,
                symbol_type=symbol_type,
                line_start=line_start,
                line_end=line_end,
                language=Language.PYTHON,
                signature=signature,
                docstring=docstring,
                parent_name=parent_name,
                col_start=col_start,
                col_end=col_end,
            )
        )

    def format_signature(node: ast.AST) -> str:
        args = getattr(node, "args", None)
        if not isinstance(args, ast.arguments):
            return f"{getattr(node, 'name', '')}()"

        parts: list[str] = []
        for arg in getattr(args, "posonlyargs", []):
            parts.append(arg.arg)
        if args.posonlyargs and (
            args.args or args.vararg or args.kwonlyargs or args.kwarg
        ):
            parts.append("/")
        for arg in args.args:
            parts.append(arg.arg)
        if args.vararg:
            parts.append(f"*{args.vararg.arg}")
        elif args.kwonlyargs:
            parts.append("*")
        for arg in args.kwonlyargs:
            parts.append(arg.arg)
        if args.kwarg:
            parts.append(f"**{args.kwarg.arg}")

        return f"{getattr(node, 'name', '')}({', '.join(parts)})"

    class Visitor(ast.NodeVisitor):
        def visit_ClassDef(self, node: ast.ClassDef) -> None:
            add_symbol(
                name=node.name,
                symbol_type=SymbolType.CLASS,
                node=node,
                signature=f"class {node.name}",
                docstring=ast.get_docstring(node),
            )
            class_stack.append(node.name)
            self.generic_visit(node)
            class_stack.pop()

        def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
            is_method = bool(class_stack)
            symbol_type = SymbolType.METHOD if is_method else SymbolType.FUNCTION
            signature = f"def {format_signature(node)}"
            add_symbol(
                name=node.name,
                symbol_type=symbol_type,
                node=node,
                parent_name=class_stack[-1] if is_method else None,
                signature=signature,
                docstring=ast.get_docstring(node),
            )
            self.generic_visit(node)

        def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
            is_method = bool(class_stack)
            symbol_type = SymbolType.METHOD if is_method else SymbolType.FUNCTION
            signature = f"async def {format_signature(node)}"
            add_symbol(
                name=node.name,
                symbol_type=symbol_type,
                node=node,
                parent_name=class_stack[-1] if is_method else None,
                signature=signature,
                docstring=ast.get_docstring(node),
            )
            self.generic_visit(node)

    Visitor().visit(tree)
    return symbols


def _extract_python_symbols_regex(source: str) -> list[Symbol]:
    """Extract Python symbols using a basic regex fallback."""
    symbols: list[Symbol] = []
    class_pattern = re.compile(r"^\s*class\s+([A-Za-z_][A-Za-z0-9_]*)")
    def_pattern = re.compile(r"^\s*(async\s+def|def)\s+([A-Za-z_][A-Za-z0-9_]*)")

    for line_no, line in enumerate(source.splitlines(), start=1):
        class_match = class_pattern.match(line)
        if class_match:
            name = class_match.group(1)
            symbols.append(
                Symbol(
                    name=name,
                    symbol_type=SymbolType.CLASS,
                    line_start=line_no,
                    line_end=line_no,
                    language=Language.PYTHON,
                    signature=f"class {name}",
                )
            )
            continue

        def_match = def_pattern.match(line)
        if def_match:
            name = def_match.group(2)
            signature = line.strip().split(":")[0]
            symbols.append(
                Symbol(
                    name=name,
                    symbol_type=SymbolType.FUNCTION,
                    line_start=line_no,
                    line_end=line_no,
                    language=Language.PYTHON,
                    signature=signature,
                )
            )

    return symbols


def has_python_symbols(symbols: Iterable[Symbol]) -> bool:
    """Return True if the symbol list includes Python symbols."""
    return any(symbol.language == Language.PYTHON for symbol in symbols)
