"""Deterministic call/import edge extraction for code indexing.

Builds AST-based edges for TypeScript, Java, Go, and Python without
regex-based heuristics. Intended to complement inferred edges.
"""

from __future__ import annotations

import ast as py_ast
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import tree_sitter_go as ts_go
import tree_sitter_java as ts_java
import tree_sitter_typescript as ts_typescript
from tree_sitter import Language as TSLanguage, Parser, Node

from openmemory.api.indexing.ast_parser import Language, Symbol, SymbolType

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SymbolRef:
    """Symbol reference with resolved ID and metadata."""

    symbol_id: str
    symbol: Symbol
    file_path: Path


@dataclass
class DeterministicEdges:
    """Deterministic edges for a single file."""

    call_edges: list[tuple[str, str]] = field(default_factory=list)
    import_targets: list[Path] = field(default_factory=list)
    field_reads: list[tuple[str, str]] = field(default_factory=list)
    field_writes: list[tuple[str, str]] = field(default_factory=list)
    schema_fields: list["SchemaFieldDefinition"] = field(default_factory=list)
    schema_expose_edges: list[tuple[str, str]] = field(default_factory=list)
    path_literals: list["PathLiteralDefinition"] = field(default_factory=list)


@dataclass(frozen=True)
class SchemaFieldDefinition:
    """Schema field node definition."""

    schema_id: str
    name: str
    schema_type: str
    schema_name: Optional[str]
    file_path: Path
    line_start: Optional[int]
    line_end: Optional[int]
    nullable: Optional[bool] = None
    field_type: Optional[str] = None


@dataclass(frozen=True)
class PathLiteralDefinition:
    """Path literal node definition."""

    path: str
    normalized_path: str
    segments: list[str]
    leaf: str
    file_path: Path
    line_start: int
    line_end: int
    confidence: str
    start_byte: Optional[int] = None
    end_byte: Optional[int] = None


@dataclass(frozen=True)
class TypeHint:
    """Type hint for TypeScript resolution."""

    kind: str
    name: Optional[str] = None
    schema_name: Optional[str] = None
    schema_file_path: Optional[Path] = None


class RepoSymbolIndex:
    """Repository-wide symbol index for deterministic resolution."""

    def __init__(self, root_path: Path):
        self.root_path = Path(root_path)
        self.symbols_by_file: dict[Path, list[SymbolRef]] = {}
        self.functions_by_file: dict[Path, dict[str, str]] = {}
        self.classes_by_file: dict[Path, dict[str, str]] = {}
        self.methods_by_file: dict[Path, dict[str, dict[str, str]]] = {}
        self.fields_by_file: dict[Path, dict[str, dict[str, str]]] = {}

        self.python_module_by_file: dict[Path, str] = {}
        self.python_module_map: dict[str, Path] = {}

        self.java_package_by_file: dict[Path, str] = {}
        self.java_class_map: dict[str, Path] = {}

        self.go_module_path: Optional[str] = self._read_go_module()
        self.go_functions_by_dir: dict[Path, dict[str, str]] = {}

    def add_file_symbols(
        self,
        file_path: Path,
        language: Language,
        symbol_pairs: list[tuple[Symbol, str]],
        content: str,
    ) -> None:
        file_path = Path(file_path)
        self.symbols_by_file[file_path] = [
            SymbolRef(symbol_id=symbol_id, symbol=symbol, file_path=file_path)
            for symbol, symbol_id in symbol_pairs
        ]

        for symbol, symbol_id in symbol_pairs:
            if symbol.symbol_type == SymbolType.FUNCTION:
                self.functions_by_file.setdefault(file_path, {})[symbol.name] = symbol_id
            elif symbol.symbol_type == SymbolType.CLASS:
                self.classes_by_file.setdefault(file_path, {})[symbol.name] = symbol_id
            elif symbol.symbol_type == SymbolType.METHOD and symbol.parent_name:
                self.methods_by_file.setdefault(file_path, {}).setdefault(symbol.parent_name, {})[
                    symbol.name
                ] = symbol_id
            elif symbol.symbol_type in (SymbolType.FIELD, SymbolType.PROPERTY) and symbol.parent_name:
                self.fields_by_file.setdefault(file_path, {}).setdefault(symbol.parent_name, {})[
                    symbol.name
                ] = symbol_id

            if language == Language.GO and symbol.symbol_type == SymbolType.FUNCTION:
                self.go_functions_by_dir.setdefault(file_path.parent, {})[symbol.name] = symbol_id

        if language == Language.PYTHON:
            module_name = self._python_module_name(file_path)
            if module_name:
                self.python_module_by_file[file_path] = module_name
                self.python_module_map.setdefault(module_name, file_path)

        if language == Language.JAVA:
            package_name = self._java_package_name(content)
            self.java_package_by_file[file_path] = package_name
            for symbol, _ in symbol_pairs:
                if symbol.symbol_type in (SymbolType.CLASS, SymbolType.INTERFACE, SymbolType.ENUM):
                    full_name = f"{package_name}.{symbol.name}" if package_name else symbol.name
                    self.java_class_map.setdefault(full_name, file_path)

    def resolve_function_in_file(self, file_path: Path, name: str) -> Optional[str]:
        return self.functions_by_file.get(Path(file_path), {}).get(name)

    def resolve_class_in_file(self, file_path: Path, name: str) -> Optional[str]:
        return self.classes_by_file.get(Path(file_path), {}).get(name)

    def resolve_method_in_class(self, file_path: Path, class_name: str, method_name: str) -> Optional[str]:
        return (
            self.methods_by_file.get(Path(file_path), {})
            .get(class_name, {})
            .get(method_name)
        )

    def resolve_field_in_type(self, file_path: Path, type_name: str, field_name: str) -> Optional[str]:
        return (
            self.fields_by_file.get(Path(file_path), {})
            .get(type_name, {})
            .get(field_name)
        )

    def resolve_field_global(self, type_name: str, field_name: str) -> Optional[str]:
        for fields in self.fields_by_file.values():
            field_id = fields.get(type_name, {}).get(field_name)
            if field_id:
                return field_id
        return None

    def resolve_symbol_in_file(self, file_path: Path, name: str) -> Optional[str]:
        symbol_id = self.resolve_function_in_file(file_path, name)
        if symbol_id:
            return symbol_id
        return self.resolve_class_in_file(file_path, name)

    def resolve_python_module(self, module_name: str) -> Optional[Path]:
        return self.python_module_map.get(module_name)

    def resolve_java_class(self, full_name: str) -> Optional[Path]:
        return self.java_class_map.get(full_name)

    def resolve_go_import_dir(self, import_path: str) -> Optional[Path]:
        if not self.go_module_path:
            return None
        if not import_path.startswith(self.go_module_path):
            return None
        rel = import_path[len(self.go_module_path) :].lstrip("/")
        return (self.root_path / rel).resolve()

    def resolve_go_function_in_dir(self, dir_path: Path, name: str) -> Optional[str]:
        return self.go_functions_by_dir.get(Path(dir_path), {}).get(name)

    def resolve_go_package_file(self, dir_path: Path) -> Optional[Path]:
        dir_path = Path(dir_path)
        for path in self.symbols_by_file:
            if path.parent == dir_path and path.suffix == ".go":
                return path
        return None

    def python_module_for_file(self, file_path: Path) -> Optional[str]:
        return self.python_module_by_file.get(Path(file_path))

    def java_package_for_file(self, file_path: Path) -> Optional[str]:
        return self.java_package_by_file.get(Path(file_path))

    def _python_module_name(self, file_path: Path) -> str:
        try:
            rel_path = file_path.relative_to(self.root_path)
        except ValueError:
            rel_path = file_path

        parts = list(rel_path.parts)
        if not parts:
            return ""

        if parts[-1] == "__init__.py":
            parts = parts[:-1]
        else:
            parts[-1] = parts[-1].replace(".py", "")
        return ".".join([p for p in parts if p])

    def _java_package_name(self, content: str) -> str:
        match = re.search(r"^\\s*package\\s+([\\w.]+)\\s*;", content, re.MULTILINE)
        return match.group(1) if match else ""

    def _read_go_module(self) -> Optional[str]:
        go_mod = self.root_path / "go.mod"
        if not go_mod.exists():
            return None
        try:
            for line in go_mod.read_text().splitlines():
                line = line.strip()
                if line.startswith("module "):
                    return line.split(" ", 1)[1].strip()
        except IOError:
            return None
        return None


class DeterministicEdgeExtractor:
    """Extract deterministic call/import edges per language."""

    def __init__(
        self,
        root_path: Path,
        symbol_index: RepoSymbolIndex,
        package_map: Optional[dict[str, Path]] = None,
    ):
        self.root_path = Path(root_path)
        self.symbol_index = symbol_index
        self._ts_parser = Parser(TSLanguage(ts_typescript.language_typescript()))
        self._tsx_parser = Parser(TSLanguage(ts_typescript.language_tsx()))
        self._java_parser = Parser(TSLanguage(ts_java.language()))
        self._go_parser = Parser(TSLanguage(ts_go.language()))
        self._package_map = package_map or {}
        self._global_zod_schema_map: dict[str, set[Path]] = {}

    def build_global_zod_schema_map(
        self,
        source_files: list[Path],
        parse_cache: dict[Path, "FileParseCache"],
    ) -> None:
        schema_map: dict[str, set[Path]] = {}
        for file_path in source_files:
            cache_entry = parse_cache.get(file_path)
            if not cache_entry or cache_entry.language not in (Language.TYPESCRIPT, Language.TSX):
                continue
            try:
                content = file_path.read_text(errors="ignore")
            except Exception:
                continue
            source = content.encode("utf-8")
            parser = self._tsx_parser if cache_entry.language == Language.TSX else self._ts_parser
            root = parser.parse(source).root_node
            zod_aliases = self._collect_ts_zod_aliases(source, root)
            if not zod_aliases:
                zod_aliases.add("z")
            schema_names = self._collect_ts_zod_schema_names(source, root, zod_aliases)
            for name in schema_names:
                schema_map.setdefault(name, set()).add(file_path)

        self._global_zod_schema_map = schema_map

    def extract_edges(self, file_path: Path, language: Language, content: str) -> DeterministicEdges:
        if language in (Language.TYPESCRIPT, Language.TSX):
            return self._extract_typescript_edges(file_path, language, content)
        if language == Language.JAVA:
            return self._extract_java_edges(file_path, content)
        if language == Language.GO:
            return self._extract_go_edges(file_path, content)
        if language == Language.PYTHON:
            return self._extract_python_edges(file_path, content)
        return DeterministicEdges()

    # ---------------------------------------------------------------------
    # TypeScript
    # ---------------------------------------------------------------------

    def _extract_typescript_edges(
        self,
        file_path: Path,
        language: Language,
        content: str,
    ) -> DeterministicEdges:
        source = content.encode("utf-8")
        parser = self._tsx_parser if language == Language.TSX else self._ts_parser
        tree = parser.parse(source)
        root = tree.root_node

        import_map, import_targets, import_module_map = self._collect_ts_imports(
            file_path,
            source,
            root,
        )
        call_edges: list[tuple[str, str]] = []
        field_reads: list[tuple[str, str]] = []
        field_writes: list[tuple[str, str]] = []
        type_scopes: list[dict[str, TypeHint]] = [{}]
        return_type_scopes: list[Optional[TypeHint]] = [None]
        zod_aliases = self._collect_ts_zod_aliases(source, root)
        if not zod_aliases:
            zod_aliases.add("z")
        local_zod_schemas = self._collect_ts_zod_schema_names(source, root, zod_aliases)

        def source_for_access(current_class: Optional[str], current_caller: Optional[str]) -> Optional[str]:
            if current_caller:
                return current_caller
            if current_class:
                return self.symbol_index.resolve_class_in_file(file_path, current_class)
            return None

        def register_type(name: str, type_hint: Optional[TypeHint]) -> None:
            if not name or not type_hint:
                return
            type_scopes[-1][name] = type_hint

        def current_return_type() -> Optional[TypeHint]:
            return return_type_scopes[-1]

        def extract_return_type(fn_node: Node) -> Optional[TypeHint]:
            type_node = fn_node.child_by_field_name("return_type") or fn_node.child_by_field_name(
                "type"
            )
            return self._ts_type_hint(
                type_node,
                source,
                file_path,
                import_map,
                import_module_map,
                zod_aliases,
                local_zod_schemas,
            )

        def collect_param_types(params_node: Optional[Node]) -> dict[str, TypeHint]:
            param_types: dict[str, TypeHint] = {}
            if not params_node:
                return param_types
            for param in params_node.children:
                if param.type not in ("required_parameter", "optional_parameter"):
                    continue
                param_name = param.child_by_field_name("pattern") or param.child_by_field_name("name")
                if not param_name:
                    param_name = next(
                        (
                            child
                            for child in param.children
                            if child.type in ("identifier", "property_identifier")
                        ),
                        None,
                    )
                if not param_name:
                    continue
                name = self._node_text(source, param_name)
                type_node = param.child_by_field_name("type") or param.child_by_field_name("type_annotation")
                type_hint = self._ts_type_hint(
                    type_node,
                    source,
                    file_path,
                    import_map,
                    import_module_map,
                    zod_aliases,
                    local_zod_schemas,
                )
                if name and type_hint:
                    param_types[name] = type_hint
            return param_types

        def record_field_access(
            access_list: list[tuple[str, str]],
            field_id: Optional[str],
            current_class: Optional[str],
            current_caller: Optional[str],
        ) -> None:
            source_id = source_for_access(current_class, current_caller)
            if field_id and source_id:
                access_list.append((source_id, field_id))

        def property_key_text(pair_node: Node) -> Optional[str]:
            key_node = pair_node.child_by_field_name("key") or pair_node.child_by_field_name(
                "property"
            )
            if not key_node:
                key_node = next(
                    (
                        child
                        for child in pair_node.children
                        if child.type in (
                            "property_identifier",
                            "identifier",
                            "string",
                            "template_string",
                        )
                    ),
                    None,
                )
            if not key_node:
                return None
            if key_node.type == "string":
                return self._string_literal_value(source, key_node)
            if key_node.type == "template_string":
                raw = self._node_text(source, key_node)
                if raw.startswith("`") and raw.endswith("`"):
                    raw = raw[1:-1]
                return raw
            return self._node_text(source, key_node)

        def to_pascal_case(value: str) -> str:
            spaced = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", value)
            spaced = re.sub(r"[^A-Za-z0-9]+", " ", spaced)
            parts = [part for part in spaced.split() if part]
            return "".join(part[:1].upper() + part[1:] for part in parts)

        def singularize(value: str) -> str:
            lowered = value.lower()
            if lowered.endswith("ies") and len(value) > 3:
                return value[:-3] + "y"
            if lowered.endswith("ses") and len(value) > 3:
                return value[:-2]
            if lowered.endswith("s") and not lowered.endswith("ss") and len(value) > 2:
                return value[:-1]
            return value

        def type_candidates_for_hint(value: str) -> list[str]:
            if not value:
                return []
            candidates = []
            singular = singularize(value)
            for candidate in (singular, value):
                name = to_pascal_case(candidate)
                if not name or name in candidates:
                    continue
                candidates.append(name)
            return candidates

        def infer_object_type(
            property_hint: Optional[str],
            property_keys: list[str],
        ) -> Optional[TypeHint]:
            if not property_hint or not property_keys:
                return None
            best_candidate = None
            best_matches = 0
            for candidate in type_candidates_for_hint(property_hint):
                match_count = 0
                for key in property_keys:
                    if not key:
                        continue
                    field_id = self.symbol_index.resolve_field_in_type(file_path, candidate, key)
                    if not field_id:
                        field_id = self.symbol_index.resolve_field_global(candidate, key)
                    if field_id:
                        match_count += 1
                if match_count > best_matches:
                    best_candidate = candidate
                    best_matches = match_count
            if best_candidate and best_matches > 0:
                return TypeHint(kind="class", name=best_candidate)
            return None

        def parse_call_schema(node: Node) -> Optional[TypeHint]:
            if node.type != "call_expression":
                return None
            fn_node = node.child_by_field_name("function") or (node.children[0] if node.children else None)
            if not fn_node or fn_node.type != "member_expression":
                return None
            prop_node = fn_node.child_by_field_name("property")
            obj_node = fn_node.child_by_field_name("object")
            if not prop_node or not obj_node:
                return None
            prop_name = self._node_text(source, prop_node)
            if prop_name not in ("parse", "safeParse", "parseAsync", "safeParseAsync"):
                return None
            schema_ref = self._resolve_ts_schema_ref(
                obj_node,
                source,
                file_path,
                import_map,
                import_module_map,
                local_zod_schemas,
            )
            if not schema_ref:
                return None
            schema_name, schema_file_path = schema_ref
            return TypeHint(kind="zod_schema", schema_name=schema_name, schema_file_path=schema_file_path)

        def visit(
            node: Node,
            current_class: Optional[str],
            current_caller: Optional[str],
            object_type: Optional[TypeHint] = None,
            property_hint: Optional[str] = None,
        ) -> None:
            if node.type == "class_declaration":
                name_node = node.child_by_field_name("name")
                class_name = self._node_text(source, name_node) if name_node else None
                for child in node.children:
                    visit(child, class_name, current_caller)
                return

            if node.type == "function_declaration":
                name_node = node.child_by_field_name("name")
                fn_name = self._node_text(source, name_node) if name_node else None
                caller_id = (
                    self.symbol_index.resolve_function_in_file(file_path, fn_name)
                    if fn_name
                    else None
                )
                params_node = node.child_by_field_name("parameters")
                type_scopes.append(collect_param_types(params_node))
                return_type_scopes.append(extract_return_type(node))
                for child in node.children:
                    visit(child, current_class, caller_id)
                return_type_scopes.pop()
                type_scopes.pop()
                return

            if node.type == "method_definition":
                name_node = node.child_by_field_name("name")
                method_name = self._node_text(source, name_node) if name_node else None
                caller_id = (
                    self.symbol_index.resolve_method_in_class(file_path, current_class, method_name)
                    if current_class and method_name
                    else None
                )
                params_node = node.child_by_field_name("parameters")
                type_scopes.append(collect_param_types(params_node))
                return_type_scopes.append(extract_return_type(node))
                for child in node.children:
                    visit(child, current_class, caller_id)
                return_type_scopes.pop()
                type_scopes.pop()
                return

            if node.type in ("arrow_function", "function_expression"):
                params_node = node.child_by_field_name("parameters")
                type_scopes.append(collect_param_types(params_node))
                return_type = extract_return_type(node)
                return_type_scopes.append(return_type)
                body_node = node.child_by_field_name("body")
                body_expr = self._unwrap_parenthesized_expression(body_node)
                if body_expr and body_expr.type in ("object", "object_literal") and return_type:
                    visit(body_expr, current_class, current_caller, object_type=return_type)
                else:
                    for child in node.children:
                        visit(child, current_class, current_caller)
                return_type_scopes.pop()
                type_scopes.pop()
                return

            if node.type == "variable_declarator":
                name_node = node.child_by_field_name("name")
                value_node = node.child_by_field_name("value")
                type_node = node.child_by_field_name("type") or node.child_by_field_name("type_annotation")
                type_hint = None
                if name_node:
                    var_name = self._node_text(source, name_node)
                    type_hint = self._ts_type_hint(
                        type_node,
                        source,
                        file_path,
                        import_map,
                        import_module_map,
                        zod_aliases,
                        local_zod_schemas,
                    )
                    if not type_hint:
                        inferred_name = self._ts_type_from_value(value_node, source)
                        if inferred_name:
                            type_hint = TypeHint(kind="class", name=inferred_name)
                    register_type(var_name, type_hint)
                if name_node and value_node and value_node.type == "arrow_function":
                    fn_name = self._node_text(source, name_node)
                    caller_id = self.symbol_index.resolve_function_in_file(file_path, fn_name)
                    params_node = value_node.child_by_field_name("parameters")
                    type_scopes.append(collect_param_types(params_node))
                    return_type_scopes.append(extract_return_type(value_node))
                    visit(value_node, current_class, caller_id)
                    return_type_scopes.pop()
                    type_scopes.pop()
                elif value_node and value_node.type in ("object", "object_literal") and type_hint:
                    visit(value_node, current_class, current_caller, object_type=type_hint)
                else:
                    for child in node.children:
                        visit(child, current_class, current_caller)
                return

            if node.type == "assignment_expression":
                left_node = node.child_by_field_name("left")
                right_node = node.child_by_field_name("right")
                if left_node and left_node.type == "member_expression":
                    field_id = self._resolve_ts_field_access(
                        file_path, source, left_node, current_class, type_scopes
                    )
                    record_field_access(field_writes, field_id, current_class, current_caller)
                if (
                    left_node
                    and left_node.type == "identifier"
                    and right_node
                    and right_node.type in ("object", "object_literal")
                ):
                    var_name = self._node_text(source, left_node)
                    hinted_type = self._resolve_ts_type_hint(var_name, type_scopes)
                    if hinted_type:
                        visit(right_node, current_class, current_caller, object_type=hinted_type)
                        return
                if right_node:
                    visit(right_node, current_class, current_caller)
                return

            if node.type in ("satisfies_expression", "as_expression", "type_assertion"):
                expr_node = node.child_by_field_name("expression") or node.child_by_field_name(
                    "left"
                )
                type_node = node.child_by_field_name("type") or node.child_by_field_name("right")
                if not expr_node or not type_node:
                    named_children = [child for child in node.children if child.is_named]
                    if not expr_node and named_children:
                        expr_node = named_children[0]
                    if not type_node and len(named_children) > 1:
                        type_node = named_children[-1]
                type_hint = self._ts_type_hint(
                    type_node,
                    source,
                    file_path,
                    import_map,
                    import_module_map,
                    zod_aliases,
                    local_zod_schemas,
                )
                if expr_node and expr_node.type in ("object", "object_literal") and type_hint:
                    visit(expr_node, current_class, current_caller, object_type=type_hint)
                    return
                if expr_node:
                    visit(expr_node, current_class, current_caller)
                return

            if node.type == "return_statement":
                value_node = node.child_by_field_name("argument") or node.child_by_field_name(
                    "expression"
                )
                if not value_node:
                    value_node = next((child for child in node.children if child.is_named), None)
                if value_node and value_node.type in ("object", "object_literal"):
                    return_type = current_return_type()
                    if return_type:
                        visit(value_node, current_class, current_caller, object_type=return_type)
                        return
                if value_node:
                    visit(value_node, current_class, current_caller)
                return

            if node.type in ("object", "object_literal"):
                pairs = [child for child in node.children if child.type == "pair"]
                property_keys = [
                    key
                    for key in (property_key_text(pair) for pair in pairs)
                    if key
                ]
                resolved_type = object_type or infer_object_type(property_hint, property_keys)
                if resolved_type:
                    if (
                        resolved_type.kind == "zod_schema"
                        and resolved_type.schema_name
                        and resolved_type.schema_file_path
                    ):
                        for prop_key in property_keys:
                            schema_id = self._schema_field_id(
                                schema_type="zod",
                                file_path=resolved_type.schema_file_path,
                                schema_name=resolved_type.schema_name,
                                field_name=prop_key,
                            )
                            record_field_access(field_writes, schema_id, current_class, current_caller)
                    elif resolved_type.kind == "class" and resolved_type.name:
                        for prop_key in property_keys:
                            field_id = self.symbol_index.resolve_field_in_type(
                                file_path, resolved_type.name, prop_key
                            )
                            if not field_id:
                                field_id = self.symbol_index.resolve_field_global(
                                    resolved_type.name, prop_key
                                )
                            record_field_access(field_writes, field_id, current_class, current_caller)
                for pair in pairs:
                    value_node = pair.child_by_field_name("value")
                    if not value_node:
                        continue
                    prop_name = property_key_text(pair)
                    visit(
                        value_node,
                        current_class,
                        current_caller,
                        object_type=None,
                        property_hint=prop_name,
                    )
                for child in node.children:
                    if child.type == "pair" or not child.is_named:
                        continue
                    visit(
                        child,
                        current_class,
                        current_caller,
                        object_type=object_type,
                        property_hint=property_hint,
                    )
                return

            if node.type == "array":
                for child in node.children:
                    visit(
                        child,
                        current_class,
                        current_caller,
                        object_type=object_type,
                        property_hint=property_hint,
                    )
                return

            if node.type == "update_expression":
                member_node = next(
                    (child for child in node.children if child.type == "member_expression"),
                    None,
                )
                if member_node:
                    field_id = self._resolve_ts_field_access(
                        file_path, source, member_node, current_class, type_scopes
                    )
                    record_field_access(field_reads, field_id, current_class, current_caller)
                    record_field_access(field_writes, field_id, current_class, current_caller)
                return

            if node.type == "call_expression" and current_caller:
                args_node = node.child_by_field_name("arguments")
                schema_hint = parse_call_schema(node)
                handled_arg = None
                if schema_hint and args_node:
                    first_arg = next((child for child in args_node.children if child.is_named), None)
                    if first_arg and first_arg.type in ("object", "object_literal"):
                        handled_arg = first_arg
                        visit(first_arg, current_class, current_caller, object_type=schema_hint)
                target_id = self._resolve_ts_call(
                    file_path,
                    source,
                    node,
                    current_class,
                    import_map,
                )
                if target_id:
                    call_edges.append((current_caller, target_id))
                if args_node:
                    for child in args_node.children:
                        if handled_arg is not None and child == handled_arg:
                            continue
                        visit(child, current_class, current_caller)
                return

            if node.type == "member_expression":
                field_id = self._resolve_ts_field_access(
                    file_path, source, node, current_class, type_scopes
                )
                record_field_access(field_reads, field_id, current_class, current_caller)
                return

            for child in node.children:
                visit(child, current_class, current_caller)

        visit(root, None, None)
        schema_fields, schema_expose_edges = self._extract_ts_schema_exposures(
            file_path,
            source,
            root,
        )
        path_literals = self._extract_ts_path_literals(
            file_path,
            source,
            root,
        )
        return DeterministicEdges(
            call_edges=call_edges,
            import_targets=import_targets,
            field_reads=field_reads,
            field_writes=field_writes,
            schema_fields=schema_fields,
            schema_expose_edges=schema_expose_edges,
            path_literals=path_literals,
        )

    def _collect_ts_imports(
        self,
        file_path: Path,
        source: bytes,
        root: Node,
    ) -> tuple[
        dict[str, dict[str, tuple[Path, str]]],
        list[Path],
        dict[str, str],
    ]:
        import_map: dict[str, dict[str, tuple[Path, str]]] = {
            "named": {},
            "namespace": {},
            "default": {},
        }
        import_targets: list[Path] = []
        import_module_map: dict[str, str] = {}

        for node in root.children:
            if node.type != "import_statement":
                continue
            module_node = node.child_by_field_name("source")
            module_spec = self._string_literal_value(source, module_node) if module_node else None
            target_path = self._resolve_ts_module(file_path, module_spec) if module_spec else None
            if target_path:
                import_targets.append(target_path)

            import_clause = next((c for c in node.children if c.type == "import_clause"), None)
            if not import_clause:
                continue

            for child in import_clause.children:
                if child.type == "identifier":
                    local_name = self._node_text(source, child)
                    if module_spec:
                        import_module_map[local_name] = module_spec
                    if target_path:
                        import_map["default"][local_name] = (target_path, local_name)
                elif child.type == "named_imports":
                    for spec in child.children:
                        if spec.type != "import_specifier":
                            continue
                        identifiers = [c for c in spec.children if c.type == "identifier"]
                        if not identifiers:
                            continue
                        if len(identifiers) == 1:
                            imported = identifiers[0]
                            local = identifiers[0]
                        else:
                            imported, local = identifiers[0], identifiers[-1]
                        imported_name = self._node_text(source, imported)
                        local_name = self._node_text(source, local)
                        if module_spec:
                            import_module_map[local_name] = module_spec
                        if target_path:
                            import_map["named"][local_name] = (target_path, imported_name)
                elif child.type == "namespace_import":
                    ident = next((c for c in child.children if c.type == "identifier"), None)
                    if ident:
                        alias = self._node_text(source, ident)
                        if module_spec:
                            import_module_map[alias] = module_spec
                        if target_path:
                            import_map["namespace"][alias] = (target_path, alias)

        return import_map, import_targets, import_module_map

    def _resolve_ts_module(self, file_path: Path, module_spec: Optional[str]) -> Optional[Path]:
        if not module_spec or not module_spec.startswith("."):
            return None
        base = (file_path.parent / module_spec).resolve()

        candidates = []
        if base.suffix:
            candidates.append(base)
        else:
            candidates.extend([
                base.with_suffix(".ts"),
                base.with_suffix(".tsx"),
                base.with_suffix(".js"),
                base.with_suffix(".jsx"),
                base.with_suffix(".d.ts"),
            ])
            candidates.extend([
                base / "index.ts",
                base / "index.tsx",
                base / "index.js",
                base / "index.jsx",
                base / "index.d.ts",
            ])

        for candidate in candidates:
            if candidate.exists():
                try:
                    candidate.relative_to(self.root_path)
                    return candidate
                except ValueError:
                    continue
        return None

    def _resolve_ts_call(
        self,
        file_path: Path,
        source: bytes,
        node: Node,
        current_class: Optional[str],
        import_map: dict[str, dict[str, tuple[Path, str]]],
    ) -> Optional[str]:
        fn_node = node.child_by_field_name("function") or (node.children[0] if node.children else None)
        if not fn_node:
            return None

        if fn_node.type == "identifier":
            name = self._node_text(source, fn_node)
            target = self.symbol_index.resolve_function_in_file(file_path, name)
            if target:
                return target
            if name in import_map["named"]:
                target_path, imported = import_map["named"][name]
                return self.symbol_index.resolve_symbol_in_file(target_path, imported)
            return None

        if fn_node.type == "member_expression":
            obj = fn_node.child_by_field_name("object")
            prop = fn_node.child_by_field_name("property")
            if not obj or not prop:
                return None
            obj_name = self._node_text(source, obj)
            prop_name = self._node_text(source, prop)

            if obj_name == "this" and current_class:
                return self.symbol_index.resolve_method_in_class(file_path, current_class, prop_name)

            if obj_name in import_map["namespace"]:
                target_path, _ = import_map["namespace"][obj_name]
                return self.symbol_index.resolve_symbol_in_file(target_path, prop_name)

            if current_class and obj_name == current_class:
                return self.symbol_index.resolve_method_in_class(file_path, current_class, prop_name)

        return None

    def _resolve_ts_field_access(
        self,
        file_path: Path,
        source: bytes,
        node: Node,
        current_class: Optional[str],
        type_scopes: list[dict[str, TypeHint]],
    ) -> Optional[str]:
        if node.type != "member_expression":
            return None

        obj = node.child_by_field_name("object")
        prop = node.child_by_field_name("property")
        if not obj or not prop:
            return None

        prop_name = self._node_text(source, prop)
        if not prop_name:
            return None

        if obj.type == "this" and current_class:
            return self.symbol_index.resolve_field_in_type(file_path, current_class, prop_name)

        if obj.type == "identifier":
            obj_name = self._node_text(source, obj)
            if obj_name:
                hinted_type = self._resolve_ts_type_hint(obj_name, type_scopes)
                if hinted_type:
                    if hinted_type.kind == "zod_schema":
                        if hinted_type.schema_name and hinted_type.schema_file_path:
                            return self._schema_field_id(
                                schema_type="zod",
                                file_path=hinted_type.schema_file_path,
                                schema_name=hinted_type.schema_name,
                                field_name=prop_name,
                            )
                    elif hinted_type.kind == "class" and hinted_type.name:
                        target = self.symbol_index.resolve_field_in_type(
                            file_path, hinted_type.name, prop_name
                        )
                        if not target:
                            target = self.symbol_index.resolve_field_global(
                                hinted_type.name, prop_name
                            )
                        if target:
                            return target

                target = self.symbol_index.resolve_field_in_type(file_path, obj_name, prop_name)
                if target:
                    return target
                target = self.symbol_index.resolve_field_global(obj_name, prop_name)
                if target:
                    return target
                if current_class and obj_name == current_class:
                    return self.symbol_index.resolve_field_in_type(file_path, current_class, prop_name)

        return None

    def _resolve_ts_type_hint(
        self,
        name: str,
        type_scopes: list[dict[str, TypeHint]],
    ) -> Optional[TypeHint]:
        for scope in reversed(type_scopes):
            type_hint = scope.get(name)
            if type_hint:
                return type_hint
        return None

    def _ts_type_hint(
        self,
        node: Optional[Node],
        source: bytes,
        file_path: Path,
        import_map: dict[str, dict[str, tuple[Path, str]]],
        import_module_map: dict[str, str],
        zod_aliases: set[str],
        local_zod_schemas: set[str],
    ) -> Optional[TypeHint]:
        if not node:
            return None
        if node.type == "type_annotation":
            inner = node.child_by_field_name("type")
            if not inner:
                inner = next((child for child in node.children if child.is_named), None)
            return self._ts_type_hint(
                inner,
                source,
                file_path,
                import_map,
                import_module_map,
                zod_aliases,
                local_zod_schemas,
            )
        if node.type in ("union_type", "intersection_type", "parenthesized_type"):
            for child in node.children:
                if not child.is_named:
                    continue
                hint = self._ts_type_hint(
                    child,
                    source,
                    file_path,
                    import_map,
                    import_module_map,
                    zod_aliases,
                    local_zod_schemas,
                )
                if hint:
                    return hint
            return None
        if node.type == "generic_type":
            schema_ref = self._ts_zod_schema_reference(
                node,
                source,
                file_path,
                import_map,
                import_module_map,
                zod_aliases,
                local_zod_schemas,
            )
            if schema_ref:
                schema_name, schema_file_path = schema_ref
                return TypeHint(
                    kind="zod_schema",
                    schema_name=schema_name,
                    schema_file_path=schema_file_path,
                )
        type_name = self._ts_type_name(node, source)
        if type_name:
            return TypeHint(kind="class", name=type_name)
        return None

    def _ts_type_name(self, node: Optional[Node], source: bytes) -> Optional[str]:
        if not node:
            return None
        if node.type == "type_annotation":
            inner = node.child_by_field_name("type")
            if not inner:
                inner = next((child for child in node.children if child.is_named), None)
            return self._ts_type_name(inner, source)
        if node.type in ("type_identifier", "identifier"):
            return self._node_text(source, node)
        if node.type == "generic_type":
            name_node = node.child_by_field_name("name")
            if name_node:
                return self._node_text(source, name_node)
        if node.type in ("qualified_name", "qualified_type_identifier", "nested_type_identifier"):
            right = (
                node.child_by_field_name("right")
                or node.child_by_field_name("name")
                or (node.children[-1] if node.children else None)
            )
            return self._node_text(source, right) if right else None
        if node.type == "array_type":
            element = node.child_by_field_name("element")
            return self._ts_type_name(element, source)
        if node.type in ("union_type", "intersection_type", "parenthesized_type"):
            for child in node.children:
                name = self._ts_type_name(child, source)
                if name:
                    return name
        if node.type == "predefined_type":
            return self._node_text(source, node)
        return None

    def _ts_zod_schema_reference(
        self,
        node: Node,
        source: bytes,
        file_path: Path,
        import_map: dict[str, dict[str, tuple[Path, str]]],
        import_module_map: dict[str, str],
        zod_aliases: set[str],
        local_zod_schemas: set[str],
    ) -> Optional[tuple[str, Path]]:
        name_node = node.child_by_field_name("name")
        if not name_node:
            return None
        if not self._is_zod_infer_name(name_node, source, zod_aliases):
            return None

        args_node = node.child_by_field_name("type_arguments") or node.child_by_field_name(
            "type_parameters"
        )
        if not args_node:
            return None
        first_arg = next((child for child in args_node.children if child.is_named), None)
        if not first_arg:
            return None
        schema_node = self._unwrap_ts_type_query(first_arg)
        if not schema_node:
            return None
        return self._resolve_ts_schema_ref(
            schema_node,
            source,
            file_path,
            import_map,
            import_module_map,
            local_zod_schemas,
        )

    def _unwrap_ts_type_query(self, node: Node) -> Optional[Node]:
        if node.type != "type_query":
            return node
        expr = node.child_by_field_name("expr_name") or node.child_by_field_name("name")
        if expr:
            return expr
        return next((child for child in node.children if child.is_named), None)

    def _unwrap_parenthesized_expression(self, node: Optional[Node]) -> Optional[Node]:
        if not node:
            return None
        if node.type != "parenthesized_expression":
            return node
        return next((child for child in node.children if child.is_named), None)

    def _resolve_ts_schema_ref(
        self,
        node: Node,
        source: bytes,
        file_path: Path,
        import_map: dict[str, dict[str, tuple[Path, str]]],
        import_module_map: dict[str, str],
        local_zod_schemas: set[str],
    ) -> Optional[tuple[str, Path]]:
        if node.type in ("identifier", "type_identifier"):
            schema_name = self._node_text(source, node)
            if not schema_name:
                return None
            if schema_name in local_zod_schemas:
                return schema_name, file_path
            if schema_name in import_map["named"]:
                target_path, imported_name = import_map["named"][schema_name]
                return imported_name, target_path
            if schema_name in import_map["default"]:
                target_path, imported_name = import_map["default"][schema_name]
                return imported_name, target_path
            candidates = self._global_zod_schema_map.get(schema_name)
            if candidates:
                module_spec = import_module_map.get(schema_name)
                resolved_path = self._choose_schema_path(candidates, module_spec)
                if resolved_path:
                    return schema_name, resolved_path
            return None

        if node.type in (
            "qualified_name",
            "member_expression",
            "qualified_type_identifier",
            "nested_type_identifier",
        ):
            obj_name, prop_name = self._qualified_name_parts(node, source)
            if not obj_name or not prop_name:
                return None
            if obj_name in import_map["namespace"]:
                target_path, _ = import_map["namespace"][obj_name]
                return prop_name, target_path
            candidates = self._global_zod_schema_map.get(prop_name)
            if candidates:
                module_spec = import_module_map.get(obj_name)
                resolved_path = self._choose_schema_path(candidates, module_spec)
                if resolved_path:
                    return prop_name, resolved_path
        return None

    def _choose_schema_path(
        self,
        candidates: set[Path],
        module_spec: Optional[str],
    ) -> Optional[Path]:
        if not candidates:
            return None

        candidate_list = list(candidates)
        filtered = candidate_list
        if module_spec:
            module_root = self._module_root_for_spec(module_spec)
            if module_root:
                root = module_root if module_root.is_dir() else module_root.parent
                filtered = [path for path in candidate_list if self._path_is_within(path, root)]
                if not filtered:
                    filtered = candidate_list

        def score(path: Path) -> tuple[int, int]:
            posix = path.as_posix()
            rank = 0
            if "/src/" in posix:
                rank += 2
            if "/dist/" in posix or "/build/" in posix or posix.endswith(".d.ts"):
                rank -= 2
            return (rank, -len(posix))

        filtered.sort(key=score, reverse=True)
        return filtered[0] if filtered else None

    def _module_root_for_spec(self, module_spec: str) -> Optional[Path]:
        if not module_spec:
            return None
        for package_name, package_root in self._package_map.items():
            if module_spec == package_name:
                return package_root
            prefix = f"{package_name}/"
            if module_spec.startswith(prefix):
                subpath = module_spec[len(prefix) :]
                return (package_root / subpath).resolve()
        return None

    def _path_is_within(self, path: Path, root: Path) -> bool:
        try:
            path.relative_to(root)
            return True
        except ValueError:
            return False

    def _qualified_name_parts(
        self,
        node: Node,
        source: bytes,
    ) -> tuple[Optional[str], Optional[str]]:
        obj_node = (
            node.child_by_field_name("left")
            or node.child_by_field_name("object")
            or node.child_by_field_name("qualifier")
            or node.child_by_field_name("module")
        )
        prop_node = (
            node.child_by_field_name("right")
            or node.child_by_field_name("property")
            or node.child_by_field_name("name")
        )
        if not obj_node or not prop_node:
            return None, None
        return self._node_text(source, obj_node), self._node_text(source, prop_node)

    def _is_zod_infer_name(
        self,
        node: Node,
        source: bytes,
        aliases: set[str],
    ) -> bool:
        obj_name = None
        prop_name = None
        if node.type in (
            "qualified_name",
            "member_expression",
            "qualified_type_identifier",
            "nested_type_identifier",
        ):
            obj_name, prop_name = self._qualified_name_parts(node, source)
        else:
            name_text = self._node_text(source, node)
            if name_text and "." in name_text:
                obj_name, prop_name = name_text.split(".", 1)
        if not obj_name or not prop_name:
            return False
        return obj_name in aliases and prop_name in ("infer", "input", "output")

    def _ts_type_from_value(self, node: Optional[Node], source: bytes) -> Optional[str]:
        if not node:
            return None
        if node.type == "new_expression":
            ctor = node.child_by_field_name("constructor")
            if ctor:
                if ctor.type == "identifier":
                    return self._node_text(source, ctor)
                if ctor.type == "member_expression":
                    prop = ctor.child_by_field_name("property")
                    if prop:
                        return self._node_text(source, prop)
            fallback = next(
                (child for child in node.children if child.type == "identifier"),
                None,
            )
            return self._node_text(source, fallback) if fallback else None
        return None

    def _extract_ts_schema_exposures(
        self,
        file_path: Path,
        source: bytes,
        root: Node,
    ) -> tuple[list[SchemaFieldDefinition], list[tuple[str, str]]]:
        schema_fields: list[SchemaFieldDefinition] = []
        schema_edges: list[tuple[str, str]] = []
        seen_schema_ids: set[str] = set()

        # GraphQL: @Field decorators on class fields
        for symbol_ref in self.symbol_index.symbols_by_file.get(Path(file_path), []):
            symbol = symbol_ref.symbol
            if symbol.symbol_type != SymbolType.FIELD:
                continue
            if not symbol.decorators:
                continue
            if not any(self._decorator_base_name(dec.name) == "Field" for dec in symbol.decorators):
                continue

            schema_name = symbol.parent_name or ""
            graphql_nullable, graphql_type = self._graphql_field_metadata(symbol)
            schema_id = self._schema_field_id(
                schema_type="graphql",
                file_path=file_path,
                schema_name=schema_name,
                field_name=symbol.name,
            )
            if schema_id not in seen_schema_ids:
                seen_schema_ids.add(schema_id)
                schema_fields.append(
                    SchemaFieldDefinition(
                        schema_id=schema_id,
                        name=symbol.name,
                        schema_type="graphql",
                        schema_name=schema_name,
                        file_path=file_path,
                        line_start=symbol.line_start,
                        line_end=symbol.line_end,
                        nullable=graphql_nullable,
                        field_type=graphql_type,
                    )
                )
            schema_edges.append((symbol_ref.symbol_id, schema_id))

        # DTO validators: class-validator style decorators
        for symbol_ref in self.symbol_index.symbols_by_file.get(Path(file_path), []):
            symbol = symbol_ref.symbol
            if symbol.symbol_type != SymbolType.FIELD:
                continue
            if not symbol.decorators:
                continue

            dto_metadata = self._dto_field_metadata(symbol.decorators)
            if not dto_metadata:
                continue

            dto_nullable, dto_type = dto_metadata
            schema_name = symbol.parent_name or ""
            schema_id = self._schema_field_id(
                schema_type="dto",
                file_path=file_path,
                schema_name=schema_name,
                field_name=symbol.name,
            )
            if schema_id not in seen_schema_ids:
                seen_schema_ids.add(schema_id)
                schema_fields.append(
                    SchemaFieldDefinition(
                        schema_id=schema_id,
                        name=symbol.name,
                        schema_type="dto",
                        schema_name=schema_name,
                        file_path=file_path,
                        line_start=symbol.line_start,
                        line_end=symbol.line_end,
                        nullable=dto_nullable,
                        field_type=dto_type,
                    )
                )
            schema_edges.append((symbol_ref.symbol_id, schema_id))

        # Zod: z.object({ ... }) shapes
        zod_aliases = self._collect_ts_zod_aliases(source, root)
        if not zod_aliases:
            zod_aliases.add("z")

        def pair_key(pair: Node) -> Optional[str]:
            key_node = pair.child_by_field_name("key") or pair.child_by_field_name("property")
            if not key_node:
                key_node = next(
                    (
                        child
                        for child in pair.children
                        if child.type in ("property_identifier", "string")
                    ),
                    None,
                )
            if not key_node:
                return None
            return self._property_key_text(source, key_node)

        def collect_zod_fields(shape: Node, prefix: str = "") -> list[tuple[str, Node, Optional[Node]]]:
            results: list[tuple[str, Node, Optional[Node]]] = []
            for pair in shape.children:
                if pair.type != "pair":
                    continue
                field_key = pair_key(pair)
                if not field_key:
                    continue
                field_path = f"{prefix}.{field_key}" if prefix else field_key
                value_node = pair.child_by_field_name("value")
                results.append((field_path, pair, value_node))

                nested_shape = self._extract_zod_nested_shape(value_node, source, zod_aliases)
                if nested_shape:
                    results.extend(collect_zod_fields(nested_shape, field_path))
            return results

        def visit(node: Node) -> None:
            if node.type == "variable_declarator":
                name_node = node.child_by_field_name("name")
                value_node = node.child_by_field_name("value")
                base_node = (
                    self._unwrap_zod_call(value_node, source, zod_aliases)
                    if value_node
                    else None
                )
                if name_node and base_node and self._is_zod_object_call(
                    base_node,
                    source,
                    zod_aliases,
                ):
                    schema_name = self._node_text(source, name_node)
                    schema_base = self._schema_base_name(schema_name)
                    shape_node = self._extract_zod_shape_node(base_node)
                    if shape_node:
                        for field_name, pair, value_node in collect_zod_fields(shape_node):
                            field_type, nullable = self._zod_field_metadata(
                                value_node,
                                source,
                                zod_aliases,
                            )
                            schema_id = self._schema_field_id(
                                schema_type="zod",
                                file_path=file_path,
                                schema_name=schema_name,
                                field_name=field_name,
                            )
                            if schema_id not in seen_schema_ids:
                                seen_schema_ids.add(schema_id)
                                schema_fields.append(
                                    SchemaFieldDefinition(
                                        schema_id=schema_id,
                                        name=field_name,
                                        schema_type="zod",
                                        schema_name=schema_name,
                                        file_path=file_path,
                                        line_start=pair.start_point[0] + 1,
                                        line_end=pair.end_point[0] + 1,
                                        nullable=nullable,
                                        field_type=field_type,
                                    )
                                )
                            if schema_base and "." not in field_name:
                                field_id = self.symbol_index.resolve_field_in_type(
                                    file_path,
                                    schema_base,
                                    field_name,
                                )
                                if not field_id:
                                    field_id = self.symbol_index.resolve_field_global(
                                        schema_base,
                                        field_name,
                                    )
                                if field_id:
                                    schema_edges.append((field_id, schema_id))

                return

            for child in node.children:
                visit(child)

        visit(root)
        return schema_fields, schema_edges

    def _extract_ts_path_literals(
        self,
        file_path: Path,
        source: bytes,
        root: Node,
    ) -> list[PathLiteralDefinition]:
        path_literals: list[PathLiteralDefinition] = []
        seen_nodes: set[tuple[int, int]] = set()

        def record_literal(
            node: Node,
            raw_path: str,
            normalized_path: str,
            segments: list[str],
            confidence: str,
        ) -> None:
            if not raw_path or not segments:
                return
            key = (node.start_byte, node.end_byte)
            if key in seen_nodes:
                return
            seen_nodes.add(key)
            path_literals.append(
                PathLiteralDefinition(
                    path=raw_path,
                    normalized_path=normalized_path,
                    segments=segments,
                    leaf=segments[-1],
                    file_path=file_path,
                    line_start=node.start_point[0] + 1,
                    line_end=node.end_point[0] + 1,
                    confidence=confidence,
                    start_byte=node.start_byte,
                    end_byte=node.end_byte,
                )
            )

        def collect_literal(node: Node, confidence: str) -> None:
            literal_values = self._path_literal_values(source, node)
            if not literal_values:
                return
            raw_path, normalized_candidate, drop_wildcards = literal_values
            segments = self._parse_path_segments(
                normalized_candidate,
                drop_wildcards=drop_wildcards,
            )
            if not segments:
                return
            normalized_path = ".".join(segments)
            record_literal(node, raw_path, normalized_path, segments, confidence)

        def visit(node: Node) -> None:
            if node.type == "pair":
                key_node = node.child_by_field_name("key") or node.child_by_field_name("property")
                if not key_node:
                    key_node = next(
                        (child for child in node.children if child.type in ("string", "template_string")),
                        None,
                    )
                if key_node and key_node.type in ("string", "template_string"):
                    collect_literal(key_node, "high")

            if node.type in ("member_expression", "subscript_expression"):
                prop_node = node.child_by_field_name("property") or node.child_by_field_name("index")
                if prop_node and prop_node.type in ("string", "template_string"):
                    collect_literal(prop_node, "high")

            if node.type == "call_expression":
                args_node = node.child_by_field_name("arguments")
                if args_node:
                    for child in args_node.children:
                        if child.type in ("string", "template_string"):
                            collect_literal(child, "medium")

            for child in node.children:
                visit(child)

        def visit_low(node: Node) -> None:
            if node.type in ("string", "template_string"):
                collect_literal(node, "low")
            for child in node.children:
                visit_low(child)

        visit(root)
        visit_low(root)
        return path_literals

    def _path_literal_values(
        self,
        source: bytes,
        node: Node,
    ) -> Optional[tuple[str, str, bool]]:
        if node.type == "string":
            raw_value = self._string_literal_value(source, node)
            if not raw_value:
                return None
            if self._path_literal_guardrails(raw_value):
                return None
            return raw_value, raw_value, raw_value.startswith("$")

        if node.type == "template_string":
            raw_text = self._node_text(source, node)
            raw_value = (
                raw_text[1:-1]
                if raw_text.startswith("`") and raw_text.endswith("`")
                else raw_text
            )
            normalized = self._template_literal_normalized(source, node)
            if not normalized or self._path_literal_guardrails(normalized):
                return None
            if self._path_literal_guardrails(raw_value):
                return None
            return raw_value, normalized, raw_value.startswith("$")

        return None

    def _template_literal_normalized(self, source: bytes, node: Node) -> str:
        parts: list[str] = []
        for child in node.children:
            if child.type == "string_fragment":
                parts.append(self._node_text(source, child))
            elif child.type == "template_substitution":
                parts.append("*")
        return "".join(parts)

    def _path_literal_guardrails(self, value: str) -> bool:
        if len(value) > 200:
            return True
        if any(char.isspace() for char in value):
            return True
        if "://" in value or value.startswith(("http://", "https://", "www.")):
            return True
        if value.count(",") > 2:
            return True
        return False

    def _parse_path_segments(
        self,
        raw_path: str,
        drop_wildcards: bool = False,
    ) -> list[str]:
        if not raw_path:
            return []
        if "." not in raw_path and "[" not in raw_path:
            return []

        path = raw_path.strip()
        if path.startswith("$"):
            path = path[1:]
            if path.startswith("."):
                path = path[1:]
            drop_wildcards = True

        segments: list[str] = []
        current = ""
        idx = 0

        while idx < len(path):
            char = path[idx]
            if char == ".":
                if current:
                    segments.append(current)
                    current = ""
                idx += 1
                continue
            if char == "[":
                if current:
                    segments.append(current)
                    current = ""
                end = path.find("]", idx + 1)
                if end == -1:
                    return []
                content = path[idx + 1 : end].strip()
                if content.startswith(("'", '"')) and content.endswith(("'", '"')) and len(content) >= 2:
                    content = content[1:-1]
                if content in ("", "*"):
                    if not drop_wildcards:
                        segments.append("*")
                else:
                    segments.append(content)
                idx = end + 1
                continue

            current += char
            idx += 1

        if current:
            segments.append(current)

        segments = [segment for segment in segments if segment]
        if len(segments) < 2:
            return []

        for segment in segments:
            if not re.match(r"^[A-Za-z0-9_*$-]+$", segment):
                return []

        return segments

    def _decorator_base_name(self, name: str) -> str:
        return name.split(".")[-1]

    def _graphql_field_metadata(self, symbol: Symbol) -> tuple[Optional[bool], Optional[str]]:
        for dec in symbol.decorators:
            if self._decorator_base_name(dec.name) != "Field":
                continue
            raw_text = dec.raw_text or ""
            return self._parse_nullable_option(raw_text), self._parse_graphql_type(raw_text)
        return None, None

    def _parse_nullable_option(self, raw_text: str) -> Optional[bool]:
        match = re.search(r"nullable\s*:\s*(true|false)", raw_text)
        if not match:
            return None
        return match.group(1) == "true"

    def _parse_graphql_type(self, raw_text: str) -> Optional[str]:
        match = re.search(r"=>\s*(\[[^\]]+\]|[A-Za-z_][A-Za-z0-9_]*)", raw_text)
        if not match:
            return None
        type_expr = match.group(1).strip()
        if type_expr.startswith("[") and type_expr.endswith("]"):
            inner = type_expr[1:-1].strip()
            if inner:
                return f"{inner}[]"
            return None
        return type_expr

    def _dto_field_metadata(
        self,
        decorators: list,
    ) -> Optional[tuple[Optional[bool], Optional[str]]]:
        if not decorators:
            return None

        type_by_decorator = {
            "IsString": "string",
            "IsInt": "number",
            "IsNumber": "number",
            "IsFloat": "number",
            "IsBoolean": "boolean",
            "IsDate": "date",
            "IsEmail": "email",
            "IsUUID": "uuid",
            "IsEnum": "enum",
            "IsObject": "object",
            "IsArray": "array",
        }
        optional_decorators = {"IsOptional"}
        required_decorators = {"IsDefined", "IsNotEmpty"}

        decorator_names = [self._decorator_base_name(dec.name) for dec in decorators]
        if not any(
            name in type_by_decorator
            or name in optional_decorators
            or name in required_decorators
            for name in decorator_names
        ):
            return None

        nullable: Optional[bool] = None
        if any(name in optional_decorators for name in decorator_names):
            nullable = True
        elif any(name in required_decorators for name in decorator_names):
            nullable = False

        base_type = next(
            (type_by_decorator[name] for name in decorator_names if name in type_by_decorator),
            None,
        )
        if "IsArray" in decorator_names and base_type and base_type != "array":
            field_type = f"{base_type}[]"
        else:
            field_type = base_type

        return nullable, field_type

    def _zod_field_metadata(
        self,
        value_node: Optional[Node],
        source: bytes,
        aliases: set[str],
    ) -> tuple[Optional[str], Optional[bool]]:
        if not value_node:
            return None, None

        optional = False
        nullable = False
        array_modifier = False
        current = value_node

        while current and current.type == "call_expression":
            fn_node = current.child_by_field_name("function") or (
                current.children[0] if current.children else None
            )
            if not fn_node or fn_node.type != "member_expression":
                break
            obj_node = fn_node.child_by_field_name("object")
            prop_node = fn_node.child_by_field_name("property")
            prop_name = self._node_text(source, prop_node) if prop_node else ""
            if prop_name in ("optional", "nullable", "nullish", "array"):
                if prop_name in ("optional", "nullish"):
                    optional = True
                if prop_name in ("nullable", "nullish"):
                    nullable = True
                if prop_name == "array":
                    array_modifier = True
                current = obj_node
                continue
            break

        base_type = self._zod_base_type_from_call(current, source, aliases)
        if not base_type:
            if optional or nullable:
                return None, True
            return None, None

        field_type = self._zod_type_from_base(base_type, current, source, aliases)
        if array_modifier:
            field_type = f"{field_type}[]" if field_type else "array"

        effective_nullable = True if (optional or nullable) else False
        return field_type, effective_nullable

    def _zod_base_type_from_call(
        self,
        node: Optional[Node],
        source: bytes,
        aliases: set[str],
    ) -> Optional[str]:
        if not node or node.type != "call_expression":
            return None
        fn_node = node.child_by_field_name("function") or (
            node.children[0] if node.children else None
        )
        if not fn_node or fn_node.type != "member_expression":
            return None
        obj_node = fn_node.child_by_field_name("object")
        prop_node = fn_node.child_by_field_name("property")
        if not obj_node or not prop_node:
            return None
        if obj_node.type != "identifier":
            return None
        obj_name = self._node_text(source, obj_node)
        if obj_name not in aliases:
            return None
        base_type = self._node_text(source, prop_node)
        return {
            "nativeEnum": "enum",
        }.get(base_type, base_type)

    def _zod_type_from_base(
        self,
        base_type: str,
        node: Optional[Node],
        source: bytes,
        aliases: set[str],
    ) -> Optional[str]:
        if base_type == "array":
            item_type = self._zod_array_item_type(node, source, aliases)
            if item_type:
                return f"{item_type}[]"
            return "array"
        return base_type

    def _zod_array_item_type(
        self,
        node: Optional[Node],
        source: bytes,
        aliases: set[str],
    ) -> Optional[str]:
        if not node or node.type != "call_expression":
            return None
        args = node.child_by_field_name("arguments")
        if not args:
            return None
        for child in args.children:
            if child.type != "call_expression":
                continue
            inner_type = self._zod_base_type_from_call(child, source, aliases)
            if inner_type:
                return self._zod_type_from_base(inner_type, child, source, aliases)
        return None

    def _collect_ts_zod_aliases(self, source: bytes, root: Node) -> set[str]:
        aliases: set[str] = set()
        for node in root.children:
            if node.type != "import_statement":
                continue
            module_node = node.child_by_field_name("source")
            module_spec = self._string_literal_value(source, module_node) if module_node else None
            if module_spec != "zod":
                continue
            import_clause = next((c for c in node.children if c.type == "import_clause"), None)
            if not import_clause:
                continue
            for child in import_clause.children:
                if child.type == "identifier":
                    aliases.add(self._node_text(source, child))
                elif child.type == "named_imports":
                    for spec in child.children:
                        if spec.type != "import_specifier":
                            continue
                        identifiers = [c for c in spec.children if c.type == "identifier"]
                        if identifiers:
                            aliases.add(self._node_text(source, identifiers[-1]))
                elif child.type == "namespace_import":
                    ident = next((c for c in child.children if c.type == "identifier"), None)
                    if ident:
                        aliases.add(self._node_text(source, ident))
        return aliases

    def _collect_ts_zod_schema_names(
        self,
        source: bytes,
        root: Node,
        aliases: set[str],
    ) -> set[str]:
        schema_names: set[str] = set()

        def visit(node: Node) -> None:
            if node.type == "variable_declarator":
                name_node = node.child_by_field_name("name")
                value_node = node.child_by_field_name("value")
                base_node = (
                    self._unwrap_zod_call(value_node, source, aliases)
                    if value_node
                    else None
                )
                if (
                    name_node
                    and base_node
                    and self._is_zod_object_call(base_node, source, aliases)
                ):
                    schema_name = self._node_text(source, name_node)
                    if schema_name:
                        schema_names.add(schema_name)
                return

            for child in node.children:
                if child.is_named:
                    visit(child)

        visit(root)
        return schema_names

    def _is_zod_object_call(self, node: Node, source: bytes, aliases: set[str]) -> bool:
        if node.type != "call_expression":
            return False
        fn_node = node.child_by_field_name("function") or (node.children[0] if node.children else None)
        if not fn_node or fn_node.type != "member_expression":
            return False
        obj_node = fn_node.child_by_field_name("object")
        prop_node = fn_node.child_by_field_name("property")
        if not obj_node or not prop_node:
            return False
        obj_name = self._node_text(source, obj_node)
        prop_name = self._node_text(source, prop_node)
        return prop_name == "object" and obj_name in aliases

    def _extract_zod_shape_node(self, node: Node) -> Optional[Node]:
        args = node.child_by_field_name("arguments")
        if not args:
            return None
        return next((child for child in args.children if child.type == "object"), None)

    def _extract_zod_nested_shape(
        self,
        node: Optional[Node],
        source: bytes,
        aliases: set[str],
    ) -> Optional[Node]:
        if not node:
            return None

        base = self._unwrap_zod_call(node, source, aliases)
        if not base or base.type != "call_expression":
            return None
        if self._is_zod_object_call(base, source, aliases):
            return self._extract_zod_shape_node(base)
        if self._is_zod_array_call(base, source, aliases):
            args = base.child_by_field_name("arguments")
            if args:
                for child in args.children:
                    if child.type == "call_expression" and self._is_zod_object_call(
                        child,
                        source,
                        aliases,
                    ):
                        return self._extract_zod_shape_node(child)

        fn_node = base.child_by_field_name("function") or (base.children[0] if base.children else None)
        if fn_node and fn_node.type == "member_expression":
            prop_node = fn_node.child_by_field_name("property")
            if prop_node and self._node_text(source, prop_node) == "array":
                obj_node = fn_node.child_by_field_name("object")
                if obj_node and obj_node.type == "call_expression":
                    if self._is_zod_object_call(obj_node, source, aliases):
                        return self._extract_zod_shape_node(obj_node)
        return None

    def _unwrap_zod_call(
        self,
        node: Optional[Node],
        source: bytes,
        aliases: set[str],
    ) -> Optional[Node]:
        current = node
        while current and current.type == "call_expression":
            fn_node = current.child_by_field_name("function") or (
                current.children[0] if current.children else None
            )
            if not fn_node or fn_node.type != "member_expression":
                break
            obj_node = fn_node.child_by_field_name("object")
            prop_node = fn_node.child_by_field_name("property")
            prop_name = self._node_text(source, prop_node) if prop_node else ""
            if prop_name in (
                "optional",
                "nullable",
                "nullish",
                "default",
                "refine",
                "superRefine",
                "transform",
                "pipe",
                "describe",
                "catch",
                "brand",
                "readonly",
                "openapi",
            ):
                current = obj_node
                continue
            break
        return current

    def _is_zod_array_call(self, node: Node, source: bytes, aliases: set[str]) -> bool:
        if node.type != "call_expression":
            return False
        fn_node = node.child_by_field_name("function") or (node.children[0] if node.children else None)
        if not fn_node or fn_node.type != "member_expression":
            return False
        obj_node = fn_node.child_by_field_name("object")
        prop_node = fn_node.child_by_field_name("property")
        if not obj_node or not prop_node:
            return False
        obj_name = self._node_text(source, obj_node)
        prop_name = self._node_text(source, prop_node)
        return prop_name == "array" and obj_name in aliases

    def _schema_base_name(self, schema_name: str) -> Optional[str]:
        if schema_name.endswith("Schema") and len(schema_name) > len("Schema"):
            return schema_name[: -len("Schema")]
        return None

    def _property_key_text(self, source: bytes, node: Node) -> str:
        if node.type == "string":
            return self._string_literal_value(source, node) or ""
        return self._node_text(source, node)

    def _schema_field_id(
        self,
        schema_type: str,
        file_path: Path,
        schema_name: str,
        field_name: str,
    ) -> str:
        return f"schema::{schema_type}:{file_path}:{schema_name}:{field_name}"

    # ---------------------------------------------------------------------
    # Java
    # ---------------------------------------------------------------------

    def _extract_java_edges(self, file_path: Path, content: str) -> DeterministicEdges:
        source = content.encode("utf-8")
        tree = self._java_parser.parse(source)
        root = tree.root_node

        import_map, import_targets = self._collect_java_imports(file_path, source, root)
        call_edges: list[tuple[str, str]] = []

        def visit(node: Node, current_class: Optional[str], current_caller: Optional[str]) -> None:
            if node.type in ("class_declaration", "interface_declaration", "enum_declaration"):
                name_node = node.child_by_field_name("name")
                class_name = self._node_text(source, name_node) if name_node else None
                for child in node.children:
                    visit(child, class_name, current_caller)
                return

            if node.type == "method_declaration":
                name_node = node.child_by_field_name("name")
                method_name = self._node_text(source, name_node) if name_node else None
                caller_id = (
                    self.symbol_index.resolve_method_in_class(file_path, current_class, method_name)
                    if current_class and method_name
                    else None
                )
                for child in node.children:
                    visit(child, current_class, caller_id)
                return

            if node.type == "method_invocation" and current_caller:
                target_id = self._resolve_java_call(
                    file_path,
                    source,
                    node,
                    current_class,
                    import_map,
                )
                if target_id:
                    call_edges.append((current_caller, target_id))

            for child in node.children:
                visit(child, current_class, current_caller)

        visit(root, None, None)
        return DeterministicEdges(call_edges=call_edges, import_targets=import_targets)

    def _collect_java_imports(
        self,
        file_path: Path,
        source: bytes,
        root: Node,
    ) -> tuple[dict[str, str], list[Path]]:
        import_map: dict[str, str] = {}
        import_targets: list[Path] = []

        for node in root.children:
            if node.type != "import_declaration":
                continue
            scoped = next((c for c in node.children if c.type == "scoped_identifier"), None)
            if not scoped:
                continue
            full_name = self._node_text(source, scoped)
            if full_name.endswith(".*"):
                continue
            simple_name = full_name.split(".")[-1]
            import_map[simple_name] = full_name
            target_path = self.symbol_index.resolve_java_class(full_name)
            if target_path:
                import_targets.append(target_path)

        return import_map, import_targets

    def _resolve_java_call(
        self,
        file_path: Path,
        source: bytes,
        node: Node,
        current_class: Optional[str],
        import_map: dict[str, str],
    ) -> Optional[str]:
        name_node = node.child_by_field_name("name")
        method_name = self._node_text(source, name_node) if name_node else None
        if not method_name:
            return None

        obj_node = node.child_by_field_name("object")
        if obj_node is None:
            if current_class:
                return self.symbol_index.resolve_method_in_class(file_path, current_class, method_name)
            return None

        obj_name = self._node_text(source, obj_node)
        if obj_name == "this" and current_class:
            return self.symbol_index.resolve_method_in_class(file_path, current_class, method_name)

        if current_class and obj_name == current_class:
            return self.symbol_index.resolve_method_in_class(file_path, current_class, method_name)

        if obj_name in import_map:
            full_name = import_map[obj_name]
            target_path = self.symbol_index.resolve_java_class(full_name)
            if not target_path:
                return None
            return self.symbol_index.resolve_method_in_class(target_path, obj_name, method_name)

        return None

    # ---------------------------------------------------------------------
    # Go
    # ---------------------------------------------------------------------

    def _extract_go_edges(self, file_path: Path, content: str) -> DeterministicEdges:
        source = content.encode("utf-8")
        tree = self._go_parser.parse(source)
        root = tree.root_node

        import_map, import_targets = self._collect_go_imports(file_path, source, root)
        call_edges: list[tuple[str, str]] = []

        def visit(node: Node, current_caller: Optional[str]) -> None:
            if node.type == "function_declaration":
                name_node = node.child_by_field_name("name")
                fn_name = self._node_text(source, name_node) if name_node else None
                caller_id = (
                    self.symbol_index.resolve_function_in_file(file_path, fn_name)
                    if fn_name
                    else None
                )
                for child in node.children:
                    visit(child, caller_id)
                return

            if node.type == "method_declaration":
                name_node = node.child_by_field_name("name")
                method_name = self._node_text(source, name_node) if name_node else None
                receiver_node = node.child_by_field_name("receiver")
                receiver_type = self._go_receiver_type(source, receiver_node) if receiver_node else None
                caller_id = (
                    self.symbol_index.resolve_method_in_class(file_path, receiver_type, method_name)
                    if receiver_type and method_name
                    else None
                )
                for child in node.children:
                    visit(child, caller_id)
                return

            if node.type == "call_expression" and current_caller:
                target_id = self._resolve_go_call(file_path, source, node, import_map)
                if target_id:
                    call_edges.append((current_caller, target_id))

            for child in node.children:
                visit(child, current_caller)

        visit(root, None)
        return DeterministicEdges(call_edges=call_edges, import_targets=import_targets)

    def _collect_go_imports(
        self,
        file_path: Path,
        source: bytes,
        root: Node,
    ) -> tuple[dict[str, str], list[Path]]:
        import_map: dict[str, str] = {}
        import_targets: list[Path] = []

        for node in root.children:
            if node.type != "import_declaration":
                continue
            for spec in node.children:
                if spec.type != "import_spec":
                    continue
                alias_node = next((c for c in spec.children if c.type == "identifier"), None)
                path_node = next(
                    (c for c in spec.children if c.type == "interpreted_string_literal"),
                    None,
                )
                if not path_node:
                    continue
                import_path = self._node_text(source, path_node).strip("\"")
                if alias_node:
                    alias = self._node_text(source, alias_node)
                else:
                    alias = import_path.split("/")[-1]
                if alias in (".", "_"):
                    continue
                import_map[alias] = import_path
                target_dir = self.symbol_index.resolve_go_import_dir(import_path)
                if target_dir and target_dir.exists():
                    target_file = self.symbol_index.resolve_go_package_file(target_dir)
                    if target_file:
                        import_targets.append(target_file)

        return import_map, import_targets

    def _resolve_go_call(
        self,
        file_path: Path,
        source: bytes,
        node: Node,
        import_map: dict[str, str],
    ) -> Optional[str]:
        fn_node = node.child_by_field_name("function") or (node.children[0] if node.children else None)
        if not fn_node:
            return None

        if fn_node.type == "identifier":
            name = self._node_text(source, fn_node)
            return self.symbol_index.resolve_go_function_in_dir(file_path.parent, name)

        if fn_node.type == "selector_expression":
            operand = fn_node.child_by_field_name("operand") or (fn_node.children[0] if fn_node.children else None)
            field = fn_node.child_by_field_name("field") or (fn_node.children[-1] if fn_node.children else None)
            if not operand or not field:
                return None
            operand_name = self._node_text(source, operand)
            field_name = self._node_text(source, field)
            if operand_name in import_map:
                import_path = import_map[operand_name]
                target_dir = self.symbol_index.resolve_go_import_dir(import_path)
                if not target_dir:
                    return None
                return self.symbol_index.resolve_go_function_in_dir(target_dir, field_name)

        return None

    def _go_receiver_type(self, source: bytes, node: Optional[Node]) -> Optional[str]:
        if not node:
            return None
        for child in node.children:
            if child.type == "parameter_declaration":
                type_node = child.child_by_field_name("type")
                name = self._node_text(source, type_node) if type_node else None
                if name:
                    return name.lstrip("*")
        return None

    # ---------------------------------------------------------------------
    # Python
    # ---------------------------------------------------------------------

    def _extract_python_edges(self, file_path: Path, content: str) -> DeterministicEdges:
        try:
            module = py_ast.parse(content)
        except SyntaxError:
            return DeterministicEdges()

        current_module = self.symbol_index.python_module_for_file(file_path) or ""
        import_aliases: dict[str, str] = {}
        imported_symbols: dict[str, tuple[str, str]] = {}
        import_targets: list[Path] = []

        for node in module.body:
            if isinstance(node, py_ast.Import):
                for alias in node.names:
                    module_name = alias.name
                    local_name = alias.asname or module_name.split(".")[0]
                    import_aliases[local_name] = module_name
                    target_path = self.symbol_index.resolve_python_module(module_name)
                    if target_path:
                        import_targets.append(target_path)
            elif isinstance(node, py_ast.ImportFrom):
                abs_module = self._python_resolve_import(current_module, node.module, node.level or 0)
                if not abs_module:
                    continue
                target_path = self.symbol_index.resolve_python_module(abs_module)
                if target_path:
                    import_targets.append(target_path)
                for alias in node.names:
                    if alias.name == "*":
                        continue
                    local_name = alias.asname or alias.name
                    imported_symbols[local_name] = (abs_module, alias.name)

        call_edges: list[tuple[str, str]] = []

        class CallVisitor(py_ast.NodeVisitor):
            def __init__(self, outer: DeterministicEdgeExtractor):
                self.outer = outer
                self.current_class: Optional[str] = None
                self.current_caller: Optional[str] = None

            def visit_ClassDef(self, node: py_ast.ClassDef) -> None:
                prev_class = self.current_class
                self.current_class = node.name
                self.generic_visit(node)
                self.current_class = prev_class

            def visit_FunctionDef(self, node: py_ast.FunctionDef) -> None:
                self._visit_function(node)

            def visit_AsyncFunctionDef(self, node: py_ast.AsyncFunctionDef) -> None:
                self._visit_function(node)

            def _visit_function(self, node: py_ast.AST) -> None:
                name = getattr(node, "name", None)
                if not name:
                    self.generic_visit(node)
                    return
                if self.current_class:
                    caller = self.outer.symbol_index.resolve_method_in_class(file_path, self.current_class, name)
                else:
                    caller = self.outer.symbol_index.resolve_function_in_file(file_path, name)
                prev = self.current_caller
                self.current_caller = caller
                self.generic_visit(node)
                self.current_caller = prev

            def visit_Call(self, node: py_ast.Call) -> None:
                if not self.current_caller:
                    return
                target = self._resolve_call(node.func)
                if target:
                    call_edges.append((self.current_caller, target))
                self.generic_visit(node)

            def _resolve_call(self, func: py_ast.AST) -> Optional[str]:
                if isinstance(func, py_ast.Name):
                    name = func.id
                    if name in imported_symbols:
                        module_name, symbol_name = imported_symbols[name]
                        target_path = self.outer.symbol_index.resolve_python_module(module_name)
                        if target_path:
                            return self.outer.symbol_index.resolve_symbol_in_file(target_path, symbol_name)
                    return self.outer.symbol_index.resolve_function_in_file(file_path, name)

                if isinstance(func, py_ast.Attribute) and isinstance(func.value, py_ast.Name):
                    obj_name = func.value.id
                    attr_name = func.attr
                    if obj_name in ("self", "cls") and self.current_class:
                        return self.outer.symbol_index.resolve_method_in_class(file_path, self.current_class, attr_name)
                    if obj_name == self.current_class and self.current_class:
                        return self.outer.symbol_index.resolve_method_in_class(file_path, self.current_class, attr_name)
                    if obj_name in import_aliases:
                        module_name = import_aliases[obj_name]
                        target_path = self.outer.symbol_index.resolve_python_module(module_name)
                        if target_path:
                            return self.outer.symbol_index.resolve_symbol_in_file(target_path, attr_name)
                return None

        CallVisitor(self).visit(module)
        return DeterministicEdges(call_edges=call_edges, import_targets=import_targets)

    def _python_resolve_import(self, current_module: str, module: Optional[str], level: int) -> str:
        if level <= 0:
            return module or ""
        base_parts = current_module.split(".") if current_module else []
        if level > len(base_parts):
            return module or ""
        base_parts = base_parts[: -level]
        module_parts = module.split(".") if module else []
        return ".".join([p for p in (base_parts + module_parts) if p])

    # ---------------------------------------------------------------------
    # Helpers
    # ---------------------------------------------------------------------

    def _node_text(self, source: bytes, node: Optional[Node]) -> str:
        if not node:
            return ""
        return source[node.start_byte : node.end_byte].decode("utf-8")

    def _string_literal_value(self, source: bytes, node: Optional[Node]) -> Optional[str]:
        if not node:
            return None
        raw = self._node_text(source, node)
        if raw.startswith(("'", '"')) and raw.endswith(("'", '"')):
            return raw[1:-1]
        fragment = next((c for c in node.children if c.type == "string_fragment"), None)
        if fragment:
            return self._node_text(source, fragment)
        return raw
