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

    def __init__(self, root_path: Path, symbol_index: RepoSymbolIndex):
        self.root_path = Path(root_path)
        self.symbol_index = symbol_index
        self._ts_parser = Parser(TSLanguage(ts_typescript.language_typescript()))
        self._tsx_parser = Parser(TSLanguage(ts_typescript.language_tsx()))
        self._java_parser = Parser(TSLanguage(ts_java.language()))
        self._go_parser = Parser(TSLanguage(ts_go.language()))

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

        import_map, import_targets = self._collect_ts_imports(file_path, source, root)
        call_edges: list[tuple[str, str]] = []
        field_reads: list[tuple[str, str]] = []
        field_writes: list[tuple[str, str]] = []

        def source_for_access(current_class: Optional[str], current_caller: Optional[str]) -> Optional[str]:
            if current_caller:
                return current_caller
            if current_class:
                return self.symbol_index.resolve_class_in_file(file_path, current_class)
            return None

        def record_field_access(
            access_list: list[tuple[str, str]],
            field_id: Optional[str],
            current_class: Optional[str],
            current_caller: Optional[str],
        ) -> None:
            source_id = source_for_access(current_class, current_caller)
            if field_id and source_id:
                access_list.append((source_id, field_id))

        def visit(node: Node, current_class: Optional[str], current_caller: Optional[str]) -> None:
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
                for child in node.children:
                    visit(child, current_class, caller_id)
                return

            if node.type == "method_definition":
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

            if node.type == "variable_declarator":
                name_node = node.child_by_field_name("name")
                value_node = node.child_by_field_name("value")
                if name_node and value_node and value_node.type == "arrow_function":
                    fn_name = self._node_text(source, name_node)
                    caller_id = self.symbol_index.resolve_function_in_file(file_path, fn_name)
                    visit(value_node, current_class, caller_id)
                else:
                    for child in node.children:
                        visit(child, current_class, current_caller)
                return

            if node.type == "assignment_expression":
                left_node = node.child_by_field_name("left")
                right_node = node.child_by_field_name("right")
                if left_node and left_node.type == "member_expression":
                    field_id = self._resolve_ts_field_access(
                        file_path, source, left_node, current_class
                    )
                    record_field_access(field_writes, field_id, current_class, current_caller)
                if right_node:
                    visit(right_node, current_class, current_caller)
                return

            if node.type == "update_expression":
                member_node = next(
                    (child for child in node.children if child.type == "member_expression"),
                    None,
                )
                if member_node:
                    field_id = self._resolve_ts_field_access(
                        file_path, source, member_node, current_class
                    )
                    record_field_access(field_reads, field_id, current_class, current_caller)
                    record_field_access(field_writes, field_id, current_class, current_caller)
                return

            if node.type == "call_expression" and current_caller:
                target_id = self._resolve_ts_call(
                    file_path,
                    source,
                    node,
                    current_class,
                    import_map,
                )
                if target_id:
                    call_edges.append((current_caller, target_id))
                args_node = node.child_by_field_name("arguments")
                if args_node:
                    for child in args_node.children:
                        visit(child, current_class, current_caller)
                return

            if node.type == "member_expression":
                field_id = self._resolve_ts_field_access(
                    file_path, source, node, current_class
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
        return DeterministicEdges(
            call_edges=call_edges,
            import_targets=import_targets,
            field_reads=field_reads,
            field_writes=field_writes,
            schema_fields=schema_fields,
            schema_expose_edges=schema_expose_edges,
        )

    def _collect_ts_imports(
        self,
        file_path: Path,
        source: bytes,
        root: Node,
    ) -> tuple[dict[str, dict[str, tuple[Path, str]]], list[Path]]:
        import_map: dict[str, dict[str, tuple[Path, str]]] = {
            "named": {},
            "namespace": {},
            "default": {},
        }
        import_targets: list[Path] = []

        for node in root.children:
            if node.type != "import_statement":
                continue
            module_node = node.child_by_field_name("source")
            module_spec = self._string_literal_value(source, module_node) if module_node else None
            target_path = self._resolve_ts_module(file_path, module_spec) if module_spec else None
            if target_path:
                import_targets.append(target_path)

            import_clause = next((c for c in node.children if c.type == "import_clause"), None)
            if not import_clause or not target_path:
                continue

            for child in import_clause.children:
                if child.type == "identifier":
                    local_name = self._node_text(source, child)
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
                        import_map["named"][local_name] = (target_path, imported_name)
                elif child.type == "namespace_import":
                    ident = next((c for c in child.children if c.type == "identifier"), None)
                    if ident:
                        alias = self._node_text(source, ident)
                        import_map["namespace"][alias] = (target_path, alias)

        return import_map, import_targets

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
                target = self.symbol_index.resolve_field_in_type(file_path, obj_name, prop_name)
                if target:
                    return target
                if current_class and obj_name == current_class:
                    return self.symbol_index.resolve_field_in_type(file_path, current_class, prop_name)

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

        def visit(node: Node) -> None:
            if node.type == "variable_declarator":
                name_node = node.child_by_field_name("name")
                value_node = node.child_by_field_name("value")
                if name_node and value_node and self._is_zod_object_call(value_node, source, zod_aliases):
                    schema_name = self._node_text(source, name_node)
                    schema_base = self._schema_base_name(schema_name)
                    shape_node = self._extract_zod_shape_node(value_node)
                    if shape_node:
                        for pair in shape_node.children:
                            if pair.type != "pair":
                                continue
                            key_node = pair.child_by_field_name("key") or pair.child_by_field_name("property")
                            if not key_node:
                                key_node = next(
                                    (child for child in pair.children if child.type in ("property_identifier", "string")),
                                    None,
                                )
                            if not key_node:
                                continue
                            field_name = self._property_key_text(source, key_node)
                            if not field_name:
                                continue
                            value_node = pair.child_by_field_name("value")
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
                            if schema_base:
                                field_id = self.symbol_index.resolve_field_in_type(
                                    file_path,
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
