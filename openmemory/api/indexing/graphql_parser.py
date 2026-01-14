"""GraphQL schema/document parsing for field reference extraction."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass(frozen=True)
class GraphQLFieldRef:
    parent_type: str
    field_name: str


@dataclass
class GraphQLSchemaIndex:
    """Minimal GraphQL schema index for field type resolution."""

    fields: dict[str, dict[str, str]] = field(default_factory=dict)

    def field_type(self, type_name: Optional[str], field_name: str) -> Optional[str]:
        if not type_name:
            return None
        return self.fields.get(type_name, {}).get(field_name)

    @classmethod
    def from_files(cls, files: list[Path]) -> "GraphQLSchemaIndex":
        fields: dict[str, dict[str, str]] = {}
        for file_path in files:
            try:
                content = file_path.read_text(errors="ignore")
            except Exception:
                continue
            for type_name, type_fields in _parse_schema_fields(content).items():
                fields.setdefault(type_name, {}).update(type_fields)
        return cls(fields=fields)


@dataclass
class _SelectionResult:
    fields: list[GraphQLFieldRef]
    spreads: list[str]


@dataclass
class _FragmentDef:
    type_name: str
    fields: list[GraphQLFieldRef]
    spreads: list[str]


_TYPE_DECL_RE = re.compile(r"^\s*(extend\s+)?(type|interface)\s+([_A-Za-z][_0-9A-Za-z]*)")
_FIELD_RE = re.compile(r"^\s*([_A-Za-z][_0-9A-Za-z]*)\s*(\([^)]*\))?\s*:\s*([^#]+)")


def _parse_schema_fields(content: str) -> dict[str, dict[str, str]]:
    fields: dict[str, dict[str, str]] = {}
    current_type: Optional[str] = None

    for raw_line in content.splitlines():
        line = raw_line.split("#", 1)[0].strip()
        if not line:
            continue

        if current_type is None:
            match = _TYPE_DECL_RE.match(line)
            if match:
                current_type = match.group(3)
                fields.setdefault(current_type, {})
                if "}" in line:
                    current_type = None
            continue

        if "}" in line:
            current_type = None
            continue

        match = _FIELD_RE.match(line)
        if not match or not current_type:
            continue
        field_name = match.group(1)
        type_expr = match.group(3)
        type_name = _base_type_name(type_expr)
        if type_name:
            fields[current_type][field_name] = type_name

    return fields


def _base_type_name(type_expr: str) -> Optional[str]:
    cleaned = re.sub(r"[\[\]!]", " ", type_expr)
    cleaned = cleaned.split("@", 1)[0].strip()
    if not cleaned:
        return None
    return cleaned.split()[0]


def extract_graphql_field_refs(
    content: str,
    schema_index: GraphQLSchemaIndex,
) -> list[GraphQLFieldRef]:
    parser = _GraphQLDocParser(content, schema_index)
    return parser.parse()


class _GraphQLDocParser:
    def __init__(self, content: str, schema_index: GraphQLSchemaIndex):
        self.tokens = _tokenize(content)
        self.schema_index = schema_index
        self.pos = 0
        self.fragments: dict[str, _FragmentDef] = {}

    def parse(self) -> list[GraphQLFieldRef]:
        operations: list[_SelectionResult] = []

        while not self._eof():
            token = self._peek()
            if token == ("NAME", "fragment"):
                self._parse_fragment_def()
            elif token and token[0] == "NAME" and token[1] in ("query", "mutation", "subscription"):
                operations.append(self._parse_operation())
            elif token and token[0] == "{":
                operations.append(self._parse_anonymous_query())
            else:
                self._advance()

        refs: list[GraphQLFieldRef] = []
        for selection in operations:
            refs.extend(selection.fields)
            refs.extend(self._resolve_spreads(selection.spreads, set()))

        # De-duplicate
        seen: set[tuple[str, str]] = set()
        unique_refs: list[GraphQLFieldRef] = []
        for ref in refs:
            key = (ref.parent_type, ref.field_name)
            if key in seen:
                continue
            seen.add(key)
            unique_refs.append(ref)
        return unique_refs

    def _parse_fragment_def(self) -> None:
        self._expect("NAME", "fragment")
        name = self._expect("NAME")[1]
        self._expect("NAME", "on")
        type_name = self._expect("NAME")[1]
        selection = self._parse_selection_set(type_name)
        self.fragments[name] = _FragmentDef(
            type_name=type_name,
            fields=selection.fields,
            spreads=selection.spreads,
        )

    def _parse_operation(self) -> _SelectionResult:
        op_type = self._expect("NAME")[1]
        root_type = {
            "query": "Query",
            "mutation": "Mutation",
            "subscription": "Subscription",
        }.get(op_type, "Query")

        if self._peek() and self._peek()[0] == "NAME":
            self._advance()  # operation name

        if self._peek() and self._peek()[0] == "(":
            self._skip_parens()

        return self._parse_selection_set(root_type)

    def _parse_anonymous_query(self) -> _SelectionResult:
        return self._parse_selection_set("Query")

    def _parse_selection_set(self, parent_type: Optional[str]) -> _SelectionResult:
        self._expect("{")
        fields: list[GraphQLFieldRef] = []
        spreads: list[str] = []

        while not self._eof():
            token = self._peek()
            if token == ("}", "}"):
                self._advance()
                break
            if token == ("ELLIPSIS", "..."):
                self._advance()
                if self._peek() == ("NAME", "on"):
                    self._advance()
                    type_name = self._expect("NAME")[1]
                    nested = self._parse_selection_set(type_name)
                    fields.extend(nested.fields)
                    spreads.extend(nested.spreads)
                else:
                    frag_name = self._expect("NAME")[1]
                    spreads.append(frag_name)
                continue
            if token and token[0] == "NAME":
                field_name = self._advance()[1]
                if self._peek() and self._peek()[0] == ":":
                    self._advance()
                    field_name = self._expect("NAME")[1]

                if parent_type:
                    fields.append(GraphQLFieldRef(parent_type=parent_type, field_name=field_name))

                if self._peek() and self._peek()[0] == "(":
                    self._skip_parens()

                while self._peek() and self._peek()[0] == "@":
                    self._advance()
                    self._expect("NAME")
                    if self._peek() and self._peek()[0] == "(":
                        self._skip_parens()

                if self._peek() and self._peek()[0] == "{":
                    child_type = self.schema_index.field_type(parent_type, field_name)
                    nested = self._parse_selection_set(child_type)
                    fields.extend(nested.fields)
                    spreads.extend(nested.spreads)
                continue

            self._advance()

        return _SelectionResult(fields=fields, spreads=spreads)

    def _resolve_spreads(self, spreads: list[str], seen: set[str]) -> list[GraphQLFieldRef]:
        refs: list[GraphQLFieldRef] = []
        for name in spreads:
            if name in seen:
                continue
            seen.add(name)
            frag = self.fragments.get(name)
            if not frag:
                continue
            refs.extend(frag.fields)
            if frag.spreads:
                refs.extend(self._resolve_spreads(frag.spreads, seen))
        return refs

    def _skip_parens(self) -> None:
        if not self._peek() or self._peek()[0] != "(":
            return
        depth = 0
        while not self._eof():
            token = self._advance()
            if token[0] == "(":
                depth += 1
            elif token[0] == ")":
                depth -= 1
                if depth == 0:
                    return

    def _peek(self) -> Optional[tuple[str, str]]:
        if self.pos >= len(self.tokens):
            return None
        return self.tokens[self.pos]

    def _advance(self) -> tuple[str, str]:
        token = self.tokens[self.pos]
        self.pos += 1
        return token

    def _expect(self, kind: str, value: Optional[str] = None) -> tuple[str, str]:
        token = self._peek()
        if token is None:
            return ("", "")
        if token[0] != kind:
            return ("", "")
        if value is not None and token[1] != value:
            return ("", "")
        return self._advance()

    def _eof(self) -> bool:
        return self.pos >= len(self.tokens)


def _tokenize(content: str) -> list[tuple[str, str]]:
    tokens: list[tuple[str, str]] = []
    i = 0
    length = len(content)

    while i < length:
        ch = content[i]

        if ch.isspace() or ch == ",":
            i += 1
            continue

        if ch == "#":
            while i < length and content[i] != "\n":
                i += 1
            continue

        if content.startswith('"""', i):
            i += 3
            end = content.find('"""', i)
            i = end + 3 if end != -1 else length
            continue

        if ch == '"':
            i += 1
            while i < length:
                if content[i] == "\\":
                    i += 2
                    continue
                if content[i] == '"':
                    i += 1
                    break
                i += 1
            continue

        if content.startswith("...", i):
            tokens.append(("ELLIPSIS", "..."))
            i += 3
            continue

        if ch.isalpha() or ch == "_":
            j = i + 1
            while j < length and (content[j].isalnum() or content[j] == "_"):
                j += 1
            tokens.append(("NAME", content[i:j]))
            i = j
            continue

        if ch in "{}():@$":
            tokens.append((ch, ch))
            i += 1
            continue

        i += 1

    return tokens
