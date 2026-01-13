"""Code indexing service for CODE_* graph and OpenSearch."""

from __future__ import annotations

import hashlib
import logging
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional

from openmemory.api.indexing.api_boundaries import create_api_boundary_analyzer
from openmemory.api.indexing.ast_parser import ASTParser, Language, Symbol, SymbolType
from openmemory.api.indexing.deterministic_edges import (
    DeterministicEdgeExtractor,
    RepoSymbolIndex,
)
from openmemory.api.indexing.fallback_symbols import extract_python_symbols
from openmemory.api.indexing.graph_projection import CodeEdgeType, CodeNodeType, GraphProjection
from openmemory.api.indexing.openapi_parser import OpenAPISpecExtractor
from openmemory.api.indexing.scip_symbols import SCIPSymbolExtractor
from openmemory.api.retrieval.opensearch import Document, IndexConfig, IndexManager

logger = logging.getLogger(__name__)

_CALLER_TYPES = {SymbolType.FUNCTION, SymbolType.METHOD}
_TARGET_TYPES = {SymbolType.FUNCTION, SymbolType.METHOD, SymbolType.CLASS}
_INDEX_TYPES = {
    SymbolType.FUNCTION,
    SymbolType.METHOD,
    SymbolType.CLASS,
    SymbolType.INTERFACE,
    SymbolType.ENUM,
    SymbolType.TYPE_ALIAS,
    SymbolType.VARIABLE,
    SymbolType.FIELD,
    SymbolType.PROPERTY,
}
_FILE_DOC_MAX_CHARS = 8000


@dataclass
class FileIndexStats:
    """Per-file indexing stats."""

    skipped: bool = False
    symbols_indexed: int = 0
    documents_indexed: int = 0
    call_edges_indexed: int = 0


@dataclass
class CodeIndexSummary:
    """Summary stats for a repository indexing run."""

    repo_id: str
    files_indexed: int = 0
    files_failed: int = 0
    symbols_indexed: int = 0
    documents_indexed: int = 0
    call_edges_indexed: int = 0
    duration_ms: float = 0.0
    errors: list[str] = field(default_factory=list)


@dataclass
class FileParseCache:
    """Cached parse data for deterministic edge extraction."""

    language: Language
    symbols: list[Symbol]
    symbol_pairs: list[tuple[Symbol, str]]


class IndexingCancelled(Exception):
    """Raised to abort an indexing run early (e.g., cancellation)."""

    pass


class CodeIndexingService:
    """Indexes a repository into CODE_* graph and OpenSearch."""

    def __init__(
        self,
        root_path: Path,
        repo_id: str,
        graph_driver: Any,
        opensearch_client: Optional[Any] = None,
        embedding_service: Optional[Any] = None,
        index_name: str = "code",
        include_api_boundaries: bool = True,
        extensions: Optional[list[str]] = None,
        ignore_patterns: Optional[list[str]] = None,
    ):
        self.root_path = Path(root_path)
        self.repo_id = repo_id
        self.graph_driver = graph_driver
        self.opensearch_client = opensearch_client
        self.embedding_service = embedding_service
        self.index_name = index_name
        self.extensions = extensions or [".py", ".ts", ".tsx", ".java", ".go", ".js", ".jsx"]
        self.ignore_patterns = ignore_patterns or [
            "__pycache__",
            "node_modules",
            ".git",
            "venv",
            ".venv",
        ]

        self._parser = ASTParser()
        self._extractor = SCIPSymbolExtractor(self.root_path)
        self._projection = GraphProjection(driver=graph_driver)
        self._api_analyzer = (
            create_api_boundary_analyzer(self._projection)
            if include_api_boundaries
            else None
        )
        self._embedding_dim = self._resolve_embedding_dim()
        self._symbol_index: Optional[RepoSymbolIndex] = None
        self._edge_extractor: Optional[DeterministicEdgeExtractor] = None

    def _resolve_embedding_dim(self) -> int:
        provider = getattr(self.embedding_service, "provider", None)
        dimension = getattr(provider, "dimension", None)
        if isinstance(dimension, int) and dimension > 0:
            return dimension
        return 768

    def _ensure_graph_constraints(self) -> None:
        try:
            if not self.graph_driver.has_constraint("code_symbol_scip_id_unique"):
                self.graph_driver.create_constraint(
                    name="code_symbol_scip_id_unique",
                    node_type=CodeNodeType.SYMBOL,
                    property_name="scip_id",
                    constraint_type="UNIQUE",
                )
        except Exception as exc:
            logger.warning(f"Failed to ensure graph constraints: {exc}")

    def _ensure_search_index(self) -> None:
        if not self.opensearch_client:
            return

        manager = IndexManager(self.opensearch_client)
        config = IndexConfig.for_code(
            name=self.index_name,
            embedding_dim=self._embedding_dim,
        )
        try:
            manager.create_index(config, ignore_existing=True)
        except Exception as exc:
            logger.warning(f"Failed to ensure OpenSearch index: {exc}")

    def _iter_source_files(self, max_files: Optional[int] = None) -> list[Path]:
        files: list[Path] = []
        for path in self.root_path.rglob("*"):
            if not path.is_file():
                continue
            if path.suffix not in self.extensions:
                continue
            path_str = str(path)
            if any(pattern in path_str for pattern in self.ignore_patterns):
                continue
            files.append(path)
            if max_files is not None and len(files) >= max_files:
                break
        return files

    def _iter_openapi_files(self) -> list[Path]:
        files: list[Path] = []
        for path in self.root_path.rglob("*"):
            if not path.is_file():
                continue
            if path.suffix.lower() not in (".yaml", ".yml", ".json"):
                continue
            path_str = str(path)
            if any(pattern in path_str for pattern in self.ignore_patterns):
                continue
            name = path.name.lower()
            if "openapi" not in name and "swagger" not in name:
                continue
            files.append(path)
        return files

    def _build_symbol_index(
        self,
        source_files: list[Path],
    ) -> tuple[RepoSymbolIndex, dict[Path, FileParseCache]]:
        symbol_index = RepoSymbolIndex(self.root_path)
        parse_cache: dict[Path, FileParseCache] = {}

        for file_path in source_files:
            language = Language.from_path(file_path)
            if not language:
                continue
            try:
                content = file_path.read_text(errors="ignore")
            except Exception as exc:
                logger.debug(f"Failed to read {file_path} for symbol prepass: {exc}")
                content = ""

            try:
                parse_result = self._parser.parse_file(file_path)
            except Exception as exc:
                logger.debug(f"Parse failed for {file_path}: {exc}")
                continue

            symbols = parse_result.symbols
            if not symbols and language == Language.PYTHON:
                symbols = extract_python_symbols(content)

            symbol_pairs = [
                (symbol, str(self._extractor.extract(symbol, file_path)))
                for symbol in symbols
            ]

            symbol_index.add_file_symbols(file_path, language, symbol_pairs, content)
            parse_cache[file_path] = FileParseCache(
                language=language,
                symbols=symbols,
                symbol_pairs=symbol_pairs,
            )

        return symbol_index, parse_cache

    def reset_repo(self) -> None:
        if hasattr(self.graph_driver, "delete_repo_nodes"):
            self.graph_driver.delete_repo_nodes(self.repo_id)

        if self.opensearch_client:
            try:
                query = {"query": {"term": {"repo_id": self.repo_id}}}
                self.opensearch_client._client.delete_by_query(
                    index=self.index_name,
                    body=query,
                    conflicts="proceed",
                    refresh=True,
                )
            except Exception as exc:
                logger.warning(f"Failed to clear OpenSearch docs for repo: {exc}")

    def index_repository(
        self,
        max_files: Optional[int] = None,
        reset: bool = False,
        progress_callback: Optional[Callable[[dict[str, Any]], None]] = None,
    ) -> CodeIndexSummary:
        start = time.perf_counter()
        summary = CodeIndexSummary(repo_id=self.repo_id)
        files_scanned = 0
        total_files_estimate = 0

        def report_progress(phase: str, current_file: Optional[Path] = None) -> None:
            if not progress_callback:
                return
            progress_callback(
                {
                    "files_scanned": files_scanned,
                    "files_indexed": summary.files_indexed,
                    "files_failed": summary.files_failed,
                    "total_files_estimate": total_files_estimate,
                    "current_file": str(current_file) if current_file else None,
                    "current_phase": phase,
                }
            )

        if reset:
            self.reset_repo()

        self._ensure_graph_constraints()
        self._ensure_search_index()

        source_files = self._iter_source_files(max_files)
        total_files_estimate = len(source_files)
        report_progress("scan")

        self._symbol_index, parse_cache = self._build_symbol_index(source_files)
        self._edge_extractor = DeterministicEdgeExtractor(self.root_path, self._symbol_index)

        for file_path in source_files:
            files_scanned += 1
            report_progress("parse", file_path)
            try:
                cache_entry = parse_cache.get(file_path)
                stats = self.index_file(
                    file_path,
                    phase_callback=lambda phase, path=file_path: report_progress(phase, path),
                    symbols_override=cache_entry.symbols if cache_entry else None,
                    skip_call_edges=True,
                )
            except IndexingCancelled:
                raise
            except Exception as exc:
                summary.files_failed += 1
                error_message = f"{file_path}: {exc}"
                summary.errors.append(error_message)
                logger.warning(f"Indexing failed for {file_path}: {exc}")
                report_progress("parse", file_path)
                continue

            if stats.skipped:
                continue

            summary.files_indexed += 1
            summary.symbols_indexed += stats.symbols_indexed
            summary.documents_indexed += stats.documents_indexed
            summary.call_edges_indexed += stats.call_edges_indexed
            report_progress("graph_projection", file_path)

        summary.call_edges_indexed += self._index_deterministic_edges(
            source_files,
            parse_cache,
            progress_callback=progress_callback,
        )
        self._index_openapi_specs()

        if self.opensearch_client:
            try:
                report_progress("search_indexing")
                self.opensearch_client._client.indices.refresh(index=self.index_name)
            except Exception as exc:
                logger.warning(f"OpenSearch refresh failed: {exc}")

        summary.duration_ms = (time.perf_counter() - start) * 1000
        return summary

    def index_file(
        self,
        file_path: Path,
        phase_callback: Optional[Callable[[str, Optional[Path]], None]] = None,
        symbols_override: Optional[list[Symbol]] = None,
        skip_call_edges: bool = False,
    ) -> FileIndexStats:
        language = Language.from_path(file_path)
        if not language:
            return FileIndexStats(skipped=True)

        content = file_path.read_text(errors="ignore")
        lines = content.splitlines()
        if symbols_override is not None:
            symbols = symbols_override
        else:
            parse_result = self._parser.parse_file(file_path)
            symbols = parse_result.symbols
            if not symbols and language == Language.PYTHON:
                symbols = extract_python_symbols(content)
        content_hash = hashlib.sha1(content.encode("utf-8")).hexdigest()
        file_size = file_path.stat().st_size

        self._projection.create_file_node(
            path=file_path,
            language=language,
            size=file_size,
            content_hash=content_hash,
            repo_id=self.repo_id,
        )

        existing_symbol_ids = self._get_existing_symbol_ids(file_path)
        symbol_pairs = self._index_symbols(file_path, symbols)
        self._index_field_edges(symbol_pairs)
        new_symbol_ids = {symbol_id for _, symbol_id in symbol_pairs}

        stale_symbols = existing_symbol_ids - new_symbol_ids
        for symbol_id in stale_symbols:
            self.graph_driver.delete_node(symbol_id)

        call_edges = 0 if skip_call_edges else self._index_call_edges(symbol_pairs, lines)

        if self._api_analyzer:
            self._index_api_boundaries(language, content, file_path)

        documents_indexed = 0
        if self.opensearch_client:
            if phase_callback:
                phase_callback("search_indexing", file_path)
            documents_indexed = self._index_documents(
                file_path=file_path,
                symbols=symbol_pairs,
                lines=lines,
                content=content,
                language=language,
            )

        return FileIndexStats(
            skipped=False,
            symbols_indexed=len(symbol_pairs),
            documents_indexed=documents_indexed,
            call_edges_indexed=call_edges,
        )

    def _get_existing_symbol_ids(self, file_path: Path) -> set[str]:
        symbol_ids: set[str] = set()
        try:
            edges = self.graph_driver.get_outgoing_edges(str(file_path))
            for edge in edges:
                edge_type = (
                    edge.edge_type.value
                    if hasattr(edge.edge_type, "value")
                    else str(edge.edge_type)
                )
                if edge_type == CodeEdgeType.CONTAINS.value:
                    symbol_ids.add(edge.target_id)
        except Exception as exc:
            logger.debug(f"Failed to read existing symbols for {file_path}: {exc}")
        return symbol_ids

    def _index_symbols(
        self,
        file_path: Path,
        symbols: list[Symbol],
    ) -> list[tuple[Symbol, str]]:
        symbol_pairs: list[tuple[Symbol, str]] = []

        for symbol in symbols:
            scip_id = self._extractor.extract(symbol, file_path)
            symbol_id = str(scip_id)

            self._projection.create_symbol_node(
                symbol=symbol,
                scip_id=scip_id,
                file_path=file_path,
                repo_id=self.repo_id,
            )

            self._projection.create_edge(
                edge_type=CodeEdgeType.CONTAINS,
                source_id=str(file_path),
                target_id=symbol_id,
                properties={"repo_id": self.repo_id},
            )

            symbol_pairs.append((symbol, symbol_id))

        return symbol_pairs

    def _index_field_edges(self, symbol_pairs: list[tuple[Symbol, str]]) -> None:
        parent_ids = {
            symbol.name: symbol_id
            for symbol, symbol_id in symbol_pairs
            if symbol.symbol_type in (SymbolType.CLASS, SymbolType.INTERFACE)
        }
        for symbol, symbol_id in symbol_pairs:
            if symbol.symbol_type not in (SymbolType.FIELD, SymbolType.PROPERTY):
                continue
            if not symbol.parent_name:
                continue
            parent_id = parent_ids.get(symbol.parent_name)
            if not parent_id:
                continue
            self._projection.create_edge(
                edge_type=CodeEdgeType.HAS_FIELD,
                source_id=parent_id,
                target_id=symbol_id,
                properties={"repo_id": self.repo_id},
            )

    def _index_call_edges(
        self,
        symbol_pairs: list[tuple[Symbol, str]],
        lines: list[str],
    ) -> int:
        targets = {
            symbol.name: symbol_id
            for symbol, symbol_id in symbol_pairs
            if symbol.symbol_type in _TARGET_TYPES
        }
        if not targets:
            return 0

        edges_added = 0
        for symbol, symbol_id in symbol_pairs:
            if symbol.symbol_type not in _CALLER_TYPES:
                continue

            body = self._symbol_body(lines, symbol)
            if not body:
                continue

            for target_name, target_id in targets.items():
                if target_id == symbol_id:
                    continue
                if self._calls_target(body, target_name):
                    self._projection.create_edge(
                        edge_type=CodeEdgeType.CALLS,
                        source_id=symbol_id,
                        target_id=target_id,
                        properties={"inferred": True},
                    )
                    edges_added += 1

        return edges_added

    def _symbol_body(self, lines: list[str], symbol: Symbol) -> str:
        if symbol.line_start <= 0 or symbol.line_end <= 0:
            return ""
        start = max(symbol.line_start - 1, 0)
        end = min(symbol.line_end, len(lines))
        body_lines = lines[start:end]
        if body_lines and symbol.symbol_type in _CALLER_TYPES:
            body_lines = body_lines[1:]
        return "\n".join(body_lines)

    def _calls_target(self, body: str, target_name: str) -> bool:
        try:
            pattern = re.compile(rf"\b{re.escape(target_name)}\s*\(")
        except re.error as exc:
            logger.debug(f"Invalid call pattern for {target_name}: {exc}")
            return False
        return bool(pattern.search(body))

    def _index_api_boundaries(
        self,
        language: Language,
        content: str,
        file_path: Path,
    ) -> None:
        try:
            if language == Language.PYTHON:
                self._api_analyzer.analyze_python_file(content, file_path)
            elif language in (Language.TYPESCRIPT, Language.TSX):
                self._api_analyzer.analyze_typescript_file(content, file_path)
        except Exception as exc:
            logger.debug(f"API boundary analysis failed for {file_path}: {exc}")

    def _index_documents(
        self,
        file_path: Path,
        symbols: list[tuple[Symbol, str]],
        lines: list[str],
        content: str,
        language: Language,
    ) -> int:
        indexable = [(symbol, symbol_id) for symbol, symbol_id in symbols if symbol.symbol_type in _INDEX_TYPES]
        file_doc_count = self._index_file_document(
            file_path=file_path,
            content=content,
            language=language,
            lines=lines,
        )
        if not indexable:
            return file_doc_count

        contents = [
            self._symbol_body(lines, symbol) or symbol.name
            for symbol, _ in indexable
        ]
        embeddings = self._embed_contents(contents)
        documents = []
        last_modified = datetime.fromtimestamp(
            file_path.stat().st_mtime, tz=timezone.utc
        ).isoformat()

        for (symbol, symbol_id), content, embedding in zip(indexable, contents, embeddings):
            embedding_value = self._normalize_embedding(embedding)
            metadata = {
                "file_path": str(file_path),
                "language": symbol.language.value,
                "symbol_name": symbol.name,
                "symbol_type": symbol.symbol_type.value,
                "line_start": symbol.line_start,
                "line_end": symbol.line_end,
                "repo_id": self.repo_id,
                "chunk_hash": hashlib.sha1(content.encode("utf-8")).hexdigest(),
                "last_modified": last_modified,
                "symbol_id": symbol_id,
            }
            if symbol.signature:
                metadata["signature"] = symbol.signature
            if symbol.docstring:
                metadata["docstring"] = symbol.docstring
            if symbol.parent_name:
                metadata["parent_name"] = symbol.parent_name

            documents.append(
                Document(
                    id=symbol_id,
                    content=content,
                    embedding=embedding_value,
                    metadata=metadata,
                )
            )

        try:
            result = self.opensearch_client.bulk_index(
                index_name=self.index_name,
                documents=documents,
            )
            return result.succeeded + file_doc_count
        except Exception as exc:
            logger.warning(f"OpenSearch indexing failed for {file_path}: {exc}")
            return file_doc_count

    def _index_file_document(
        self,
        file_path: Path,
        content: str,
        language: Language,
        lines: list[str],
    ) -> int:
        snippet = content[:_FILE_DOC_MAX_CHARS]
        embeddings = self._embed_contents([snippet])
        embedding_value = self._normalize_embedding(embeddings[0] if embeddings else None)
        last_modified = datetime.fromtimestamp(
            file_path.stat().st_mtime, tz=timezone.utc
        ).isoformat()
        doc_id = f"file::{self.repo_id}:{file_path}"
        metadata = {
            "file_path": str(file_path),
            "language": language.value,
            "symbol_name": file_path.name,
            "symbol_type": "file",
            "line_start": 1,
            "line_end": len(lines),
            "repo_id": self.repo_id,
            "chunk_hash": hashlib.sha1(snippet.encode("utf-8")).hexdigest(),
            "last_modified": last_modified,
            "symbol_id": doc_id,
        }
        document = Document(
            id=doc_id,
            content=snippet,
            embedding=embedding_value,
            metadata=metadata,
        )

        try:
            result = self.opensearch_client.bulk_index(
                index_name=self.index_name,
                documents=[document],
            )
            return result.succeeded
        except Exception as exc:
            logger.warning(f"OpenSearch indexing failed for {file_path}: {exc}")
            return 0

    def _embed_contents(self, contents: list[str]) -> list[list[float]]:
        if not contents:
            return []

        if not self.embedding_service:
            return [self._zero_embedding() for _ in contents]

        try:
            if hasattr(self.embedding_service, "embed_batch"):
                results = self.embedding_service.embed_batch(contents)
                return [self._extract_embedding(result) for result in results]
            return [
                self._extract_embedding(self.embedding_service.embed(content))
                for content in contents
            ]
        except Exception as exc:
            logger.warning(f"Embedding failed, using zero vectors: {exc}")
            return [self._zero_embedding() for _ in contents]

    def _extract_embedding(self, result: Any) -> list[float]:
        if isinstance(result, list):
            return result
        if hasattr(result, "embedding"):
            return list(result.embedding)
        try:
            return list(result)
        except TypeError:
            return self._zero_embedding()

    def _normalize_embedding(self, embedding: Optional[list[float]]) -> Optional[list[float]]:
        if not embedding or not any(embedding):
            return None
        return embedding

    def _zero_embedding(self) -> list[float]:
        return [0.0] * self._embedding_dim

    def _index_deterministic_edges(
        self,
        source_files: list[Path],
        parse_cache: dict[Path, FileParseCache],
        progress_callback: Optional[Callable[[dict[str, Any]], None]] = None,
    ) -> int:
        if not self._edge_extractor:
            return 0

        edges_added = 0
        created_schema_ids: set[str] = set()

        for file_path in source_files:
            cache_entry = parse_cache.get(file_path)
            if not cache_entry:
                continue
            if progress_callback:
                progress_callback(
                    {
                        "current_phase": "call_edges",
                        "current_file": str(file_path),
                    }
                )
            try:
                content = file_path.read_text(errors="ignore")
            except Exception as exc:
                logger.debug(f"Failed to read {file_path} for edge extraction: {exc}")
                continue

            lines = content.splitlines()
            if cache_entry.symbol_pairs:
                edges_added += self._index_call_edges(cache_entry.symbol_pairs, lines)

            edges = self._edge_extractor.extract_edges(file_path, cache_entry.language, content)
            for source_id, target_id in edges.call_edges:
                self._projection.create_edge(
                    edge_type=CodeEdgeType.CALLS,
                    source_id=source_id,
                    target_id=target_id,
                    properties={"inferred": False, "resolution": "ast"},
                )
            edges_added += len(edges.call_edges)

            for source_id, target_id in edges.field_reads:
                self._projection.create_edge(
                    edge_type=CodeEdgeType.READS,
                    source_id=source_id,
                    target_id=target_id,
                    properties={"resolution": "ast", "edge_kind": "field"},
                )

            for source_id, target_id in edges.field_writes:
                self._projection.create_edge(
                    edge_type=CodeEdgeType.WRITES,
                    source_id=source_id,
                    target_id=target_id,
                    properties={"resolution": "ast", "edge_kind": "field"},
                )

            for schema_field in edges.schema_fields:
                if schema_field.schema_id in created_schema_ids:
                    continue
                created_schema_ids.add(schema_field.schema_id)
                self._projection.create_schema_field_node(
                    schema_id=schema_field.schema_id,
                    name=schema_field.name,
                    schema_type=schema_field.schema_type,
                    schema_name=schema_field.schema_name,
                    nullable=schema_field.nullable,
                    field_type=schema_field.field_type,
                    file_path=schema_field.file_path,
                    line_start=schema_field.line_start,
                    line_end=schema_field.line_end,
                    repo_id=self.repo_id,
                )

            for source_id, target_id in edges.schema_expose_edges:
                self._projection.create_edge(
                    edge_type=CodeEdgeType.SCHEMA_EXPOSES,
                    source_id=source_id,
                    target_id=target_id,
                    properties={"resolution": "ast", "edge_kind": "schema"},
                )

            for target in {t for t in edges.import_targets if t}:
                if not target.exists() or not target.is_file():
                    continue
                try:
                    target.relative_to(self.root_path)
                except ValueError:
                    continue
                self._projection.create_edge(
                    edge_type=CodeEdgeType.IMPORTS,
                    source_id=str(file_path),
                    target_id=str(target),
                    properties={"resolution": "path"},
                )

        return edges_added

    def _index_openapi_specs(self) -> int:
        if not self._symbol_index:
            return 0

        extractor = OpenAPISpecExtractor(self.root_path, self._symbol_index)
        created_schema_ids: set[str] = set()
        created_openapi_ids: set[str] = set()
        edges_added = 0

        for file_path in self._iter_openapi_files():
            extraction = extractor.extract_from_file(file_path)
            if not extraction.definitions and not extraction.schema_fields:
                continue

            for definition in extraction.definitions:
                if definition.openapi_id in created_openapi_ids:
                    continue
                created_openapi_ids.add(definition.openapi_id)
                self._projection.create_openapi_def_node(
                    openapi_id=definition.openapi_id,
                    name=definition.name,
                    file_path=definition.file_path,
                    title=definition.title,
                    repo_id=self.repo_id,
                )

            for schema_field in extraction.schema_fields:
                if schema_field.schema_id in created_schema_ids:
                    continue
                created_schema_ids.add(schema_field.schema_id)
                self._projection.create_schema_field_node(
                    schema_id=schema_field.schema_id,
                    name=schema_field.name,
                    schema_type=schema_field.schema_type,
                    schema_name=schema_field.schema_name,
                    nullable=schema_field.nullable,
                    field_type=schema_field.field_type,
                    file_path=schema_field.file_path,
                    line_start=schema_field.line_start,
                    line_end=schema_field.line_end,
                    repo_id=self.repo_id,
                )

            for source_id, target_id in extraction.schema_expose_edges:
                self._projection.create_edge(
                    edge_type=CodeEdgeType.SCHEMA_EXPOSES,
                    source_id=source_id,
                    target_id=target_id,
                    properties={"resolution": "openapi", "edge_kind": "schema"},
                )
                edges_added += 1

        return edges_added
