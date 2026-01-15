"""
MCP Server for OpenMemory with resilient memory client handling.

This module implements an MCP (Model Context Protocol) server that provides
memory operations for OpenMemory. The memory client is initialized lazily
to prevent server crashes when external dependencies (like Ollama) are
unavailable. If the memory client cannot be initialized, the server will
continue running with limited functionality and appropriate error messages.

Key features:
- Lazy memory client initialization
- Graceful error handling for unavailable dependencies
- Fallback to database-only mode when vector store is unavailable
- Proper logging for debugging connection issues
- Environment variable parsing for API keys
"""

import asyncio
import contextvars
import datetime
import json
import logging
import uuid
from typing import Any, Optional
from pathlib import Path

from app.database import SessionLocal
from app.models import Memory, MemoryAccessLog, MemoryState, MemoryStatusHistory, Config as ConfigModel
from sqlalchemy import and_, or_
from app.utils.structured_memory import (
    build_structured_memory,
    validate_update_fields,
    apply_metadata_updates,
    validate_text,
    StructuredMemoryError,
    normalize_tags_input,
    normalize_tag_list_input,
    SHARED_SCOPES,
)
from app.utils.code_reference import (
    CodeReference,
    CodeReferenceError,
    validate_code_refs_input,
    serialize_code_refs_to_tags,
    deserialize_code_refs_from_tags,
    format_code_refs_for_response,
    has_code_refs_in_tags,
)
from app.utils.reranking import (
    SearchContext,
    ExclusionFilters,
    compute_boost,
    should_exclude,
    calculate_final_score,
    normalize_tags,
    parse_datetime,
)
from app.utils.response_format import (
    format_search_results,
    format_memory_list,
    format_add_memories_response,
    format_compact_relations,
)
from app.utils.db import get_user_and_app
from app.utils.memory import get_memory_client
from app.utils.permissions import check_memory_access_permissions
from app.utils.debug_timing import DebugTimer
from app.indexing_jobs import create_index_job, get_index_job, run_index_job
from app.security.access import resolve_access_entity_for_scope
from app.graph.graph_ops import (
    project_memory_to_graph,
    delete_memory_from_graph,
    get_meta_relations_for_memories,
    find_related_memories_in_graph,
    get_memory_subgraph_from_graph,
    aggregate_memories_in_graph,
    tag_cooccurrence_in_graph,
    path_between_entities_in_graph,
    get_graph_relations,
    is_graph_enabled,
    is_mem0_graph_enabled,
    # New edge operations
    update_entity_edges_on_memory_add,
    update_entity_edges_on_memory_delete,
    get_entity_network_from_graph,
    update_tag_edges_on_memory_add,
    get_related_tags_from_graph,
    project_similarity_edges_for_memory,
    delete_similarity_edges_for_memory,
    get_similar_memories_from_graph,
    is_similarity_enabled,
    # Entity normalization operations
    find_duplicate_entities_in_graph,
    normalize_entities_in_graph,
    # Typed relations operations
    get_entity_relations_from_graph,
    # Biographical timeline operations
    get_biography_timeline_from_graph,
)
from app.graph.entity_bridge import (
    bridge_entities_to_om_graph,
    bridge_entities_to_om_graph_from_extraction,
)
from app.graph.entity_extraction import (
    extract_entities_and_relations,
    write_mem0_graph_from_extraction,
)
from app.memory_jobs import create_memory_job, get_memory_job, list_memory_jobs, update_memory_job
from app.path_utils import resolve_repo_root
from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.routing import APIRouter
from mcp.server.fastmcp import FastMCP
from mcp.types import ToolAnnotations
from app.mcp.sse_transport import SessionAwareSseTransport
from app.security.session_binding import get_session_binding_store

# Load environment variables
load_dotenv()

# Initialize MCP for memory tools
mcp = FastMCP("mem0-mcp-server")

# Initialize separate MCP for Business Concept tools (reduces context window bloat)
concept_mcp = FastMCP("business-concepts-mcp-server")

# Don't initialize memory client at import time - do it lazily when needed
def get_memory_client_safe():
    """Get memory client with error handling. Returns None if client cannot be initialized."""
    try:
        return get_memory_client()
    except Exception as e:
        logging.warning(f"Failed to get memory client: {e}")
        return None


def _get_graph_write_on_add(db) -> bool:
    """Return OpenMemory mem0.graph_write_on_add flag (default: False)."""
    try:
        row = db.query(ConfigModel).filter(ConfigModel.key == "main").first()
        if not row or not row.value:
            return False
        config = row.value
        openmemory_cfg = config.get("openmemory", {})
        om_mem0_cfg = openmemory_cfg.get("mem0", {})
        if "graph_write_on_add" in om_mem0_cfg:
            return bool(om_mem0_cfg.get("graph_write_on_add"))
        mem0_cfg = config.get("mem0", {})
        if "graph_write_on_add" in mem0_cfg:
            return bool(mem0_cfg.get("graph_write_on_add"))
    except Exception:
        return False
    return False


def _project_graph_for_results(
    valid_results: list[dict],
    user_id: str,
    combined_metadata: dict,
    graph_write_on_add: bool,
) -> None:
    """Project memory updates into graph stores (best-effort)."""
    logging.info(f"Graph projection: checking {len(valid_results)} results, graph_enabled={is_graph_enabled()}")
    for result in valid_results:
        logging.info(f"Graph projection: result id={result.get('id')}, event={result.get('event')}")
        if 'id' in result and result.get('event') == 'ADD':
            try:
                logging.info(f"Graph projection: projecting {result['id']}")
                project_memory_to_graph(
                    memory_id=result['id'],
                    user_id=user_id,  # Use string user_id for graph scoping
                    content=result.get('memory', ''),
                    metadata=combined_metadata,
                    state="active",
                )
                # Bridge multi-entity extraction from Mem0 to OM graph
                # This creates OM_ABOUT edges for ALL extracted entities (not just metadata.re)
                # and typed OM_RELATION edges between related entities
                if is_mem0_graph_enabled():
                    try:
                        if graph_write_on_add:
                            bridge_result = bridge_entities_to_om_graph(
                                memory_id=result['id'],
                                user_id=user_id,
                                content=result.get('memory', ''),
                                existing_entity=combined_metadata.get('entity'),
                            )
                        else:
                            extraction = extract_entities_and_relations(
                                content=result.get('memory', ''),
                                user_id=user_id,
                            )
                            write_mem0_graph_from_extraction(
                                content=result.get('memory', ''),
                                user_id=user_id,
                                extraction=extraction,
                            )
                            bridge_result = bridge_entities_to_om_graph_from_extraction(
                                memory_id=result['id'],
                                user_id=user_id,
                                extraction=extraction,
                                existing_entity=combined_metadata.get('entity'),
                            )
                        logging.info(
                            "Entity bridge: %s entities, %s relations",
                            bridge_result.get('entities_bridged', 0),
                            bridge_result.get('relations_created', 0),
                        )
                    except Exception as bridge_error:
                        logging.warning(f"Entity bridge failed for {result['id']}: {bridge_error}")
                # Update entity-to-entity co-mention edges (now works with multi-entity bridging)
                update_entity_edges_on_memory_add(result['id'], user_id)
                # Update tag-to-tag co-occurrence edges
                update_tag_edges_on_memory_add(result['id'], user_id)
                # Project similarity edges (K nearest neighbors)
                project_similarity_edges_for_memory(result['id'], user_id)
            except Exception as graph_error:
                logging.warning(f"Graph projection failed for {result['id']}: {graph_error}")
        elif 'id' in result and result.get('event') == 'DELETE':
            try:
                # Update entity edges before deleting memory
                update_entity_edges_on_memory_delete(result['id'], user_id)
                # Delete similarity edges
                delete_similarity_edges_for_memory(result['id'], user_id)
                # Delete the memory node
                delete_memory_from_graph(result['id'])
            except Exception as graph_error:
                logging.warning(f"Graph deletion failed for {result['id']}: {graph_error}")


async def _run_graph_projection_async(
    valid_results: list[dict],
    user_id: str,
    combined_metadata: dict,
    graph_write_on_add: bool,
    graph_job_id: str,
) -> None:
    """Run graph projection in a background thread to avoid blocking the response."""
    try:
        update_memory_job(
            graph_job_id,
            status="running",
            started_at=datetime.datetime.now(datetime.UTC).isoformat(),
        )
        await asyncio.to_thread(
            _project_graph_for_results,
            valid_results,
            user_id,
            combined_metadata,
            graph_write_on_add,
        )
        update_memory_job(
            graph_job_id,
            status="succeeded",
            finished_at=datetime.datetime.now(datetime.UTC).isoformat(),
            result=_summarize_graph_job(valid_results),
        )
    except Exception as e:
        logging.warning(f"Graph projection task failed: {e}")
        update_memory_job(
            graph_job_id,
            status="failed",
            finished_at=datetime.datetime.now(datetime.UTC).isoformat(),
            error=str(e),
        )


def _summarize_add_job(
    structured_metadata: dict,
    valid_results: list[dict],
) -> dict:
    return {
        "category": structured_metadata.get("category"),
        "scope": structured_metadata.get("scope"),
        "entity": structured_metadata.get("entity"),
        "access_entity": structured_metadata.get("access_entity"),
        "memory_ids": [r.get("id") for r in valid_results if r.get("id")],
        "count": len(valid_results),
    }


def _summarize_graph_job(
    valid_results: list[dict],
    parent_add_job_id: str | None = None,
) -> dict:
    summary = {
        "memory_ids": [r.get("id") for r in valid_results if r.get("id")],
        "count": len(valid_results),
    }
    if parent_add_job_id:
        summary["parent_add_job_id"] = parent_add_job_id
    return summary


def _summarize_update_job(
    *,
    memory_ids: list[str],
    validated_fields: dict,
    content_text: str | None,
    normalized_add_tags: dict | None,
    normalized_remove_tags: list | None,
    mcp_client: str | None,
    is_batch: bool,
) -> dict:
    fields_updated = list(validated_fields.keys())
    if content_text is not None:
        fields_updated.append("text")
    if mcp_client is not None:
        fields_updated.append("mcp_client")
    summary = {
        "memory_ids": list(memory_ids),
        "count": len(memory_ids),
        "batch": is_batch,
    }
    if fields_updated:
        summary["fields_updated"] = fields_updated
    if normalized_add_tags:
        summary["add_tags"] = list(normalized_add_tags.keys())
    if normalized_remove_tags:
        summary["remove_tags"] = list(normalized_remove_tags)
    if mcp_client is not None:
        summary["mcp_client"] = mcp_client
    return summary


def _update_memory_core(
    *,
    user_id: str,
    client_name: str,
    ids_to_update: list[str],
    validated_fields: dict,
    content_text: str | None,
    normalized_add_tags: dict | None,
    normalized_remove_tags: list | None,
    preserve_timestamps: bool,
    mcp_client: str | None,
    is_batch: bool,
    principal,
) -> dict:
    try:
        db = SessionLocal()
        try:
            user, app = get_user_and_app(db, user_id=user_id, app_id=client_name)
            memory_client = get_memory_client_safe()

            updated_ids = []
            failed_ids = []
            single_failure = None

            for mid in ids_to_update:
                try:
                    memory = db.query(Memory).filter(
                        Memory.id == uuid.UUID(mid),
                    ).first()

                    if not memory:
                        failed_ids.append({"id": mid, "error": "not found"})
                        if not is_batch:
                            single_failure = {"error": f"Memory '{mid}' not found"}
                        continue

                    current_metadata = memory.metadata_ or {}
                    current_access_entity = current_metadata.get("access_entity")

                    if not current_access_entity and memory.user_id != user.id:
                        failed_ids.append({"id": mid, "error": "not found"})
                        if not is_batch:
                            single_failure = {"error": f"Memory '{mid}' not found"}
                        continue

                    if principal:
                        from app.security.access import can_write_to_access_entity

                        if current_access_entity:
                            if not can_write_to_access_entity(principal, current_access_entity):
                                failed_ids.append({"id": mid, "error": "access denied", "code": "FORBIDDEN"})
                                if not is_batch:
                                    single_failure = {
                                        "error": (
                                            "Access denied: you don't have permission to update "
                                            f"memories with access_entity='{current_access_entity}'"
                                        ),
                                        "code": "FORBIDDEN",
                                    }
                                continue

                        new_access_entity = validated_fields.get("access_entity")
                        if new_access_entity and new_access_entity != current_access_entity:
                            if not can_write_to_access_entity(principal, new_access_entity):
                                failed_ids.append({
                                    "id": mid,
                                    "error": f"cannot set access_entity={new_access_entity}",
                                    "code": "FORBIDDEN",
                                })
                                if not is_batch:
                                    single_failure = {
                                        "error": (
                                            "Access denied: you don't have permission to set "
                                            f"access_entity='{new_access_entity}'"
                                        ),
                                        "code": "FORBIDDEN",
                                    }
                                continue

                    updated_metadata = apply_metadata_updates(
                        current_metadata=current_metadata,
                        validated_fields=validated_fields,
                        add_tags=normalized_add_tags,
                        remove_tags=normalized_remove_tags,
                    )
                    if mcp_client is not None:
                        updated_metadata["mcp_client"] = mcp_client

                    content_updated = False
                    if content_text is not None:
                        memory.content = content_text
                        content_updated = True

                    memory.metadata_ = updated_metadata

                    if not preserve_timestamps:
                        memory.updated_at = datetime.datetime.now(datetime.UTC)

                    metadata_updated = bool(
                        validated_fields or normalized_add_tags or normalized_remove_tags or mcp_client
                    )

                    if memory_client:
                        if content_updated:
                            try:
                                memory_client.update(mid, content_text)
                                from app.utils.vector_sync import sync_metadata_to_qdrant_with_mcp_client
                                sync_metadata_to_qdrant_with_mcp_client(
                                    memory_id=mid,
                                    memory=memory,
                                    metadata=updated_metadata,
                                    mcp_client_override=mcp_client,
                                    memory_client=memory_client,
                                )
                            except Exception as vs_error:
                                logging.warning(f"Failed to update vector store for {mid}: {vs_error}")
                        elif metadata_updated:
                            try:
                                from app.utils.vector_sync import sync_metadata_to_qdrant_with_mcp_client
                                sync_metadata_to_qdrant_with_mcp_client(
                                    memory_id=mid,
                                    memory=memory,
                                    metadata=updated_metadata,
                                    mcp_client_override=mcp_client,
                                    memory_client=memory_client,
                                )
                            except Exception as vs_error:
                                logging.warning(f"Failed to sync metadata to vector store for {mid}: {vs_error}")

                    access_log = MemoryAccessLog(
                        memory_id=memory.id,
                        app_id=app.id,
                        access_type="update",
                        metadata_={
                            "fields_updated": (
                                list(validated_fields.keys())
                                + (["text"] if content_text is not None else [])
                                + (["mcp_client"] if mcp_client else [])
                            ),
                            "add_tags": normalized_add_tags,
                            "remove_tags": normalized_remove_tags,
                            "preserve_timestamps": preserve_timestamps,
                            "mcp_client_override": mcp_client,
                        },
                    )
                    db.add(access_log)

                    try:
                        project_memory_to_graph(
                            memory_id=mid,
                            user_id=user_id,
                            content=memory.content,
                            metadata=updated_metadata,
                            created_at=memory.created_at.isoformat() if memory.created_at else None,
                            updated_at=memory.updated_at.isoformat() if memory.updated_at else None,
                            state=memory.state.value if memory.state else "active",
                        )
                    except Exception as graph_error:
                        logging.warning(f"Graph projection failed for updated memory {mid}: {graph_error}")

                    updated_ids.append(mid)

                except Exception as mem_error:
                    logging.warning(f"Failed to update memory {mid}: {mem_error}")
                    failed_ids.append({"id": mid, "error": str(mem_error)})
                    if not is_batch:
                        single_failure = {"error": str(mem_error)}

            db.commit()

            if is_batch:
                response = {
                    "status": "batch_update_complete",
                    "updated": len(updated_ids),
                    "failed": len(failed_ids),
                    "updated_ids": updated_ids,
                }
                if failed_ids:
                    response["failures"] = failed_ids
                if mcp_client:
                    response["mcp_client_set_to"] = mcp_client
            else:
                if updated_ids:
                    response = {
                        "status": "updated",
                        "memory_id": updated_ids[0],
                    }
                    if validated_fields:
                        response["fields_updated"] = list(validated_fields.keys())
                    if mcp_client:
                        response["mcp_client"] = mcp_client
                    if normalized_add_tags or normalized_remove_tags:
                        response["current_tags"] = updated_metadata.get("tags", {})
                else:
                    return single_failure or {"error": "Unknown error"}

            return response
        finally:
            db.close()
    except Exception as e:
        logging.exception(e)
        return {"error": f"Error updating memory: {e}"}


async def _run_update_memory_job(
    *,
    job_id: str,
    user_id: str,
    client_name: str,
    ids_to_update: list[str],
    validated_fields: dict,
    content_text: str | None,
    normalized_add_tags: dict | None,
    normalized_remove_tags: list | None,
    preserve_timestamps: bool,
    mcp_client: str | None,
    is_batch: bool,
    principal,
) -> None:
    update_memory_job(
        job_id,
        status="running",
        started_at=datetime.datetime.now(datetime.UTC).isoformat(),
    )
    response = await asyncio.to_thread(
        _update_memory_core,
        user_id=user_id,
        client_name=client_name,
        ids_to_update=ids_to_update,
        validated_fields=validated_fields,
        content_text=content_text,
        normalized_add_tags=normalized_add_tags,
        normalized_remove_tags=normalized_remove_tags,
        preserve_timestamps=preserve_timestamps,
        mcp_client=mcp_client,
        is_batch=is_batch,
        principal=principal,
    )

    status = "succeeded"
    error_message = None
    if response.get("error"):
        status = "failed"
        error_message = response.get("error")
    elif is_batch:
        failed_count = response.get("failed", 0) if isinstance(response, dict) else 0
        updated_count = response.get("updated", 0) if isinstance(response, dict) else 0
        if failed_count and updated_count:
            status = "partial"
        elif failed_count and not updated_count:
            status = "failed"
            error_message = "All updates failed"

    update_memory_job(
        job_id,
        status=status,
        finished_at=datetime.datetime.now(datetime.UTC).isoformat(),
        result=response,
        error=error_message,
    )

def _add_memories_core(
    *,
    memory_client: object,
    user_id: str,
    client_name: str,
    clean_text: str,
    structured_metadata: dict,
    combined_metadata: dict,
    infer: bool,
    debug: bool = False,
) -> tuple[dict, list[dict], dict | None, bool]:
    """Run the add_memories logic and return (response, valid_results, timing, graph_write_on_add).

    Returns:
        Tuple of (formatted_response, valid_results, timing_dict, graph_write_on_add).
        timing_dict is None if debug=False, otherwise contains internal timings.
    """
    import time
    timing = {} if debug else None

    try:
        db = SessionLocal()
        try:
            # Get or create user and app
            if debug:
                t0 = time.perf_counter()
            user, app = get_user_and_app(db, user_id=user_id, app_id=client_name)
            if debug:
                timing["db_get_user_app"] = round((time.perf_counter() - t0) * 1000, 2)

            # Check if app is active
            if not app.is_active:
                return (
                    {"error": f"App {app.name} is currently paused on OpenMemory. Cannot create new memories."},
                    [],
                    timing,
                    False,
                )

            graph_write_on_add = _get_graph_write_on_add(db)

            # Call Mem0 client (this includes embedding + optional LLM + Qdrant store)
            if debug:
                t0 = time.perf_counter()
            response = memory_client.add(
                clean_text,
                user_id=user_id,
                metadata=combined_metadata,
                infer=infer,
                graph_write=graph_write_on_add,
                debug=debug,
            )
            if debug:
                timing["mem0_client_add"] = round((time.perf_counter() - t0) * 1000, 2)
                # Extract detailed timing from mem0 response
                if isinstance(response, dict) and "debug_timing" in response:
                    timing["mem0_breakdown"] = response.pop("debug_timing")

            # Process the response and update database
            if isinstance(response, dict) and 'results' in response:
                results = response.get('results', [])
                if not results:
                    return (
                        {
                            "error": "Memory client returned no results",
                            "code": "MEMORY_ADD_EMPTY_RESULT",
                            "hint": "If you expect raw storage, set infer=false",
                        },
                        [],
                        timing,
                        graph_write_on_add,
                    )

                valid_results = [
                    result for result in results
                    if result.get("id") and result.get("memory")
                ]
                invalid_results = [
                    result for result in results
                    if not result.get("id") or not result.get("memory")
                ]
                if not valid_results:
                    return (
                        {
                            "error": "Memory creation failed",
                            "code": "MEMORY_ADD_FAILED",
                            "details": invalid_results,
                        },
                        [],
                        timing,
                        graph_write_on_add,
                    )

                if debug:
                    t0 = time.perf_counter()

                for result in valid_results:
                    if 'id' not in result:
                        continue

                    memory_id = uuid.UUID(result['id'])
                    memory = db.query(Memory).filter(Memory.id == memory_id).first()

                    if result['event'] == 'ADD':
                        if not memory:
                            memory = Memory(
                                id=memory_id,
                                user_id=user.id,
                                app_id=app.id,
                                content=result['memory'],
                                metadata_=combined_metadata,
                                state=MemoryState.active,
                            )
                            db.add(memory)
                        else:
                            memory.state = MemoryState.active
                            memory.content = result['memory']
                            memory.metadata_ = combined_metadata

                        # Create history entry
                        history = MemoryStatusHistory(
                            memory_id=memory_id,
                            changed_by=user.id,
                            old_state=MemoryState.deleted if memory else None,
                            new_state=MemoryState.active
                        )
                        db.add(history)

                    elif result['event'] == 'DELETE':
                        if memory:
                            memory.state = MemoryState.deleted
                            memory.deleted_at = datetime.datetime.now(datetime.UTC)
                            # Create history entry
                            history = MemoryStatusHistory(
                                memory_id=memory_id,
                                changed_by=user.id,
                                old_state=MemoryState.active,
                                new_state=MemoryState.deleted
                            )
                            db.add(history)

                if debug:
                    t1 = time.perf_counter()

                db.commit()

                if debug:
                    timing["postgresql_write"] = round((time.perf_counter() - t0) * 1000, 2)
                    timing["postgresql_commit"] = round((time.perf_counter() - t1) * 1000, 2)

                # Format response using the new lean format
                formatted_response = format_add_memories_response(
                    results=valid_results,
                    structured_metadata=structured_metadata,
                )
                if invalid_results:
                    formatted_response["error"] = "Some memories failed to save"
                    formatted_response["failed"] = invalid_results
                return formatted_response, valid_results, timing, graph_write_on_add

            # Handle case where response is not in expected format
            return {"error": "Unexpected response format from memory client"}, [], timing, graph_write_on_add
        finally:
            db.close()
    except Exception as e:
        logging.exception(f"Error adding to memory: {e}")
        return {"error": f"Error adding to memory: {e}"}, [], timing, False


async def _run_add_memories_job(
    *,
    job_id: str,
    memory_client: object,
    user_id: str,
    client_name: str,
    clean_text: str,
    structured_metadata: dict,
    combined_metadata: dict,
    infer: bool,
    debug: bool = False,
) -> None:
    import time
    job_start_time = time.perf_counter() if debug else None

    update_memory_job(job_id, status="running", started_at=datetime.datetime.now(datetime.UTC).isoformat())
    response, valid_results, core_timing, graph_write_on_add = await asyncio.to_thread(
        _add_memories_core,
        memory_client=memory_client,
        user_id=user_id,
        client_name=client_name,
        clean_text=clean_text,
        structured_metadata=structured_metadata,
        combined_metadata=combined_metadata,
        infer=infer,
        debug=debug,
    )

    summary = _summarize_add_job(structured_metadata, valid_results)

    # Build debug timing if enabled
    debug_timing = None
    if debug and job_start_time is not None:
        total_job_ms = (time.perf_counter() - job_start_time) * 1000
        debug_timing = {
            "total_ms": round(total_job_ms, 2),
            "breakdown": {},
            "details": {},
        }
        if core_timing:
            # Add core timing breakdown with _ms suffix
            for key, value in core_timing.items():
                if isinstance(value, (int, float)):
                    debug_timing["breakdown"][f"{key}_ms"] = value
                elif isinstance(value, dict):
                    # Nested timing breakdown (e.g., mem0_breakdown) - store in details
                    debug_timing["details"][key] = value
        # Remove empty details
        if not debug_timing["details"]:
            del debug_timing["details"]
        debug_timing["note"] = "Timing from async job execution (excludes graph projection)"

    if valid_results:
        graph_job_id = create_memory_job(
            requested_by=user_id,
            summary=_summarize_graph_job(valid_results, parent_add_job_id=job_id),
            job_type="graph_projection",
        )
        summary["graph_job_id"] = graph_job_id
        status = "succeeded"
        if response.get("error"):
            status = "partial"
        update_memory_job(
            job_id,
            status=status,
            finished_at=datetime.datetime.now(datetime.UTC).isoformat(),
            result=response,
            summary=summary,
            debug_timing=debug_timing,
        )
        asyncio.create_task(
            _run_graph_projection_async(
                valid_results=list(valid_results),
                user_id=user_id,
                combined_metadata=dict(combined_metadata),
                graph_write_on_add=graph_write_on_add,
                graph_job_id=graph_job_id,
            )
        )
        return

    update_memory_job(
        job_id,
        status="failed",
        finished_at=datetime.datetime.now(datetime.UTC).isoformat(),
        result=response,
        error=response.get("error"),
        summary=summary,
        debug_timing=debug_timing,
    )

# Context variables for user_id, client_name, and org_id
user_id_var: contextvars.ContextVar[str] = contextvars.ContextVar("user_id")
client_name_var: contextvars.ContextVar[str] = contextvars.ContextVar("client_name")
org_id_var: contextvars.ContextVar[str] = contextvars.ContextVar("org_id")

# Import security dependencies for JWT validation
try:
    from app.security.jwt import validate_jwt, validate_iat_not_future
    from app.security.types import Principal, AuthenticationError, AuthorizationError, Scope
    HAS_SECURITY = True
except ImportError:
    HAS_SECURITY = False
    logging.warning("Security module not available - MCP endpoints will not require authentication")

# Create a router for MCP endpoints
mcp_router = APIRouter(prefix="/mcp")

# Create a separate router for Business Concept endpoints
concept_router = APIRouter(prefix="/concepts")

# Initialize SSE transport for MCP tools (with session ID capture)
sse = SessionAwareSseTransport("/mcp/messages/")


# --- MCP Session Health Endpoint (Phase 2) ---

@mcp_router.get("/health")
async def mcp_session_health():
    """Health check endpoint for MCP session binding store.

    Checks:
    - Session binding store connectivity (memory always ok, Valkey ping)
    - Cleanup scheduler status (if available)

    Returns:
        JSON response with status and details:
        - status: "healthy" or "unhealthy"
        - store_type: "memory" or "valkey"
        - reason: Error description if unhealthy
    """
    from datetime import datetime, timezone
    import time

    timestamp = datetime.now(timezone.utc).isoformat()
    start_time = time.perf_counter()

    try:
        store = get_session_binding_store()
        store_type = getattr(store, 'STORE_TYPE', 'memory')

        # Check store health
        if hasattr(store, 'health_check'):
            # Valkey store has health_check method
            store_healthy = store.health_check()
        else:
            # Memory store is always healthy if instantiated
            store_healthy = True

        # Check cleanup scheduler status
        try:
            from app.tasks.session_cleanup import _cleanup_scheduler
            scheduler_running = _cleanup_scheduler is not None and _cleanup_scheduler._running
        except Exception:
            scheduler_running = None  # Unknown status

        latency_ms = (time.perf_counter() - start_time) * 1000

        if store_healthy:
            return JSONResponse(
                status_code=200,
                content={
                    "status": "healthy",
                    "store_type": store_type,
                    "scheduler_running": scheduler_running,
                    "latency_ms": round(latency_ms, 2),
                    "timestamp": timestamp,
                }
            )
        else:
            return JSONResponse(
                status_code=503,
                content={
                    "status": "unhealthy",
                    "store_type": store_type,
                    "reason": "Store health check failed",
                    "scheduler_running": scheduler_running,
                    "latency_ms": round(latency_ms, 2),
                    "timestamp": timestamp,
                }
            )
    except Exception as e:
        latency_ms = (time.perf_counter() - start_time) * 1000
        return JSONResponse(
            status_code=503,
            content={
                "status": "unhealthy",
                "reason": str(e),
                "latency_ms": round(latency_ms, 2),
                "timestamp": timestamp,
            }
        )

# Initialize SSE transport for Business Concept tools (with session ID capture)
concept_sse = SessionAwareSseTransport("/concepts/messages/")


def _extract_principal_from_request(request: Request) -> "Principal":
    """
    Extract and validate the authenticated principal from the request.

    Raises:
        AuthenticationError: If authentication fails or is missing
    """
    if not HAS_SECURITY:
        # Security module not available - create a mock principal from path params
        # This is for backwards compatibility during migration
        uid = request.path_params.get("user_id", "anonymous")
        return type('Principal', (), {
            'user_id': uid,
            'org_id': 'default',
            'has_scope': lambda self, scope: True,
            'claims': None,
        })()

    # Extract Authorization header
    auth_header = request.headers.get("authorization", "")
    if not auth_header:
        raise AuthenticationError(
            message="Missing authorization header",
            code="MISSING_AUTH"
        )

    # Parse Bearer token
    if not auth_header.lower().startswith("bearer "):
        raise AuthenticationError(
            message="Invalid authorization header format - expected 'Bearer <token>'",
            code="INVALID_AUTH_FORMAT"
        )

    token = auth_header[7:]  # Remove "Bearer " prefix
    if not token:
        raise AuthenticationError(
            message="Empty authorization token",
            code="EMPTY_TOKEN"
        )

    # Validate the JWT token
    claims = validate_jwt(token)
    # Reject tokens issued in the future (clock skew protection)
    validate_iat_not_future(claims.iat)

    # Build the principal
    principal = Principal(
        user_id=claims.sub,
        org_id=claims.org_id,
        claims=claims,
    )

    return principal


async def _extract_dpop_thumbprint(request: Request, token: str) -> str | None:
    """
    Extract and validate DPoP proof from request, returning thumbprint if valid.

    For POST requests bound to a session with DPoP, this validates:
    - DPoP header is present
    - Proof has correct htm (HTTP method)
    - Proof has correct htu (HTTP URI)
    - Proof signature is valid
    - Returns the JWK thumbprint for session binding validation

    Returns None if no DPoP header is present.

    Raises:
        AuthenticationError: If DPoP header is present but invalid
    """
    dpop_header = request.headers.get("dpop")
    if not dpop_header:
        return None

    try:
        from .security.dpop import DPoPValidator, get_dpop_cache

        cache = await get_dpop_cache()
        if not cache:
            # DPoP cache not configured - skip validation
            return None

        validator = DPoPValidator(cache)

        # Build the full URL for htu validation
        http_uri = str(request.url)
        http_method = request.method

        # Validate the DPoP proof
        await validator.validate(
            dpop_proof=dpop_header,
            http_method=http_method,
            http_uri=http_uri,
            access_token=token,
        )

        # Extract and return the thumbprint
        return await validator.get_thumbprint(dpop_header)
    except Exception as e:
        if isinstance(e, AuthenticationError):
            raise
        raise AuthenticationError(
            message=f"DPoP validation failed: {e}",
            code="INVALID_DPOP"
        )


def _check_scope(principal, scope: "Scope") -> None:
    """
    Check if the principal has the required scope.

    Raises:
        AuthorizationError: If the principal lacks the required scope
    """
    if not HAS_SECURITY:
        return  # Security not available, skip check

    if not principal.has_scope(scope):
        raise AuthorizationError(
            message=f"Required scope '{scope.value}' not granted",
            code="INSUFFICIENT_SCOPE",
        )


# Context variable to store the current principal for tool scope checks
principal_var: contextvars.ContextVar["Principal"] = contextvars.ContextVar("principal")


def _check_tool_scope(required_scope: str) -> str | None:
    """
    Check if the current principal has the required scope for a tool.

    Returns None if scope check passes, or a JSON error string if it fails.
    This is designed for MCP tools which return JSON strings.
    """
    if not HAS_SECURITY:
        return None  # Security not available, skip check

    # Get principal from context var (set by SSE handlers)
    # Fail closed: if no principal, deny access
    principal = principal_var.get(None)
    if not principal:
        return json.dumps({
            "error": "Authentication required",
            "code": "MISSING_AUTH",
        })

    # Check scope
    if not principal.has_scope(required_scope):
        return json.dumps({
            "error": f"Insufficient scope",
            "code": "INSUFFICIENT_SCOPE",
            "required_scope": required_scope,
        })


def _can_access_job(job: dict, admin_scope: str) -> bool:
    """Check whether the current principal can access the job."""
    if not HAS_SECURITY:
        return True
    principal = principal_var.get(None)
    if not principal:
        return False
    if principal.has_scope(admin_scope):
        return True
    requested_by = job.get("requested_by")
    if not requested_by:
        return True
    return requested_by == principal.user_id


def _apply_access_entity_filter(query, principal: "Principal", user):
    from app.security.access import build_access_entity_patterns

    exact_matches, like_patterns = build_access_entity_patterns(principal)
    access_entity_col = Memory.metadata_["access_entity"].as_string()
    legacy_owner_filter = and_(
        or_(Memory.metadata_.is_(None), access_entity_col.is_(None)),
        Memory.user_id == user.id,
    )
    access_clauses = [access_entity_col.in_(exact_matches), legacy_owner_filter]
    access_clauses.extend(access_entity_col.like(pattern) for pattern in like_patterns)
    return query.filter(or_(*access_clauses))


def _build_graph_access_filters(principal: "Principal | None", user_id: str) -> tuple[list[str], list[str]]:
    """Build access_entity filters for Neo4j graph queries."""
    if not principal:
        return [f"user:{user_id}"], []

    from app.security.access import build_access_entity_patterns

    exact_matches, like_patterns = build_access_entity_patterns(principal)
    prefixes = [p[:-1] for p in like_patterns if p.endswith("%")]
    return exact_matches, prefixes


def _get_accessible_memories(db, principal: "Principal | None", user, app, include_archived: bool = True):
    query = db.query(Memory).filter(Memory.state != MemoryState.deleted)
    if not include_archived:
        query = query.filter(Memory.state != MemoryState.archived)

    if principal:
        query = _apply_access_entity_filter(query, principal, user)
    else:
        query = query.filter(Memory.user_id == user.id)

    memories = query.all()

    if principal:
        from app.security.access import can_read_access_entity

        filtered = []
        for memory in memories:
            access_entity = memory.metadata_.get("access_entity") if memory.metadata_ else None
            if access_entity:
                if can_read_access_entity(principal, access_entity):
                    filtered.append(memory)
            else:
                if memory.user_id == user.id:
                    filtered.append(memory)
        memories = filtered

    return [memory for memory in memories if check_memory_access_permissions(db, memory, app.id)]


def _grant_matches_access_entity(grant: str, access_entity: str) -> bool:
    if not grant or ":" not in grant or not access_entity or ":" not in access_entity:
        return False

    grant_prefix, grant_path = grant.split(":", 1)
    access_prefix, access_path = access_entity.split(":", 1)

    if grant_prefix == "user":
        return access_prefix == "user" and access_path == grant_path

    if grant_prefix == "team":
        return access_prefix == "team" and access_path == grant_path

    if grant_prefix == "project":
        if access_prefix == "project" and access_path == grant_path:
            return True
        return access_prefix == "team" and access_path.startswith(f"{grant_path}/")

    if grant_prefix == "org":
        if access_prefix == "org" and access_path == grant_path:
            return True
        return access_prefix in ("project", "team") and access_path.startswith(f"{grant_path}/")

    return False


def _matching_grants_for_access_entity(principal: "Principal", access_entity: str) -> list[str]:
    grants = set(principal.claims.grants)
    grants.add(f"user:{principal.user_id}")
    return [
        grant for grant in sorted(grants)
        if _grant_matches_access_entity(grant, access_entity)
    ]


def _structured_memory_hint(message: str) -> str | None:
    if "access_entity is required for scope" in message:
        return (
            "Provide access_entity that matches the scope (team:<org>/<team>, "
            "project:<org>/<path>, org:<org>)."
        )
    if "Scope/access_entity mismatch" in message:
        return "Make access_entity prefix match scope (user:, team:, project:, org:)."
    if "Invalid access_entity format" in message:
        return "Use <prefix>:<value> (e.g., user:alice, team:org/team)."
    if "Tags must be a dictionary" in message:
        return "Pass tags as a dict or list of strings (e.g., tags={\"priority\": \"high\"})."
    if "Evidence must be a list" in message:
        return "Pass evidence as a list of strings or a comma-separated string."
    if "remove_tags must be a list" in message:
        return "Pass remove_tags as a list of strings or a comma-separated string."
    return None


def _format_structured_memory_error(error: StructuredMemoryError) -> dict:
    message = str(error)
    payload = {"error": message}
    hint = _structured_memory_hint(message)
    if hint:
        payload["hint"] = hint
    return payload


@mcp.tool(description="""Return the current auth context for this MCP session.

Returns:
- user_id: JWT subject (user id)
- org_id: Organization id from the token
- client_name: MCP client name from the session
- scopes: Granted OAuth scopes
- grants: access_entity grants (includes user:<sub>)
- security_enabled: Whether auth is enforced
""")
async def whoami() -> str:
    principal = principal_var.get(None)
    uid = user_id_var.get(None)
    client_name = client_name_var.get(None)
    org_id = org_id_var.get(None)

    if HAS_SECURITY and not principal:
        return json.dumps({
            "error": "Authentication required",
            "code": "MISSING_AUTH",
        })

    scopes = []
    grants = []
    if principal and principal.claims:
        scopes = sorted(list(principal.claims.scopes))
        grants = sorted(list(principal.claims.grants))
        uid = principal.user_id or uid
        org_id = principal.org_id or org_id

    return json.dumps(
        {
            "user_id": uid,
            "org_id": org_id,
            "client_name": client_name,
            "scopes": scopes,
            "grants": grants,
            "security_enabled": HAS_SECURITY,
        }
    )

@mcp.tool(description="""Return the active embedder configuration and runtime details.

Returns:
- provider: Configured embedder provider (e.g., "gemini")
- model: Embedder model name
- embedding_dims: Configured embedding dimensions
- output_dimensionality: Configured output dims (if set)
- class_name: Runtime embedder class
- vector_store_provider: Vector store provider name
- vector_store_dims: Vector store embedding dimensions (if available)
- memory_client_ready: Whether the memory client is initialized
- error: Error message if the memory client is unavailable
""")
async def get_embedder_info() -> str:
    scope_error = _check_tool_scope("memories:read")
    if scope_error:
        return scope_error

    memory_client = get_memory_client_safe()
    if not memory_client:
        return json.dumps({
            "memory_client_ready": False,
            "error": "Memory client is not available",
        })

    embedder = getattr(memory_client, "embedding_model", None)
    embedder_config = getattr(embedder, "config", None)

    provider = None
    try:
        provider = getattr(getattr(memory_client.config, "embedder", None), "provider", None)
    except Exception:
        provider = None

    model = getattr(embedder_config, "model", None) if embedder_config else None
    embedding_dims = getattr(embedder_config, "embedding_dims", None) if embedder_config else None
    output_dimensionality = getattr(embedder_config, "output_dimensionality", None) if embedder_config else None

    class_name = None
    if embedder:
        class_name = f"{embedder.__class__.__module__}.{embedder.__class__.__name__}"

    vector_store_provider = None
    vector_store_dims = None
    try:
        vector_store_provider = getattr(getattr(memory_client.config, "vector_store", None), "provider", None)
    except Exception:
        vector_store_provider = None

    if hasattr(getattr(memory_client, "vector_store", None), "embedding_model_dims"):
        vector_store_dims = memory_client.vector_store.embedding_model_dims
    else:
        try:
            vector_store_config = getattr(getattr(memory_client.config, "vector_store", None), "config", None)
            if hasattr(vector_store_config, "embedding_model_dims"):
                vector_store_dims = vector_store_config.embedding_model_dims
            elif isinstance(vector_store_config, dict):
                vector_store_dims = vector_store_config.get("embedding_model_dims")
        except Exception:
            vector_store_dims = None

    return json.dumps({
        "memory_client_ready": True,
        "provider": provider,
        "model": model,
        "embedding_dims": embedding_dims,
        "output_dimensionality": output_dimensionality,
        "class_name": class_name,
        "vector_store_provider": vector_store_provider,
        "vector_store_dims": vector_store_dims,
    })

@mcp.tool(
    description="""Add a new memory with structured metadata.

Use when: Documenting decisions, conventions, architecture, or learnings worth remembering.

Parameters:
- text: Memory content (required)
- category: decision | convention | architecture | workflow | etc. (required)
- scope: user | team | project | org (optional - derived from access_entity if not provided)
- entity: What/who this is about (e.g., "AuthService", "Backend Team")
- access_entity: Access control (e.g., "project:org/repo") - required for shared scopes
- tags: Key-value metadata (e.g., {"priority": "high"})
- evidence: References (e.g., ["ADR-014", "PR-123"])
- code_refs: Link to source code (e.g., [{"file_path": "/src/auth.ts", "line_start": 42}])
- async_mode: When true, return immediately with job_id and process in background (default: true)
- debug: When true, return detailed timing breakdown for performance analysis (default: false)

Returns: id of created memory, plus saved metadata. If debug=true, includes timing breakdown.

Example:
- add_memories(text="Use JWT for auth", category="decision", entity="AuthService", access_entity="project:cloudfactory/vgbk")
- add_memories(text="...", category="decision", debug=true)  # Returns timing info
""",
    annotations=ToolAnnotations(readOnlyHint=False, title="Add Memory")
)
async def add_memories(
    text: str,
    category: str,
    scope: str = None,  # PRD-13: Now optional - derived from access_entity if not provided
    artifact_type: str = None,
    artifact_ref: str = None,
    entity: str = None,
    source: str = "user",
    evidence: list = None,
    tags: dict = None,
    access_entity: str = None,
    code_refs: list = None,
    infer: bool = False,
    async_mode: bool = True,
    debug: bool = False,
) -> str:
    # Initialize debug timer
    timer = DebugTimer(enabled=debug)
    timer.start("total")

    # Check scope - requires memories:write
    scope_error = _check_tool_scope("memories:write")
    if scope_error:
        return scope_error

    uid = user_id_var.get(None)
    client_name = client_name_var.get(None)

    if not uid:
        return json.dumps({"error": "user_id not provided"})
    if not client_name:
        return json.dumps({"error": "client_name not provided"})

    principal = principal_var.get(None)
    if access_entity == "auto":
        access_entity = None

    # PRD-13: Derive scope from access_entity if not provided
    timer.start("access_resolution")
    effective_scope = scope
    if effective_scope is None and access_entity:
        from app.utils.structured_memory import derive_scope
        try:
            effective_scope = derive_scope(access_entity, explicit_scope=None)
        except ValueError:
            effective_scope = None

    if principal and (access_entity is None) and effective_scope:
        resolved_access_entity, options = resolve_access_entity_for_scope(
            principal=principal,
            scope=effective_scope,
            access_entity=access_entity,
        )
        if resolved_access_entity:
            access_entity = resolved_access_entity
        elif effective_scope in SHARED_SCOPES and options:
            return json.dumps({
                "error": f"access_entity is required for scope='{effective_scope}'. Multiple grants available.",
                "code": "ACCESS_ENTITY_AMBIGUOUS",
                "options": options,
                "hint": "Pick one of the available access_entity values.",
            })

    # Default access_entity to user:<uid> for personal scopes (including when scope is None)
    if access_entity is None and (effective_scope is None or effective_scope in ("user", "session")):
        access_entity = f"user:{uid}"

    # Access control check: verify principal can create memory with this access_entity
    if principal and access_entity:
        from app.security.access import check_create_access
        if not check_create_access(principal, access_entity):
            return json.dumps({
                "error": f"Access denied: you don't have permission to create memories with access_entity='{access_entity}'",
                "code": "FORBIDDEN",
            })
    timer.stop("access_resolution")

    # Validate and build structured memory
    timer.start("validation")
    try:
        clean_text, structured_metadata = build_structured_memory(
            text=text,
            category=category,
            scope=scope,
            artifact_type=artifact_type,
            artifact_ref=artifact_ref,
            entity=entity,
            access_entity=access_entity,
            source=source,
            evidence=evidence,
            tags=tags,
        )
    except StructuredMemoryError as e:
        return json.dumps(_format_structured_memory_error(e))

    # Validate and serialize code_refs into tags (Phase 1: tags-based storage)
    validated_code_refs = []
    if code_refs:
        try:
            validated_code_refs = validate_code_refs_input(code_refs)
            code_ref_tags = serialize_code_refs_to_tags(validated_code_refs)
            # Merge code_ref tags into existing tags
            existing_tags = structured_metadata.get("tags", {})
            if not isinstance(existing_tags, dict):
                existing_tags = {}
            structured_metadata["tags"] = {**existing_tags, **code_ref_tags}
        except CodeReferenceError as e:
            return json.dumps({
                "error": str(e),
                "code": "INVALID_CODE_REFS",
            })
    timer.stop("validation")

    combined_metadata = {
        "source_app": "openmemory",
        "mcp_client": client_name,
        **structured_metadata,
    }
    if principal:
        combined_metadata["org_id"] = principal.org_id

    # Get memory client safely
    memory_client = get_memory_client_safe()
    if not memory_client:
        return json.dumps({"error": "Memory system is currently unavailable. Please try again later."})

    if async_mode:
        summary = _summarize_add_job(structured_metadata, [])
        job_id = create_memory_job(requested_by=uid, summary=summary)
        asyncio.create_task(
            _run_add_memories_job(
                job_id=job_id,
                memory_client=memory_client,
                user_id=uid,
                client_name=client_name,
                clean_text=clean_text,
                structured_metadata=structured_metadata,
                combined_metadata=combined_metadata,
                infer=infer,
                debug=debug,
            )
        )
        response = {
            "status": "queued",
            "job_id": job_id,
        }
        if debug:
            timer.stop("total")
            response["debug_timing"] = timer.get_timing()
            response["debug_timing"]["note"] = "Async mode: detailed timing available via add_memories_status after job completes"
        return json.dumps(response)

    # Synchronous mode with detailed timing
    timer.start("mem0_add")
    formatted_response, valid_results, core_timing, graph_write_on_add = await asyncio.to_thread(
        _add_memories_core,
        memory_client=memory_client,
        user_id=uid,
        client_name=client_name,
        clean_text=clean_text,
        structured_metadata=structured_metadata,
        combined_metadata=combined_metadata,
        infer=infer,
        debug=debug,
    )
    timer.stop("mem0_add", {"infer": infer})

    # Merge core timing if available
    if core_timing:
        for name, value in core_timing.items():
            if isinstance(value, (int, float)):
                timer.record(name, value)
            elif isinstance(value, dict):
                # Nested timing breakdown (e.g., mem0_breakdown) - store as metadata
                timer.record(name, 0.0, metadata=value)

    if valid_results:
        graph_job_id = create_memory_job(
            requested_by=uid,
            summary=_summarize_graph_job(valid_results),
            job_type="graph_projection",
        )
        timer.start("graph_projection_trigger")
        asyncio.create_task(
            _run_graph_projection_async(
                valid_results=list(valid_results),
                user_id=uid,
                combined_metadata=dict(combined_metadata),
                graph_write_on_add=graph_write_on_add,
                graph_job_id=graph_job_id,
            )
        )
        timer.stop("graph_projection_trigger")

    timer.stop("total")

    # Add timing to response if debug mode
    if debug:
        formatted_response["debug_timing"] = timer.get_timing()
    if valid_results:
        formatted_response["graph_job_id"] = graph_job_id

    return json.dumps(formatted_response)


@mcp.tool(
    description="""Get status for an async memory job (add/update), or list all jobs.

Use when: add_memories or update_memory was called with async_mode=true and returned a job_id,
or when you want to see all running/recent jobs. Also returns graph projection
jobs created by add_memories.

Parameters:
- job_id: Job UUID returned from add_memories/update_memory (optional - if omitted, lists jobs)
- status_filter: Filter jobs by status when listing (optional). Options:
  - "running": Show queued and running jobs (default when listing)
  - "completed": Show succeeded jobs
  - "failed": Show failed jobs
  - "all": Show all jobs
  - or specific: "queued", "running", "succeeded", "failed"
- limit: Max jobs to return when listing (default: 20, max: 100)

Returns:
- If job_id provided: Single job with status, timestamps, summary, and result
- If job_id omitted: List of jobs matching the filter, sorted by newest first
""",
    annotations=ToolAnnotations(readOnlyHint=True, title="Add Memory Job Status")
)
async def add_memories_status(
    job_id: str | None = None,
    status_filter: str | None = None,
    limit: int = 20,
) -> str:
    # Check scope - requires memories:read
    scope_error = _check_tool_scope("memories:read")
    if scope_error:
        return scope_error

    uid = user_id_var.get(None)
    if not uid:
        return json.dumps({"error": "user_id not provided"})

    principal = principal_var.get(None)

    # If job_id provided, return single job status
    if job_id:
        job = get_memory_job(job_id)
        if not job:
            return json.dumps({"error": "job_id not found", "code": "JOB_NOT_FOUND"})

        if principal and job.get("requested_by") != principal.user_id:
            return json.dumps({
                "error": "Access denied: you don't have permission to view this job",
                "code": "FORBIDDEN",
            })

        return json.dumps(job)

    # No job_id provided - list jobs
    # Default to "running" filter when listing without explicit filter
    effective_filter = status_filter if status_filter else "running"

    # Clamp limit
    effective_limit = min(max(1, limit), 100)

    # Filter by user if principal available
    requested_by = principal.user_id if principal else None

    jobs = list_memory_jobs(
        requested_by=requested_by,
        status_filter=effective_filter,
        limit=effective_limit,
    )

    return json.dumps({
        "jobs": jobs,
        "count": len(jobs),
        "filter": effective_filter,
        "limit": effective_limit,
    })


@mcp.tool(
    description="""Retrieves architectural context, guidelines, and historical decisions from the knowledge base.

Use when: Understanding "Why" decisions were made, finding conventions, guidelines, or past context.

WARNING: Does NOT search live code. Memory results may be stale. Use this to understand 'Why' and 'How', but NEVER rely on it for 'Where' in code or exhaustive reference finding. For code tracing, use search_code_hybrid + Read tool instead.

Parameters:
- query: Search text (required)
- entity: Boost memories about this entity (e.g., "AuthService")
- category: Boost by category (decision, architecture, workflow, etc.)
- scope: Boost by scope (project, team, org)
- filter_tags: Hard filter by tag keys or key=value pairs (comma-separated)
- filter_evidence: Hard filter by evidence refs (comma-separated)
- filter_category, filter_scope, filter_artifact_type, filter_artifact_ref, filter_entity, filter_source, filter_access_entity, filter_app: Hard filters (comma-separated)
- filter_app: Hard filter by app/client name (e.g., "claude-code", "cursor") - filters by mcp_client field
- filter_mode: "all" or "any" for tag/evidence filters (default: "all")
- limit: Max results (default: 10, max: 50)
- created_after/before: Filter by date range (ISO format)
- recency_weight: Prioritize recent (0.0-1.0, default: 0.0)
- entity_max_hops: Max hops for bridge entity traversal (1-5, default: 2). Lower = faster but less coverage.
- relation_detail: Output verbosity ("none", "minimal", "standard", "full")
- debug: When true, return detailed timing breakdown for performance analysis (default: false)

Returns: results[] with id, memory, score, category, scope, entity, access_entity. If debug=true, includes timing breakdown.

IMPORTANT: For code-related memories, ALWAYS verify against actual code before answering.

Examples:
- search_memory(query="auth flow", entity="AuthService", limit=5)
- search_memory(query="Q4 decisions", created_after="2025-10-01T00:00:00Z")
- search_memory(query="...", debug=true)  # Returns timing info
- search_memory(query="*", filter_app="claude-code", limit=50)  # Find memories written via claude-code
""",
    annotations=ToolAnnotations(readOnlyHint=True, title="Search Memories")
)
async def search_memory(
    query: str,
    # BOOST context (soft ranking - does NOT exclude)
    category: str = None,
    scope: str = None,
    artifact_type: str = None,
    artifact_ref: str = None,
    entity: str = None,
    tags: str = None,
    # HARD FILTERS (applied before reranking)
    filter_tags: Any = None,
    filter_evidence: Any = None,
    filter_category: Any = None,
    filter_scope: Any = None,
    filter_artifact_type: Any = None,
    filter_artifact_ref: Any = None,
    filter_entity: Any = None,
    filter_source: Any = None,
    filter_access_entity: Any = None,
    filter_app: Any = None,
    filter_mode: str = "all",
    # RECENCY boost (soft ranking - does NOT exclude)
    recency_weight: float = 0.0,
    recency_halflife_days: int = 45,
    # DATE FILTERS (hard exclusion - these DO exclude)
    created_after: str = None,
    created_before: str = None,
    updated_after: str = None,
    updated_before: str = None,
    # EXCLUSIONS (hard exclusion)
    exclude_states: str = "deleted",
    exclude_tags: str = None,
    # CONTROL
    limit: int = 10,
    verbose: bool = False,
    # HYBRID RETRIEVAL (Phase 2: RRF Multi-Source Fusion)
    use_rrf: bool = True,
    graph_seed_count: int = 5,
    # QUERY ROUTING (Phase 3: Intelligent Query Routing)
    auto_route: bool = True,
    # ENTITY EXPANSION (controls bridge entity traversal depth)
    entity_max_hops: int = 2,
    # RELATION OUTPUT (controls meta_relations format)
    relation_detail: str = "standard",
    # DEBUG timing
    debug: bool = False,
) -> str:
    """Search memories with re-ranking. See tool description for full docs."""
    # Initialize debug timer
    timer = DebugTimer(enabled=debug)

    # Check scope - requires memories:read
    scope_error = _check_tool_scope("memories:read")
    if scope_error:
        return scope_error

    uid = user_id_var.get(None)
    client_name = client_name_var.get(None)

    if not uid:
        return json.dumps({"error": "user_id not provided"})
    if not client_name:
        return json.dumps({"error": "client_name not provided"})

    # Get memory client safely
    memory_client = get_memory_client_safe()
    if not memory_client:
        return json.dumps({"error": "Memory system is currently unavailable"})

    # Validate and cap parameters
    limit = min(max(1, limit), 50)
    recency_weight = min(max(0.0, recency_weight), 1.0)

    try:
        db = SessionLocal()
        try:
            timer.start("acl_check")
            # Get or create user and app
            user, app = get_user_and_app(db, user_id=uid, app_id=client_name)

            # ACL check - get accessible memory IDs based on access_entity grants
            principal = principal_var.get(None)
            app_accessible = _get_accessible_memories(db, principal, user, app, include_archived=True)
            accessible_memory_ids = [memory.id for memory in app_accessible]

            allowed = set(str(mid) for mid in accessible_memory_ids)

            # Build search context for boosting
            boost_tags = [t.strip() for t in tags.split(",")] if tags else []

            context = SearchContext(
                category=category,
                scope=scope,
                artifact_type=artifact_type,
                artifact_ref=artifact_ref,
                entity=entity,
                tags=boost_tags,
                recency_weight=recency_weight,
                recency_halflife_days=recency_halflife_days,
            )

            # Build exclusion filters
            filters = ExclusionFilters(
                exclude_states=[s.strip() for s in exclude_states.split(",")] if exclude_states else ["deleted"],
                exclude_tags=[t.strip() for t in exclude_tags.split(",")] if exclude_tags else [],
                boost_tags=boost_tags,
                created_after=parse_datetime(created_after),
                created_before=parse_datetime(created_before),
                updated_after=parse_datetime(updated_after),
                updated_before=parse_datetime(updated_before),
            )

            # Build hard filters (pre-search)
            try:
                filter_mode_normalized = (filter_mode or "all").strip().lower()
                if filter_mode_normalized not in ("any", "all"):
                    raise ValueError("filter_mode must be 'any' or 'all'")

                def normalize_filter_list(value, name):
                    if value is None:
                        return []
                    if isinstance(value, str):
                        return [v.strip() for v in value.split(",") if v.strip()]
                    if isinstance(value, (list, tuple)):
                        cleaned = []
                        for item in value:
                            if not isinstance(item, str):
                                raise ValueError(f"{name} items must be strings")
                            item = item.strip()
                            if item:
                                cleaned.append(item)
                        return cleaned
                    raise ValueError(f"{name} must be a string or list of strings")

                filter_tags_list = normalize_filter_list(filter_tags, "filter_tags")
                filter_evidence_list = normalize_filter_list(filter_evidence, "filter_evidence")
                filter_category_list = normalize_filter_list(filter_category, "filter_category")
                filter_scope_list = normalize_filter_list(filter_scope, "filter_scope")
                filter_artifact_type_list = normalize_filter_list(filter_artifact_type, "filter_artifact_type")
                filter_artifact_ref_list = normalize_filter_list(filter_artifact_ref, "filter_artifact_ref")
                filter_entity_list = normalize_filter_list(filter_entity, "filter_entity")
                filter_source_list = normalize_filter_list(filter_source, "filter_source")
                filter_access_entity_list = normalize_filter_list(filter_access_entity, "filter_access_entity")
                filter_app_list = normalize_filter_list(filter_app, "filter_app")
            except ValueError as e:
                return json.dumps({"error": str(e)})

            must_conditions = []
            should_conditions = []

            if not principal:
                must_conditions.append({"key": "user_id", "value": uid})

            def add_field_filter(field_name, values):
                if not values:
                    return
                if len(values) == 1:
                    must_conditions.append({"key": field_name, "value": values[0]})
                else:
                    must_conditions.append({"key": field_name, "any": values})

            add_field_filter("category", filter_category_list)
            add_field_filter("scope", filter_scope_list)
            add_field_filter("artifact_type", filter_artifact_type_list)
            add_field_filter("artifact_ref", filter_artifact_ref_list)
            add_field_filter("entity", filter_entity_list)
            add_field_filter("source", filter_source_list)
            add_field_filter("access_entity", filter_access_entity_list)
            add_field_filter("mcp_client", filter_app_list)  # filter_app maps to mcp_client in Qdrant payload

            if filter_evidence_list:
                if filter_mode_normalized == "all" and len(filter_evidence_list) > 1:
                    for value in filter_evidence_list:
                        must_conditions.append({"key": "evidence", "value": value})
                elif len(filter_evidence_list) == 1:
                    must_conditions.append({"key": "evidence", "value": filter_evidence_list[0]})
                else:
                    must_conditions.append({"key": "evidence", "any": filter_evidence_list})

            tag_conditions = []
            for token in filter_tags_list:
                if "=" in token:
                    key, value = token.split("=", 1)
                    key = key.strip()
                    value = value.strip()
                    if not key or not value:
                        raise ValueError("filter_tags entries must be 'key' or 'key=value'")
                    tag_conditions.append({"key": f"tags.{key}", "value": value})
                else:
                    key = token.strip()
                    if not key:
                        continue
                    tag_conditions.append({"key": f"tags.{key}", "value": True})

            if tag_conditions:
                if filter_mode_normalized == "any":
                    should_conditions.extend(tag_conditions)
                else:
                    must_conditions.extend(tag_conditions)

            vector_filters = None
            if must_conditions or should_conditions:
                vector_filters = {
                    "must": must_conditions,
                    "should": should_conditions,
                }

            timer.stop("acl_check", {"accessible_count": len(allowed)})

            # =========================================================
            # PHASE 3: Query Routing (Intelligent Route Selection)
            # =========================================================
            timer.start("query_routing")
            route = None
            query_analysis = None
            rrf_alpha = 0.6  # Default RRF alpha
            principal = principal_var.get(None)
            access_entities, access_entity_prefixes = _build_graph_access_filters(principal, uid)

            if auto_route and use_rrf:
                try:
                    from app.utils.query_router import (
                        analyze_query, get_routing_config, RouteType, get_rrf_alpha_for_route
                    )
                    routing_config = get_routing_config()

                    if routing_config.enabled:
                        query_analysis = analyze_query(
                            query,
                            uid,
                            routing_config,
                            access_entities=access_entities,
                            access_entity_prefixes=access_entity_prefixes,
                        )
                        route = query_analysis.route
                        rrf_alpha = get_rrf_alpha_for_route(route)

                        if verbose:
                            logging.debug(
                                f"Query routing: route={route.value}, "
                                f"entities={[e[0] for e in query_analysis.detected_entities]}, "
                                f"keywords={query_analysis.relationship_keywords}"
                            )
                except Exception as routing_error:
                    logging.warning(f"Query routing failed, using default: {routing_error}")
            timer.stop("query_routing", {"route": route.value if route else "default"})

            # =========================================================
            # PHASE 1: Vector Search
            # =========================================================
            timer.start("vector_embedding")
            # No metadata filters - we do re-ranking instead
            embeddings = memory_client.embedding_model.embed(query, "search")
            timer.stop("vector_embedding")

            timer.start("vector_search")
            search_limit = limit * 3  # Pool for reranking

            hits = memory_client.vector_store.search(
                query=query,
                vectors=embeddings,
                limit=search_limit,
                filters=vector_filters,
            )
            timer.stop("vector_search", {"hits": len(hits)})

            # =========================================================
            # PHASE 2: Graph Retrieval & RRF Fusion
            # =========================================================
            timer.start("graph_retrieval")
            graph_results = []
            rrf_stats = None
            rrf_enabled = (
                use_rrf
                and is_graph_enabled()
                and (route is None or route.value != "vector")  # Skip for VECTOR_ONLY
            )

            if rrf_enabled:
                try:
                    from app.graph.graph_ops import (
                        retrieve_via_similarity_graph, is_similarity_enabled
                    )
                    from app.utils.rrf_fusion import (
                        RRFFusion, RRFConfig, RetrievalResult
                    )

                    # Check if similarity edges are available
                    if is_similarity_enabled():
                        # Build seed list from top vector results
                        seed_ids = []
                        for h in hits:
                            if str(h.id) in allowed:
                                seed_ids.append(str(h.id))
                            if len(seed_ids) >= graph_seed_count:
                                break

                        if len(seed_ids) >= 2:  # Need at least 2 seeds
                            # Graph retrieval via OM_SIMILAR edges
                            graph_raw = retrieve_via_similarity_graph(
                                user_id=uid,
                                seed_memory_ids=seed_ids,
                                allowed_memory_ids=allowed,
                                limit=search_limit,
                                min_score=0.5,
                            )

                            # Convert to RetrievalResult format
                            graph_results = [
                                RetrievalResult(
                                    memory_id=r["id"],
                                    rank=rank,
                                    score=r.get("avgSimilarity", 0),
                                    source="graph",
                                    payload=r,
                                )
                                for rank, r in enumerate(graph_raw, 1)
                            ]

                            if verbose and graph_results:
                                logging.debug(
                                    f"Graph retrieval: {len(graph_results)} candidates "
                                    f"from {len(seed_ids)} seeds"
                                )
                except Exception as graph_error:
                    logging.warning(f"Graph retrieval failed, using vector-only: {graph_error}")
            timer.stop("graph_retrieval", {"candidates": len(graph_results)})

            # =========================================================
            # Entity-Aware Query Expansion (for GRAPH_PRIMARY route)
            # =========================================================
            timer.start("entity_expansion")
            # When 2+ entities detected and route is GRAPH_PRIMARY,
            # find bridge entities and expand retrieval
            bridge_entities = []

            if (
                rrf_enabled
                and query_analysis
                and len(query_analysis.detected_entities) >= 2
                and route is not None
                and route.value == "graph"  # GRAPH_PRIMARY
            ):
                try:
                    from app.graph.graph_ops import (
                        find_bridge_entities, retrieve_via_entity_graph
                    )

                    entity_names = [e[0] for e in query_analysis.detected_entities]
                    # Find bridge entities (entity_max_hops controls traversal depth)
                    # Validate and cap entity_max_hops (1-5 range)
                    capped_max_hops = min(max(1, entity_max_hops), 5)
                    bridge_entities = find_bridge_entities(
                        user_id=uid,
                        entity_names=entity_names,
                        max_bridges=5,
                        min_count=2,
                        max_hops=capped_max_hops,
                        access_entities=access_entities,
                        access_entity_prefixes=access_entity_prefixes,
                    )

                    # Expand entity list with bridges
                    expanded_names = entity_names.copy()
                    for bridge in bridge_entities:
                        if bridge["name"] not in expanded_names:
                            expanded_names.append(bridge["name"])

                    # Retrieve memories via entity graph
                    entity_raw = retrieve_via_entity_graph(
                        user_id=uid,
                        entity_names=expanded_names,
                        allowed_memory_ids=allowed,
                        limit=search_limit,
                        access_entities=access_entities,
                        access_entity_prefixes=access_entity_prefixes,
                    )

                    # Convert to RetrievalResult for RRF fusion
                    entity_graph_results = []
                    for rank, r in enumerate(entity_raw, 1):
                        # Avoid duplicates already in graph_results
                        if not any(gr.memory_id == r["id"] for gr in graph_results):
                            entity_graph_results.append(
                                RetrievalResult(
                                    memory_id=r["id"],
                                    rank=rank,
                                    score=min(1.0, r.get("matchedEntities", 1) / len(expanded_names)),
                                    source="entity_graph",
                                    payload=r,
                                )
                            )

                    # Merge entity_graph_results into graph_results for RRF
                    graph_results.extend(entity_graph_results)

                    if verbose and (bridge_entities or entity_graph_results):
                        logging.debug(
                            f"Entity expansion: {entity_names} + bridges {[b['name'] for b in bridge_entities]} "
                            f"-> {len(entity_graph_results)} additional memories"
                        )

                except Exception as entity_error:
                    logging.warning(f"Entity graph retrieval failed: {entity_error}")
            timer.stop("entity_expansion", {"bridges": len(bridge_entities)})

            # =========================================================
            # PHASE 1 (continued): Graph-Enhanced Reranking
            # =========================================================
            timer.start("graph_context")
            # Fetch graph context for graph boost calculation
            graph_context = None
            if is_graph_enabled():
                try:
                    from app.graph.graph_cache import fetch_graph_context
                    memory_ids_in_pool = [str(h.id) for h in hits if h.id and str(h.id) in allowed]
                    access_entities, access_entity_prefixes = _build_graph_access_filters(principal, uid)
                    graph_context = fetch_graph_context(
                        memory_ids=memory_ids_in_pool,
                        user_id=uid,
                        context_tags=boost_tags,
                        access_entities=access_entities,
                        access_entity_prefixes=access_entity_prefixes,
                    )
                except Exception as cache_error:
                    logging.warning(f"Graph context fetch failed: {cache_error}")
            timer.stop("graph_context")

            # Process results
            timer.start("result_processing")
            results = []
            core_keys = {"data", "hash", "created_at", "updated_at"}

            for h in hits:
                memory_id = h.id
                semantic_score = h.score
                payload = h.payload

                # ACL check
                if memory_id is None or str(memory_id) not in allowed:
                    continue

                # Extract metadata
                metadata = {k: v for k, v in payload.items() if k not in core_keys}
                stored_tags = normalize_tags(metadata.get("tags", {}))

                # Exclusion check (states, tags, date ranges)
                excluded, reason = should_exclude(payload, metadata, stored_tags, filters)
                if excluded:
                    continue

                # Boost calculation (with graph context if available)
                boost, boost_breakdown = compute_boost(
                    metadata=metadata,
                    stored_tags=stored_tags,
                    context=context,
                    created_at_str=payload.get("created_at"),
                    graph_context=graph_context,
                    memory_id=str(memory_id),
                )

                final_score = calculate_final_score(semantic_score, boost)

                access_entity = metadata.get("access_entity")
                result_entry = {
                    "id": memory_id,
                    "memory": payload.get("data"),
                    "access_entity": access_entity,
                    "scores": {
                        "semantic": round(semantic_score, 4),
                        "boost": round(boost, 4),
                        "final": round(final_score, 4),
                    },
                    "metadata": {
                        "category": metadata.get("category"),
                        "scope": metadata.get("scope"),
                        "artifact_type": metadata.get("artifact_type"),
                        "artifact_ref": metadata.get("artifact_ref"),
                        "source": metadata.get("source"),
                        "evidence": metadata.get("evidence"),
                        "tags": stored_tags,
                        "entity": metadata.get("entity"),
                        "access_entity": access_entity,
                    },
                    "created_at": payload.get("created_at"),
                    "updated_at": payload.get("updated_at"),
                }

                if verbose and principal and access_entity:
                    result_entry["visibility_reason"] = {
                        "access_entity": access_entity,
                        "matched_grants": _matching_grants_for_access_entity(principal, access_entity),
                    }

                results.append(result_entry)
            timer.stop("result_processing", {"processed": len(results)})

            # =========================================================
            # PHASE 2 (continued): RRF Fusion
            # =========================================================
            timer.start("rrf_fusion")
            # If we have graph results, fuse them with vector results using RRF
            if graph_results:
                try:
                    from app.utils.rrf_fusion import RRFFusion, RRFConfig, RetrievalResult

                    # Convert vector results to RetrievalResult format
                    vector_results_for_rrf = []
                    results_by_id = {str(r["id"]): r for r in results}
                    for rank, r in enumerate(results, 1):
                        vector_results_for_rrf.append(RetrievalResult(
                            memory_id=str(r["id"]),
                            rank=rank,
                            score=r["scores"]["final"],
                            source="vector",
                            payload=r,
                        ))

                    # Create RRF fusion with configured alpha
                    fusion = RRFFusion(RRFConfig(alpha=rrf_alpha))
                    fused_with_stats = fusion.fuse_with_stats(
                        vector_results_for_rrf, graph_results
                    )
                    fused_results = fused_with_stats["results"]
                    rrf_stats = fused_with_stats["stats"]

                    # Process graph-only results that weren't in vector results
                    graph_only_ids = set()
                    for fr in fused_results:
                        if fr.memory_id not in results_by_id:
                            graph_only_ids.add(fr.memory_id)

                    # Fetch and process graph-only results
                    if graph_only_ids:
                        # Get payloads from graph results
                        graph_payload_map = {r.memory_id: r.payload for r in graph_results}

                        for fr in fused_results:
                            if fr.memory_id in graph_only_ids:
                                gp = graph_payload_map.get(fr.memory_id, {})
                                access_entity = gp.get("access_entity")
                                # Create result entry for graph-only hit
                                graph_entry = {
                                    "id": fr.memory_id,
                                    "memory": gp.get("content"),
                                    "access_entity": access_entity,
                                    "scores": {
                                        "semantic": 0.0,  # No vector score
                                        "boost": 0.0,
                                        "final": fr.rrf_score,
                                        "rrf": fr.rrf_score,
                                        "graph_similarity": fr.original_score,
                                    },
                                    "metadata": {
                                        "category": gp.get("category"),
                                        "scope": gp.get("scope"),
                                        "artifact_type": gp.get("artifact_type"),
                                        "artifact_ref": gp.get("artifact_ref"),
                                        "source": gp.get("source"),
                                        "evidence": gp.get("evidence"),
                                        "tags": {},
                                        "entity": None,
                                        "access_entity": access_entity,
                                    },
                                    "created_at": None,
                                    "updated_at": None,
                                    "source": "graph",
                                    "graph_info": {
                                        "seed_connections": gp.get("seedConnections"),
                                        "avg_similarity": gp.get("avgSimilarity"),
                                    },
                                }
                                if verbose and principal and access_entity:
                                    graph_entry["visibility_reason"] = {
                                        "access_entity": access_entity,
                                        "matched_grants": _matching_grants_for_access_entity(principal, access_entity),
                                    }
                                results_by_id[fr.memory_id] = graph_entry

                    # Re-order results by RRF score
                    reordered = []
                    for fr in fused_results:
                        if fr.memory_id in results_by_id:
                            r = results_by_id[fr.memory_id]
                            # Add RRF score to existing results
                            r["scores"]["rrf"] = fr.rrf_score
                            r["in_both_sources"] = fr.in_both
                            reordered.append(r)

                    results = reordered

                    if verbose:
                        logging.debug(
                            f"RRF fusion: {rrf_stats['fused_total']} total, "
                            f"{rrf_stats['in_both_sources']} in both sources"
                        )

                except Exception as rrf_error:
                    logging.warning(f"RRF fusion failed, using vector-only ranking: {rrf_error}")
                    # Fall back to original vector-only sorting
                    results.sort(key=lambda x: x["scores"]["final"], reverse=True)
            else:
                # No graph results - standard vector-only sorting
                results.sort(key=lambda x: x["scores"]["final"], reverse=True)
            timer.stop("rrf_fusion", {"used_graph": bool(graph_results)})

            # Apply limit
            results = results[:limit]

            timer.start("access_logging")
            # 5. Access logging
            for r in results:
                if r.get("id"):
                    try:
                        access_log = MemoryAccessLog(
                            memory_id=uuid.UUID(str(r["id"])),
                            app_id=app.id,
                            access_type="search",
                            metadata_={
                                "query": query,
                                "semantic_score": r["scores"]["semantic"],
                                "boost": r["scores"]["boost"],
                                "final_score": r["scores"]["final"],
                                "recency_weight": recency_weight,
                            }
                        )
                        db.add(access_log)
                    except Exception as log_error:
                        logging.warning(f"Failed to log access: {log_error}")

            db.commit()
            timer.stop("access_logging")

            timer.start("response_format")
            # Format response (lean by default, verbose for debugging)
            if verbose:
                # Build verbose response metadata
                context_applied = {}
                if category:
                    context_applied["category"] = category
                if scope:
                    context_applied["scope"] = scope
                if artifact_type:
                    context_applied["artifact_type"] = artifact_type
                if artifact_ref:
                    context_applied["artifact_ref"] = artifact_ref
                if entity:
                    context_applied["entity"] = entity
                if boost_tags:
                    context_applied["tags"] = boost_tags
                if recency_weight > 0:
                    context_applied["recency"] = {
                        "weight": recency_weight,
                        "halflife_days": recency_halflife_days,
                    }

                filters_applied = {}
                if filters.created_after:
                    filters_applied["created_after"] = created_after
                if filters.created_before:
                    filters_applied["created_before"] = created_before
                if filters.updated_after:
                    filters_applied["updated_after"] = updated_after
                if filters.updated_before:
                    filters_applied["updated_before"] = updated_before
                if filters.exclude_tags:
                    filters_applied["exclude_tags"] = filters.exclude_tags
                hard_filters_applied = {}
                if filter_category_list:
                    hard_filters_applied["category"] = filter_category_list
                if filter_scope_list:
                    hard_filters_applied["scope"] = filter_scope_list
                if filter_artifact_type_list:
                    hard_filters_applied["artifact_type"] = filter_artifact_type_list
                if filter_artifact_ref_list:
                    hard_filters_applied["artifact_ref"] = filter_artifact_ref_list
                if filter_entity_list:
                    hard_filters_applied["entity"] = filter_entity_list
                if filter_source_list:
                    hard_filters_applied["source"] = filter_source_list
                if filter_access_entity_list:
                    hard_filters_applied["access_entity"] = filter_access_entity_list
                if filter_app_list:
                    hard_filters_applied["app"] = filter_app_list
                if filter_evidence_list:
                    hard_filters_applied["evidence"] = filter_evidence_list
                if filter_tags_list:
                    hard_filters_applied["tags"] = filter_tags_list
                if hard_filters_applied:
                    hard_filters_applied["mode"] = filter_mode_normalized
                    filters_applied["hard_filters"] = hard_filters_applied

                response = format_search_results(
                    results=results,
                    verbose=True,
                    query=query,
                    context_applied=context_applied if context_applied else None,
                    filters_applied=filters_applied if filters_applied else None,
                    total_candidates=len(hits),
                )

                # Add hybrid retrieval info to verbose response
                if use_rrf or auto_route:
                    hybrid_info = {}
                    if query_analysis:
                        hybrid_info["routing"] = {
                            "route": route.value if route else "hybrid",
                            "detected_entities": [e[0] for e in query_analysis.detected_entities],
                            "relationship_keywords": query_analysis.relationship_keywords,
                            "confidence": query_analysis.confidence,
                            "analysis_time_ms": query_analysis.analysis_time_ms,
                        }
                    if rrf_stats:
                        hybrid_info["rrf"] = {
                            "alpha": rrf_alpha,
                            "vector_candidates": rrf_stats.get("vector_candidates", 0),
                            "graph_candidates": rrf_stats.get("graph_candidates", 0),
                            "in_both_sources": rrf_stats.get("in_both_sources", 0),
                        }
                    if graph_context and graph_context.available:
                        hybrid_info["graph_boost"] = {
                            "max_pagerank": graph_context.max_pagerank,
                            "max_degree": graph_context.max_degree,
                            "max_cluster_size": graph_context.max_cluster_size,
                        }
                    # Add entity expansion info if bridges were found
                    if bridge_entities and query_analysis:
                        hybrid_info["entity_expansion"] = {
                            "detected_entities": [e[0] for e in query_analysis.detected_entities],
                            "bridge_entities": [b["name"] for b in bridge_entities],
                            "expanded_count": len(query_analysis.detected_entities) + len(bridge_entities),
                        }
                    if hybrid_info:
                        response["hybrid_retrieval"] = hybrid_info
            else:
                # Lean format (default) - optimized for LLM processing
                response = format_search_results(results=results, verbose=False)
            timer.stop("response_format")

            # Enrich with graph relations (only if graph is available)
            # This adds two new optional fields without changing existing response shape
            # The relation_detail parameter controls the output format:
            #   - "none": No meta_relations (minimal tokens)
            #   - "minimal": Only artifact + similar IDs
            #   - "standard": artifact + similar + entities + tags + evidence (default, compact)
            #   - "full": Verbose format with all relation types (for debugging)
            try:
                # Skip meta_relations if relation_detail is "none"
                if relation_detail != "none":
                    # Get memory IDs from results for meta_relations lookup
                    memory_ids = [str(r.get("id")) for r in results if r.get("id")]

                    # 1. meta_relations: deterministic metadata relations from Neo4j OM_* projection
                    timer.start("neo4j_om_meta_relations")
                    if memory_ids and is_graph_enabled():
                        raw_meta_relations = get_meta_relations_for_memories(memory_ids)
                        if raw_meta_relations:
                            if relation_detail == "full":
                                # Full verbose format (for debugging)
                                response["meta_relations"] = raw_meta_relations
                            else:
                                # Compact format (standard or minimal)
                                compact_relations = {}
                                for memory_id, relations in raw_meta_relations.items():
                                    compact = format_compact_relations(relations)
                                    if compact:
                                        if relation_detail == "minimal":
                                            # Only keep artifact(s) and similar
                                            compact = {
                                                k: v for k, v in compact.items()
                                                if k in ("artifact", "artifacts", "similar")
                                            }
                                        if compact:  # Still has content after filtering
                                            compact_relations[memory_id] = compact
                                if compact_relations:
                                    response["meta_relations"] = compact_relations
                    timer.stop("neo4j_om_meta_relations", {"memory_count": len(memory_ids) if memory_ids else 0})

                # 2. relations: Mem0 Graph Memory relations (LLM-extracted entities)
                timer.start("neo4j_om_mem0_relations")
                if is_mem0_graph_enabled():
                    graph_relations = get_graph_relations(
                        query=query,
                        user_id=uid,
                        limit=10,
                    )
                    if graph_relations:
                        response["relations"] = graph_relations
                timer.stop("neo4j_om_mem0_relations")

            except Exception as graph_error:
                logging.warning(f"Graph enrichment failed: {graph_error}")
                # Don't fail the search, just skip graph enrichment

            # Add verification instruction for code-related memories
            response["_instructions"] = "VERIFY: For code-related memories, read the actual file before answering."

            # Add debug timing if requested
            timing_data = timer.get_timing()
            if timing_data:
                response["_debug_timing"] = timing_data

            return json.dumps(response, default=str)

        finally:
            db.close()

    except Exception as e:
        logging.error(f"Error in search_memory: {e}", exc_info=True)
        return json.dumps({"error": f"Error searching memories: {str(e)}"})


@mcp.tool(
    description="""Find memories related by metadata (tags, entity, category, etc.).

Use when: Exploring connections based on shared metadata dimensions.

NOT for: Semantic similarity - use graph_similar_memories instead.

Parameters:
- memory_id: UUID of seed memory (required)
- via: Dimensions to traverse (tag, entity, category, scope, etc.)
- limit: Max results (default: 10)

Returns: related[] with shared_relations and shared_count.
""",
    annotations=ToolAnnotations(readOnlyHint=True, title="Related Memories (Metadata)")
)
async def graph_related_memories(
    memory_id: str,
    via: str = None,
    limit: int = 10,
) -> str:
    uid = user_id_var.get(None)
    client_name = client_name_var.get(None)

    if not uid:
        return json.dumps({"error": "user_id not provided"})
    if not client_name:
        return json.dumps({"error": "client_name not provided"})

    if not is_graph_enabled():
        return json.dumps(
            {
                "seed_memory_id": memory_id,
                "related": [],
                "count": 0,
                "graph_enabled": False,
            }
        )

    limit = min(max(1, int(limit or 10)), 100)

    try:
        db = SessionLocal()
        try:
            user, app = get_user_and_app(db, user_id=uid, app_id=client_name)

            principal = principal_var.get(None)
            accessible_memories = _get_accessible_memories(db, principal, user, app, include_archived=True)
            allowed = set(str(memory.id) for memory in accessible_memories)

            if memory_id not in allowed:
                return json.dumps({"error": f"Memory '{memory_id}' not found or not accessible"})

            access_entities, access_entity_prefixes = _build_graph_access_filters(principal, uid)
            related = find_related_memories_in_graph(
                memory_id=memory_id,
                user_id=uid,
                allowed_memory_ids=list(allowed),
                access_entities=access_entities,
                access_entity_prefixes=access_entity_prefixes,
                via=via,
                limit=limit,
            )

            # Access log (best-effort)
            try:
                db.add(
                    MemoryAccessLog(
                        memory_id=uuid.UUID(memory_id),
                        app_id=app.id,
                        access_type="graph_related",
                        metadata_={"via": via, "limit": limit},
                    )
                )
                db.commit()
            except Exception as log_error:
                logging.warning(f"Failed to log graph_related access: {log_error}")

            return json.dumps(
                {
                    "seed_memory_id": memory_id,
                    "related": related,
                    "count": len(related),
                    "graph_enabled": True,
                },
                default=str,
            )
        finally:
            db.close()
    except Exception as e:
        logging.exception(f"Error in graph_related_memories: {e}")
        return json.dumps({"error": f"Error finding related memories: {str(e)}"})


@mcp.tool(
    description="""Get neighborhood subgraph around a memory.

Use when: Visualizing or exploring memory connections in graph form.

Parameters:
- memory_id: UUID of seed memory (required)
- depth: 1 (direct) or 2 (include connected memories, default: 2)
- via: Dimensions to traverse
- related_limit: Max related nodes (default: 25)
""",
    annotations=ToolAnnotations(readOnlyHint=True, title="Memory Subgraph")
)
async def graph_subgraph(
    memory_id: str,
    depth: int = 2,
    via: str = None,
    related_limit: int = 25,
) -> str:
    uid = user_id_var.get(None)
    client_name = client_name_var.get(None)

    if not uid:
        return json.dumps({"error": "user_id not provided"})
    if not client_name:
        return json.dumps({"error": "client_name not provided"})

    if not is_graph_enabled():
        return json.dumps(
            {
                "seed_memory_id": memory_id,
                "nodes": [],
                "edges": [],
                "related": [],
                "graph_enabled": False,
            }
        )

    depth = max(1, min(int(depth or 2), 2))
    related_limit = max(0, min(int(related_limit or 25), 200))

    try:
        db = SessionLocal()
        try:
            user, app = get_user_and_app(db, user_id=uid, app_id=client_name)

            principal = principal_var.get(None)
            accessible_memories = _get_accessible_memories(db, principal, user, app, include_archived=True)
            allowed = set(str(memory.id) for memory in accessible_memories)

            if memory_id not in allowed:
                return json.dumps({"error": f"Memory '{memory_id}' not found or not accessible"})

            access_entities, access_entity_prefixes = _build_graph_access_filters(principal, uid)
            subgraph = get_memory_subgraph_from_graph(
                memory_id=memory_id,
                user_id=uid,
                allowed_memory_ids=list(allowed),
                access_entities=access_entities,
                access_entity_prefixes=access_entity_prefixes,
                depth=depth,
                via=via,
                related_limit=related_limit,
            )
            if subgraph is None:
                return json.dumps({"error": f"Memory '{memory_id}' not found in graph"})

            # Access log (best-effort)
            try:
                db.add(
                    MemoryAccessLog(
                        memory_id=uuid.UUID(memory_id),
                        app_id=app.id,
                        access_type="graph_subgraph",
                        metadata_={"via": via, "depth": depth, "related_limit": related_limit},
                    )
                )
                db.commit()
            except Exception as log_error:
                logging.warning(f"Failed to log graph_subgraph access: {log_error}")

            subgraph["graph_enabled"] = True
            return json.dumps(subgraph, default=str)
        finally:
            db.close()
    except Exception as e:
        logging.exception(f"Error in graph_subgraph: {e}")
        return json.dumps({"error": f"Error building subgraph: {str(e)}"})


@mcp.tool(
    description="""Aggregate memories by dimension (category, entity, scope, etc.).

Use when: Getting overview counts - e.g., how many decisions vs workflows.

Parameters:
- group_by: category | scope | entity | tag | artifact_type | etc. (required)
- limit: Max buckets (default: 20)
""",
    annotations=ToolAnnotations(readOnlyHint=True, title="Aggregate Memories")
)
async def graph_aggregate(group_by: str, limit: int = 20) -> str:
    uid = user_id_var.get(None)
    client_name = client_name_var.get(None)

    if not uid:
        return json.dumps({"error": "user_id not provided"})
    if not client_name:
        return json.dumps({"error": "client_name not provided"})

    if not is_graph_enabled():
        return json.dumps({"group_by": group_by, "buckets": [], "count": 0, "graph_enabled": False})

    limit = max(1, min(int(limit or 20), 200))

    try:
        db = SessionLocal()
        try:
            user, app = get_user_and_app(db, user_id=uid, app_id=client_name)

            principal = principal_var.get(None)
            accessible_memories = _get_accessible_memories(db, principal, user, app, include_archived=True)
            allowed = [str(memory.id) for memory in accessible_memories]
            access_entities, access_entity_prefixes = _build_graph_access_filters(principal, uid)

            buckets = aggregate_memories_in_graph(
                user_id=uid,
                group_by=group_by,
                allowed_memory_ids=allowed,
                limit=limit,
                access_entities=access_entities,
                access_entity_prefixes=access_entity_prefixes,
            )

            return json.dumps(
                {
                    "group_by": group_by,
                    "buckets": buckets,
                    "count": len(buckets),
                    "graph_enabled": True,
                },
                default=str,
            )
        finally:
            db.close()
    except ValueError as e:
        return json.dumps({"error": str(e)})
    except Exception as e:
        logging.exception(f"Error in graph_aggregate: {e}")
        return json.dumps({"error": f"Error aggregating memories: {str(e)}"})


@mcp.tool(
    description="""Find tags that frequently appear together.

Use when: Discovering tag patterns or finding related concepts.

Parameters:
- limit: Max tag pairs (default: 20)
- min_count: Minimum co-occurrence (default: 2)
""",
    annotations=ToolAnnotations(readOnlyHint=True, title="Tag Co-occurrence")
)
async def graph_tag_cooccurrence(limit: int = 20, min_count: int = 2, sample_size: int = 3) -> str:
    uid = user_id_var.get(None)
    client_name = client_name_var.get(None)

    if not uid:
        return json.dumps({"error": "user_id not provided"})
    if not client_name:
        return json.dumps({"error": "client_name not provided"})

    if not is_graph_enabled():
        return json.dumps({"pairs": [], "count": 0, "graph_enabled": False})

    limit = max(1, min(int(limit or 20), 200))
    min_count = max(1, int(min_count or 1))
    sample_size = max(0, min(int(sample_size or 0), 10))

    try:
        db = SessionLocal()
        try:
            user, app = get_user_and_app(db, user_id=uid, app_id=client_name)

            principal = principal_var.get(None)
            accessible_memories = _get_accessible_memories(db, principal, user, app, include_archived=True)
            allowed = [str(memory.id) for memory in accessible_memories]
            access_entities, access_entity_prefixes = _build_graph_access_filters(principal, uid)

            pairs = tag_cooccurrence_in_graph(
                user_id=uid,
                allowed_memory_ids=allowed,
                limit=limit,
                min_count=min_count,
                sample_size=sample_size,
                access_entities=access_entities,
                access_entity_prefixes=access_entity_prefixes,
            )

            return json.dumps(
                {"pairs": pairs, "count": len(pairs), "graph_enabled": True},
                default=str,
            )
        finally:
            db.close()
    except Exception as e:
        logging.exception(f"Error in graph_tag_cooccurrence: {e}")
        return json.dumps({"error": f"Error computing tag co-occurrence: {str(e)}"})


@mcp.tool(
    description="""Find shortest path between two entities through memories.

Use when: Understanding how two concepts/people/services are connected.

Parameters:
- entity_a: First entity (required)
- entity_b: Second entity (required)
- max_hops: Maximum path length (default: 6)
""",
    annotations=ToolAnnotations(readOnlyHint=True, title="Path Between Entities")
)
async def graph_path_between_entities(entity_a: str, entity_b: str, max_hops: int = 6) -> str:
    uid = user_id_var.get(None)
    client_name = client_name_var.get(None)

    if not uid:
        return json.dumps({"error": "user_id not provided"})
    if not client_name:
        return json.dumps({"error": "client_name not provided"})

    if not is_graph_enabled():
        return json.dumps({"path": None, "graph_enabled": False})

    max_hops = max(2, min(int(max_hops or 6), 12))

    try:
        db = SessionLocal()
        try:
            user, app = get_user_and_app(db, user_id=uid, app_id=client_name)

            principal = principal_var.get(None)
            accessible_memories = _get_accessible_memories(db, principal, user, app, include_archived=True)
            allowed = [str(memory.id) for memory in accessible_memories]
            access_entities, access_entity_prefixes = _build_graph_access_filters(principal, uid)

            path = path_between_entities_in_graph(
                user_id=uid,
                entity_a=entity_a,
                entity_b=entity_b,
                allowed_memory_ids=allowed,
                max_hops=max_hops,
                access_entities=access_entities,
                access_entity_prefixes=access_entity_prefixes,
            )

            return json.dumps({"path": path, "graph_enabled": True}, default=str)
        finally:
            db.close()
    except Exception as e:
        logging.exception(f"Error in graph_path_between_entities: {e}")
        return json.dumps({"error": f"Error finding path: {str(e)}"})


@mcp.tool(
    description="""List all accessible memories.

Use when: Getting an overview of what's stored, or when you need to see everything.

NOT for: Searching or filtering - use search_memory with query/filters instead.

Returns: All memories with id, content, category, scope, entity, access_entity, timestamps.

NOTE: Returns potentially large result set. For targeted retrieval, use search_memory.
""",
    annotations=ToolAnnotations(readOnlyHint=True, title="List All Memories")
)
async def list_memories() -> str:
    # Check scope - requires memories:read
    scope_error = _check_tool_scope("memories:read")
    if scope_error:
        return scope_error

    uid = user_id_var.get(None)
    client_name = client_name_var.get(None)
    if not uid:
        return "Error: user_id not provided"
    if not client_name:
        return "Error: client_name not provided"

    try:
        db = SessionLocal()
        try:
            # Get or create user and app
            user, app = get_user_and_app(db, user_id=uid, app_id=client_name)

            principal = principal_var.get(None)
            accessible_memories = _get_accessible_memories(db, principal, user, app, include_archived=False)
            raw_memories = []

            for memory in accessible_memories:
                md = memory.metadata_ or {}
                raw_memories.append(
                    {
                        "id": str(memory.id),
                        "memory": memory.content,
                        "category": md.get("category"),
                        "scope": md.get("scope"),
                        "artifact_type": md.get("artifact_type"),
                        "artifact_ref": md.get("artifact_ref"),
                        "entity": md.get("entity"),
                        "access_entity": md.get("access_entity"),
                        "source": md.get("source"),
                        "evidence": md.get("evidence"),
                        "tags": md.get("tags"),
                        "created_at": memory.created_at.isoformat() if memory.created_at else None,
                        "updated_at": memory.updated_at.isoformat() if memory.updated_at else None,
                    }
                )

                access_log = MemoryAccessLog(
                    memory_id=memory.id,
                    app_id=app.id,
                    access_type="list",
                    metadata_={},
                )
                db.add(access_log)
            db.commit()

            # Apply lean formatting
            formatted_memories = format_memory_list(raw_memories)
            return json.dumps(formatted_memories, indent=2)
        finally:
            db.close()
    except Exception as e:
        logging.exception(f"Error getting memories: {e}")
        return f"Error getting memories: {e}"


@mcp.tool(
    description="""Retrieve a single memory by its UUID.

Use when: Fetching full details of a specific memory, verifying updates, or following evidence chains.

Parameters:
- memory_id: UUID of the memory (required)

Returns: Full memory with id, content, category, scope, entity, access_entity, tags, evidence, timestamps.

Errors: NOT_FOUND (doesn't exist or no access), INVALID_INPUT (bad UUID format)
""",
    annotations=ToolAnnotations(readOnlyHint=True, title="Get Memory")
)
async def get_memory(memory_id: str) -> str:
    """Retrieve a single memory by UUID with all fields."""
    # Check scope - requires memories:read
    scope_error = _check_tool_scope("memories:read")
    if scope_error:
        return scope_error

    uid = user_id_var.get(None)
    client_name = client_name_var.get(None)

    if not uid:
        return json.dumps({"error": "user_id not provided"})
    if not client_name:
        return json.dumps({"error": "client_name not provided"})

    # Validate UUID format
    try:
        memory_uuid = uuid.UUID(memory_id)
    except (ValueError, TypeError):
        return json.dumps({
            "error": "Invalid memory_id format",
            "code": "INVALID_INPUT",
        })

    try:
        db = SessionLocal()
        try:
            # Get or create user and app
            user, app = get_user_and_app(db, user_id=uid, app_id=client_name)

            # Query the memory
            memory = db.query(Memory).filter(Memory.id == memory_uuid).first()

            if not memory:
                return json.dumps({
                    "error": "Memory not found",
                    "code": "NOT_FOUND",
                })

            if memory.state == MemoryState.deleted:
                return json.dumps({
                    "error": "Memory not found",
                    "code": "NOT_FOUND",
                })

            # Access control check
            principal = principal_var.get(None)
            md = memory.metadata_ or {}
            access_entity = md.get("access_entity")

            if access_entity:
                # Multi-user routing: check if principal has access to the access_entity
                if principal:
                    from app.security.access import can_read_access_entity
                    if not can_read_access_entity(principal, access_entity):
                        # Return NOT_FOUND to avoid leaking existence (security best practice)
                        return json.dumps({
                            "error": "Memory not found",
                            "code": "NOT_FOUND",
                        })
                else:
                    # No principal but memory has access_entity - check legacy owner access
                    if memory.user_id != user.id:
                        return json.dumps({
                            "error": "Memory not found",
                            "code": "NOT_FOUND",
                        })
            else:
                # Legacy behavior: only owner can access memories without access_entity
                if memory.user_id != user.id:
                    return json.dumps({
                        "error": "Memory not found",
                        "code": "NOT_FOUND",
                    })

            # Build raw memory object for formatting
            raw_memory = {
                "id": str(memory.id),
                "memory": memory.content,
                "category": md.get("category"),
                "scope": md.get("scope"),
                "artifact_type": md.get("artifact_type"),
                "artifact_ref": md.get("artifact_ref"),
                "entity": md.get("entity"),
                "access_entity": md.get("access_entity"),
                "source": md.get("source"),
                "evidence": md.get("evidence"),
                "tags": md.get("tags"),
                "created_at": memory.created_at.isoformat() if memory.created_at else None,
                "updated_at": memory.updated_at.isoformat() if memory.updated_at else None,
            }

            # Create access log
            access_log = MemoryAccessLog(
                memory_id=memory.id,
                app_id=app.id,
                access_type="get",
                metadata_={},
            )
            db.add(access_log)
            db.commit()

            # Apply lean formatting (without score since this is not a search result)
            from app.utils.response_format import format_memory_result
            formatted_memory = format_memory_result(raw_memory, include_score=False)

            return json.dumps(formatted_memory, indent=2)
        finally:
            db.close()
    except Exception as e:
        logging.exception(f"Error getting memory: {e}")
        return json.dumps({"error": f"Error getting memory: {e}"})


@mcp.tool(
    description="""Delete specific memories by their IDs.

Use when: Removing incorrect, outdated, or duplicate memories.

Parameters:
- memory_ids: List of memory UUIDs to delete

WARNING: This action is IRREVERSIBLE. For soft-delete, use update_memory to add a "deprecated" tag instead.

NOTE: You can only delete memories you have access to based on access_entity permissions.
""",
    annotations=ToolAnnotations(readOnlyHint=False, destructiveHint=True, title="Delete Memories")
)
async def delete_memories(memory_ids: list[str]) -> str:
    # Check scope - requires memories:delete
    scope_error = _check_tool_scope("memories:delete")
    if scope_error:
        return scope_error

    uid = user_id_var.get(None)
    client_name = client_name_var.get(None)
    if not uid:
        return "Error: user_id not provided"
    if not client_name:
        return "Error: client_name not provided"

    # Get memory client safely
    memory_client = get_memory_client_safe()
    if not memory_client:
        return "Error: Memory system is currently unavailable. Please try again later."

    try:
        db = SessionLocal()
        try:
            # Get or create user and app
            user, app = get_user_and_app(db, user_id=uid, app_id=client_name)

            # Convert string IDs to UUIDs and filter accessible ones
            requested_ids = [uuid.UUID(mid) for mid in memory_ids]
            principal = principal_var.get(None)
            accessible_by_app = _get_accessible_memories(db, principal, user, app, include_archived=True)

            # Filter by write permissions (group-editable policy)
            if principal:
                from app.security.access import can_write_to_access_entity

                accessible_memory_ids = []
                for memory in accessible_by_app:
                    access_entity = memory.metadata_.get("access_entity") if memory.metadata_ else None
                    if access_entity:
                        if can_write_to_access_entity(principal, access_entity):
                            accessible_memory_ids.append(memory.id)
                    else:
                        # Legacy memory without access_entity - only owner can delete
                        accessible_memory_ids.append(memory.id)
            else:
                accessible_memory_ids = [memory.id for memory in accessible_by_app]

            # Only delete memories that are both requested and accessible
            ids_to_delete = [mid for mid in requested_ids if mid in accessible_memory_ids]

            if not ids_to_delete:
                return "Error: No accessible memories found with provided IDs"

            # Delete from vector store
            for memory_id in ids_to_delete:
                try:
                    memory_client.delete(str(memory_id))
                except Exception as delete_error:
                    logging.warning(f"Failed to delete memory {memory_id} from vector store: {delete_error}")

            # Update each memory's state and create history entries
            now = datetime.datetime.now(datetime.UTC)
            for memory_id in ids_to_delete:
                memory = db.query(Memory).filter(Memory.id == memory_id).first()
                if memory:
                    # Update memory state
                    memory.state = MemoryState.deleted
                    memory.deleted_at = now

                    # Create history entry
                    history = MemoryStatusHistory(
                        memory_id=memory_id,
                        changed_by=user.id,
                        old_state=MemoryState.active,
                        new_state=MemoryState.deleted
                    )
                    db.add(history)

                    # Create access log entry
                    access_log = MemoryAccessLog(
                        memory_id=memory_id,
                        app_id=app.id,
                        access_type="delete",
                        metadata_={"operation": "delete_by_id"}
                    )
                    db.add(access_log)

            db.commit()

            # Delete from Neo4j graph (non-blocking)
            for memory_id in ids_to_delete:
                try:
                    # Update entity edges before deleting memory
                    update_entity_edges_on_memory_delete(str(memory_id), uid)
                    # Delete similarity edges
                    delete_similarity_edges_for_memory(str(memory_id), uid)
                    # Delete the memory node
                    delete_memory_from_graph(str(memory_id))
                except Exception as graph_error:
                    logging.warning(f"Graph deletion failed for {memory_id}: {graph_error}")

            return f"Successfully deleted {len(ids_to_delete)} memories"
        finally:
            db.close()
    except Exception as e:
        logging.exception(f"Error deleting memories: {e}")
        return f"Error deleting memories: {e}"


@mcp.tool(
    description="""Delete ALL memories accessible to the current user.

Use when: Complete reset of user's memory space. Rarely needed.

WARNING: This action is IRREVERSIBLE and deletes ALL accessible memories.
Consider using search_memory + delete_memories for targeted cleanup instead.

NOTE: Only deletes memories where access_entity matches your grants.
""",
    annotations=ToolAnnotations(readOnlyHint=False, destructiveHint=True, title="Delete All Memories")
)
async def delete_all_memories() -> str:
    # Check scope - requires memories:delete
    scope_error = _check_tool_scope("memories:delete")
    if scope_error:
        return scope_error

    uid = user_id_var.get(None)
    client_name = client_name_var.get(None)
    if not uid:
        return "Error: user_id not provided"
    if not client_name:
        return "Error: client_name not provided"

    # Get memory client safely
    memory_client = get_memory_client_safe()
    if not memory_client:
        return "Error: Memory system is currently unavailable. Please try again later."

    try:
        db = SessionLocal()
        try:
            # Get or create user and app
            user, app = get_user_and_app(db, user_id=uid, app_id=client_name)

            principal = principal_var.get(None)
            accessible_memories = _get_accessible_memories(db, principal, user, app, include_archived=True)

            if principal:
                from app.security.access import can_write_to_access_entity

                accessible_memory_ids = []
                for memory in accessible_memories:
                    access_entity = memory.metadata_.get("access_entity") if memory.metadata_ else None
                    if access_entity:
                        if can_write_to_access_entity(principal, access_entity):
                            accessible_memory_ids.append(memory.id)
                    else:
                        if memory.user_id == user.id:
                            accessible_memory_ids.append(memory.id)
            else:
                accessible_memory_ids = [memory.id for memory in accessible_memories]

            # delete the accessible memories only
            for memory_id in accessible_memory_ids:
                try:
                    memory_client.delete(str(memory_id))
                except Exception as delete_error:
                    logging.warning(f"Failed to delete memory {memory_id} from vector store: {delete_error}")

            # Update each memory's state and create history entries
            now = datetime.datetime.now(datetime.UTC)
            for memory_id in accessible_memory_ids:
                memory = db.query(Memory).filter(Memory.id == memory_id).first()
                # Update memory state
                memory.state = MemoryState.deleted
                memory.deleted_at = now

                # Create history entry
                history = MemoryStatusHistory(
                    memory_id=memory_id,
                    changed_by=user.id,
                    old_state=MemoryState.active,
                    new_state=MemoryState.deleted
                )
                db.add(history)

                # Create access log entry
                access_log = MemoryAccessLog(
                    memory_id=memory_id,
                    app_id=app.id,
                    access_type="delete_all",
                    metadata_={"operation": "bulk_delete"}
                )
                db.add(access_log)

            db.commit()

            # Delete from Neo4j graph (non-blocking)
            for memory_id in accessible_memory_ids:
                try:
                    update_entity_edges_on_memory_delete(str(memory_id), uid)
                    delete_similarity_edges_for_memory(str(memory_id), uid)
                    delete_memory_from_graph(str(memory_id))
                except Exception as graph_error:
                    logging.warning(f"Graph deletion failed for memory {memory_id}: {graph_error}")

            return "Successfully deleted all memories"
        finally:
            db.close()
    except Exception as e:
        logging.exception(f"Error deleting memories: {e}")
        return f"Error deleting memories: {e}"


@mcp.tool(
    description="""Update a memory's content, metadata, or tags.

Use when: Correcting, enriching, or evolving existing memories. Only provided fields are updated.

Parameters:
- memory_id: UUID of memory to update (required if memory_ids not provided)
- memory_ids: List of UUIDs for batch update (required if memory_id not provided)
- text: New content text (single memory only, not allowed with memory_ids)
- category, scope, artifact_type, artifact_ref: Update structure
- entity, evidence: Update metadata
- mcp_client: Override the app/client that wrote the memory (e.g., "gemini", "cursor")
- add_tags: Dict of tags to add (e.g., {"confirmed": true})
- remove_tags: List of tag keys to remove
- async_mode: When true, return immediately with job_id and process in background (default: false)

Returns: Updated memory with all fields (single), or batch result summary (batch). In async mode, returns job_id.

Example:
- update_memory(memory_id="abc-123", add_tags={"verified": true}, evidence=["PR-456"])
- update_memory(memory_ids=["abc-123", "def-456"], mcp_client="gemini")
""",
    annotations=ToolAnnotations(readOnlyHint=False, title="Update Memory")
)
async def update_memory(
    memory_id: str = None,
    memory_ids: list = None,
    # Content
    text: str = None,
    # Structure
    category: str = None,
    scope: str = None,
    artifact_type: str = None,
    artifact_ref: str = None,
    access_entity: str = None,
    # Metadata
    entity: str = None,
    source: str = None,
    evidence: list = None,
    # App/client override
    mcp_client: str = None,
    # Tag operations
    add_tags: dict = None,
    remove_tags: list = None,
    # Maintenance mode
    preserve_timestamps: bool = False,
    async_mode: bool = False,
) -> str:
    # Check scope - requires memories:write
    scope_error = _check_tool_scope("memories:write")
    if scope_error:
        return scope_error

    uid = user_id_var.get(None)
    client_name = client_name_var.get(None)

    if not uid:
        return json.dumps({"error": "user_id not provided"})
    if not client_name:
        return json.dumps({"error": "client_name not provided"})

    # Validate memory_id vs memory_ids - exactly one must be provided
    if memory_id and memory_ids:
        return json.dumps({"error": "Provide either memory_id or memory_ids, not both"})
    if not memory_id and not memory_ids:
        return json.dumps({"error": "Either memory_id or memory_ids is required"})

    # Batch mode validation
    is_batch = memory_ids is not None
    if is_batch:
        if text is not None:
            return json.dumps({"error": "text updates are not allowed in batch mode (memory_ids)"})
        if not isinstance(memory_ids, list) or len(memory_ids) == 0:
            return json.dumps({"error": "memory_ids must be a non-empty list"})
        # Normalize to list of strings
        ids_to_update = [str(mid) for mid in memory_ids]
    else:
        ids_to_update = [str(memory_id)]

    # Validate update fields
    try:
        validated_fields = validate_update_fields(
            category=category,
            scope=scope,
            artifact_type=artifact_type,
            artifact_ref=artifact_ref,
            entity=entity,
            access_entity=access_entity,
            source=source,
            evidence=evidence,
        )
    except StructuredMemoryError as e:
        return json.dumps(_format_structured_memory_error(e))

    content_text = None
    if text is not None:
        try:
            content_text = validate_text(text)
        except StructuredMemoryError as e:
            return json.dumps(_format_structured_memory_error(e))

    try:
        normalized_add_tags = (
            normalize_tags_input(add_tags) if add_tags is not None else None
        )
        normalized_remove_tags = (
            normalize_tag_list_input(remove_tags) if remove_tags is not None else None
        )
    except StructuredMemoryError as e:
        return json.dumps(_format_structured_memory_error(e))

    principal = principal_var.get(None)

    if async_mode:
        summary = _summarize_update_job(
            memory_ids=ids_to_update,
            validated_fields=validated_fields,
            content_text=content_text,
            normalized_add_tags=normalized_add_tags,
            normalized_remove_tags=normalized_remove_tags,
            mcp_client=mcp_client,
            is_batch=is_batch,
        )
        job_id = create_memory_job(
            requested_by=uid,
            summary=summary,
            job_type="memory_update",
        )
        asyncio.create_task(
            _run_update_memory_job(
                job_id=job_id,
                user_id=uid,
                client_name=client_name,
                ids_to_update=ids_to_update,
                validated_fields=validated_fields,
                content_text=content_text,
                normalized_add_tags=normalized_add_tags,
                normalized_remove_tags=normalized_remove_tags,
                preserve_timestamps=preserve_timestamps,
                mcp_client=mcp_client,
                is_batch=is_batch,
                principal=principal,
            )
        )
        return json.dumps({
            "status": "queued",
            "job_id": job_id,
        })

    response = _update_memory_core(
        user_id=uid,
        client_name=client_name,
        ids_to_update=ids_to_update,
        validated_fields=validated_fields,
        content_text=content_text,
        normalized_add_tags=normalized_add_tags,
        normalized_remove_tags=normalized_remove_tags,
        preserve_timestamps=preserve_timestamps,
        mcp_client=mcp_client,
        is_batch=is_batch,
        principal=principal,
    )
    return json.dumps(response)


@mcp.tool(
    description="""Find semantically similar memories (embedding-based).

Use when: Finding memories with similar meaning/content.

NOT for: Metadata connections - use graph_related_memories instead.

Parameters:
- memory_id: UUID of seed memory (required)
- min_score: Minimum similarity (0.0-1.0, default: 0.0)
- limit: Max results (default: 10)

Returns: similar[] with content, similarity_score, rank.
""",
    annotations=ToolAnnotations(readOnlyHint=True, title="Similar Memories (Semantic)")
)
async def graph_similar_memories(
    memory_id: str,
    min_score: float = 0.0,
    limit: int = 10,
) -> str:
    uid = user_id_var.get(None)
    client_name = client_name_var.get(None)

    if not uid:
        return json.dumps({"error": "user_id not provided"})
    if not client_name:
        return json.dumps({"error": "client_name not provided"})

    if not is_similarity_enabled():
        return json.dumps({
            "seed_memory_id": memory_id,
            "similar": [],
            "count": 0,
            "similarity_enabled": False,
        })

    limit = min(max(1, int(limit or 10)), 50)
    min_score = max(0.0, min(1.0, float(min_score or 0.0)))

    try:
        db = SessionLocal()
        try:
            user, app = get_user_and_app(db, user_id=uid, app_id=client_name)

            principal = principal_var.get(None)
            accessible_memories = _get_accessible_memories(db, principal, user, app, include_archived=True)
            allowed = [str(memory.id) for memory in accessible_memories]

            if memory_id not in allowed:
                return json.dumps({"error": f"Memory '{memory_id}' not found or not accessible"})

            similar = get_similar_memories_from_graph(
                memory_id=memory_id,
                user_id=uid,
                allowed_memory_ids=allowed,
                min_score=min_score,
                limit=limit,
            )

            # Access log (best-effort)
            try:
                db.add(
                    MemoryAccessLog(
                        memory_id=uuid.UUID(memory_id),
                        app_id=app.id,
                        access_type="graph_similar",
                        metadata_={"min_score": min_score, "limit": limit},
                    )
                )
                db.commit()
            except Exception as log_error:
                logging.warning(f"Failed to log graph_similar access: {log_error}")

            return json.dumps({
                "seed_memory_id": memory_id,
                "similar": similar,
                "count": len(similar),
                "similarity_enabled": True,
            }, default=str)
        finally:
            db.close()
    except Exception as e:
        logging.exception(f"Error in graph_similar_memories: {e}")
        return json.dumps({"error": f"Error finding similar memories: {str(e)}"})


@mcp.tool(
    description="""Get entities frequently co-mentioned with a given entity.

Use when: Understanding relationship networks between people, services, or concepts.

Parameters:
- entity_name: Entity to explore (required)
- min_count: Minimum co-mention count (default: 1)
- limit: Max connections (default: 20)

Returns: connections[] with name, count, sample_memory_ids.
""",
    annotations=ToolAnnotations(readOnlyHint=True, title="Entity Network")
)
async def graph_entity_network(
    entity_name: str,
    min_count: int = 1,
    limit: int = 20,
) -> str:
    uid = user_id_var.get(None)
    client_name = client_name_var.get(None)

    if not uid:
        return json.dumps({"error": "user_id not provided"})
    if not client_name:
        return json.dumps({"error": "client_name not provided"})

    if not is_graph_enabled():
        return json.dumps({
            "entity": entity_name,
            "connections": [],
            "total": 0,
            "graph_enabled": False,
        })

    limit = min(max(1, int(limit or 20)), 100)
    min_count = max(1, int(min_count or 1))

    try:
        principal = principal_var.get(None)
        access_entities, access_entity_prefixes = _build_graph_access_filters(principal, uid)

        result = get_entity_network_from_graph(
            entity_name=entity_name,
            user_id=uid,
            min_count=min_count,
            limit=limit,
            access_entities=access_entities,
            access_entity_prefixes=access_entity_prefixes,
        )

        if result is None:
            return json.dumps({
                "entity": entity_name,
                "connections": [],
                "total": 0,
                "graph_enabled": True,
                "message": f"Entity '{entity_name}' not found",
            })

        return json.dumps({
            "entity": entity_name,
            "connections": result.get("connections", []),
            "total": result.get("total", 0),
            "graph_enabled": True,
        }, default=str)

    except Exception as e:
        logging.exception(f"Error in graph_entity_network: {e}")
        return json.dumps({"error": f"Error getting entity network: {str(e)}"})


@mcp.tool(description="""Get co-occurring tags via OM_COOCCURS edges with PMI scores.

Shows tags that frequently appear together with this tag across memories.
PMI (Pointwise Mutual Information) indicates how strongly tags are associated.

Parameters:
- tag_key: The tag key to find related tags for (required)
- min_count: Minimum co-occurrence count (default: 1)
- limit: Max tags to return (default: 20, max: 50)

Returns:
- tag: the queried tag key
- related[]: each has {key, count, pmi}
- count: number of related tags
- graph_enabled: whether feature is active
""")
async def graph_related_tags(
    tag_key: str,
    min_count: int = 1,
    limit: int = 20,
) -> str:
    uid = user_id_var.get(None)
    client_name = client_name_var.get(None)

    if not uid:
        return json.dumps({"error": "user_id not provided"})
    if not client_name:
        return json.dumps({"error": "client_name not provided"})

    if not is_graph_enabled():
        return json.dumps({
            "tag": tag_key,
            "related": [],
            "count": 0,
            "graph_enabled": False,
        })

    limit = min(max(1, int(limit or 20)), 50)
    min_count = max(1, int(min_count or 1))

    try:
        principal = principal_var.get(None)
        access_entities, access_entity_prefixes = _build_graph_access_filters(principal, uid)

        related = get_related_tags_from_graph(
            tag_key=tag_key,
            user_id=uid,
            min_count=min_count,
            limit=limit,
            access_entities=access_entities,
            access_entity_prefixes=access_entity_prefixes,
        )

        return json.dumps({
            "tag": tag_key,
            "related": related,
            "count": len(related),
            "graph_enabled": True,
        }, default=str)

    except Exception as e:
        logging.exception(f"Error in graph_related_tags: {e}")
        return json.dumps({"error": f"Error getting related tags: {str(e)}"})


@mcp.tool(description="""Find and optionally merge duplicate entities.

Identifies entity variants that should be merged (e.g., "Matthias", "matthias", "matthias_coers").

Parameters:
- dry_run: If true (default), only show what would be merged without changes
- auto: If true, automatically merge all detected duplicates
- canonical: For manual merge: the target entity name
- variants: For manual merge: comma-separated list of variant names to merge
- access_entity: Access control scope to target (defaults to `user:<user_id>` if omitted)

Returns:
- duplicates: List of detected duplicate groups (if dry_run or no merge specified)
- merged: Count of merged entities (if merge was performed)

Examples:
- Detect duplicates: graph_normalize_entities(dry_run=true)
- Auto-merge all: graph_normalize_entities(auto=true, dry_run=false)
- Manual merge: graph_normalize_entities(canonical="matthias_coers", variants="Matthias,matthias", dry_run=false)
""")
async def graph_normalize_entities(
    dry_run: bool = True,
    auto: bool = False,
    canonical: str = None,
    variants: str = None,
    access_entity: str = None,
) -> str:
    uid = user_id_var.get(None)
    client_name = client_name_var.get(None)

    if not uid:
        return json.dumps({"error": "user_id not provided"})
    if not client_name:
        return json.dumps({"error": "client_name not provided"})

    if not is_graph_enabled():
        return json.dumps({"duplicates": [], "graph_enabled": False})

    try:
        # If just checking duplicates
        if dry_run and not auto and not canonical:
            duplicates = find_duplicate_entities_in_graph(uid, access_entity=access_entity)
            return json.dumps({
                "duplicates": duplicates,
                "count": len(duplicates),
                "dry_run": True,
                "graph_enabled": True,
            }, default=str)

        # Perform normalization
        variant_list = [v.strip() for v in variants.split(",")] if variants else None

        result = normalize_entities_in_graph(
            user_id=uid,
            canonical_name=canonical,
            variant_names=variant_list,
            auto=auto,
            dry_run=dry_run,
            access_entity=access_entity,
        )

        result["graph_enabled"] = True
        return json.dumps(result, default=str)

    except Exception as e:
        logging.exception(f"Error in graph_normalize_entities: {e}")
        return json.dumps({"error": f"Error normalizing entities: {str(e)}"})


@mcp.tool(description="""Semantic entity normalization with multi-phase duplicate detection.

Extends basic case normalization with advanced similarity detection:
1. String similarity (Levenshtein/fuzzy matching, e.g., "el_juego" ≈ "eljuego")
2. Prefix/suffix matching (e.g., "marie" → "marie_schubenz")
3. Domain normalization (e.g., "eljuego.community" → "el_juego")

Parameters:
- mode: Detection mode - "detect" (find duplicates), "preview" (dry-run merge), "execute" (perform merge)
- threshold: Minimum confidence for match (0.0-1.0, default: 0.7). Lower = more matches.
- canonical: For manual merge: target entity name
- variants: For manual merge: comma-separated variant names
- access_entity: Access control scope to target (defaults to `user:<user_id>` if omitted)

Returns (mode=detect):
- duplicates[]: Each group has {canonical, variants, confidence, sources}
- count: Number of duplicate groups

Returns (mode=preview/execute):
- merge_groups: Number of groups to merge
- total_variants_merged: Total variant count
- total_edges_migrated: Edges that were/will be migrated
- merges[]: Details per group

Examples:
- Detect: graph_normalize_entities_semantic(mode="detect")
- Lower threshold: graph_normalize_entities_semantic(mode="detect", threshold=0.6)
- Preview all: graph_normalize_entities_semantic(mode="preview")
- Execute all: graph_normalize_entities_semantic(mode="execute")
- Manual merge: graph_normalize_entities_semantic(mode="execute", canonical="el_juego", variants="eljuego,el-juego")
""")
async def graph_normalize_entities_semantic(
    mode: str = "detect",
    threshold: float = 0.7,
    canonical: str = None,
    variants: str = None,
    access_entity: str = None,
) -> str:
    uid = user_id_var.get(None)
    client_name = client_name_var.get(None)

    if not uid:
        return json.dumps({"error": "user_id not provided"})
    if not client_name:
        return json.dumps({"error": "client_name not provided"})

    if not is_graph_enabled():
        return json.dumps({"duplicates": [], "graph_enabled": False})

    if mode not in ["detect", "preview", "execute"]:
        return json.dumps({"error": f"Invalid mode: {mode}. Use 'detect', 'preview', or 'execute'"})

    try:
        from app.graph.graph_ops import find_semantic_duplicates, normalize_entities_semantic

        if mode == "detect":
            # Only detect duplicates
            duplicates = await find_semantic_duplicates(
                uid,
                threshold=threshold,
                access_entity=access_entity,
            )
            return json.dumps({
                "duplicates": duplicates,
                "count": len(duplicates),
                "threshold": threshold,
                "mode": "detect",
                "graph_enabled": True,
            }, default=str)

        # Preview or execute merge
        variant_list = [v.strip() for v in variants.split(",")] if variants else None
        dry_run = mode == "preview"

        if canonical and variant_list:
            # Manual merge
            result = await normalize_entities_semantic(
                user_id=uid,
                canonical=canonical,
                variants=variant_list,
                threshold=threshold,
                dry_run=dry_run,
                access_entity=access_entity,
            )
        else:
            # Auto-detect and merge
            result = await normalize_entities_semantic(
                user_id=uid,
                auto=True,
                threshold=threshold,
                dry_run=dry_run,
                access_entity=access_entity,
            )

        result["mode"] = mode
        result["graph_enabled"] = True
        return json.dumps(result, default=str)

    except Exception as e:
        logging.exception(f"Error in graph_normalize_entities_semantic: {e}")
        return json.dumps({"error": f"Error in semantic normalization: {str(e)}"})


@mcp.tool(description="""Get typed relationships for an entity.

Shows semantic relationships extracted from memories, not just co-mentions.
Examples: "Grischa -[schwester_von]-> Julia", "Grischa -[arbeitet_bei]-> CloudFactory"

Parameters:
- entity_name: Name of the entity (required)
- relation_types: Optional comma-separated relation types to filter
                  (e.g., "schwester_von,bruder_von")
- category: Optional category to filter (family, social, work, location, creative, membership)
- direction: "outgoing", "incoming", or "both" (default: both)
- limit: Max relations to return (default: 50)

Returns:
- entity: the queried entity
- relations[]: each has {target, type, direction, memory_id, count}
- count: number of relations
""")
async def graph_entity_relations(
    entity_name: str,
    relation_types: str = None,
    category: str = None,
    direction: str = "both",
    limit: int = 50,
) -> str:
    uid = user_id_var.get(None)
    client_name = client_name_var.get(None)

    if not uid:
        return json.dumps({"error": "user_id not provided"})
    if not client_name:
        return json.dumps({"error": "client_name not provided"})

    if not is_graph_enabled():
        return json.dumps({
            "entity": entity_name,
            "relations": [],
            "count": 0,
            "graph_enabled": False,
        })

    limit = min(max(1, int(limit or 50)), 200)

    # Build type filter
    type_filter = []
    if relation_types:
        type_filter = [t.strip() for t in relation_types.split(",")]

    try:
        principal = principal_var.get(None)
        access_entities, access_entity_prefixes = _build_graph_access_filters(principal, uid)

        relations = get_entity_relations_from_graph(
            entity_name=entity_name,
            user_id=uid,
            relation_types=type_filter if type_filter else None,
            category=category,
            direction=direction,
            limit=limit,
            access_entities=access_entities,
            access_entity_prefixes=access_entity_prefixes,
        )

        return json.dumps({
            "entity": entity_name,
            "relations": relations,
            "count": len(relations),
            "graph_enabled": True,
        }, default=str)

    except Exception as e:
        logging.exception(f"Error in graph_entity_relations: {e}")
        return json.dumps({"error": f"Error getting entity relations: {str(e)}"})


@mcp.tool(description="""Get biographical timeline for a person or user.

Shows chronological events like residences, projects, work history.

Parameters:
- entity_name: Optional person name (if omitted, shows all events for user)
- event_types: Optional comma-separated types (residence, education, work, project, relationship, health, travel, milestone)
- start_year: Optional start year filter
- end_year: Optional end year filter
- limit: Max events (default: 50)

Returns:
- events[]: chronologically sorted events with dates and descriptions
- count: number of events

Example:
- Get Grischa's timeline: graph_biography_timeline(entity_name="grischa")
- Get all projects: graph_biography_timeline(event_types="project")
""")
async def graph_biography_timeline(
    entity_name: str = None,
    event_types: str = None,
    start_year: int = None,
    end_year: int = None,
    limit: int = 50,
) -> str:
    uid = user_id_var.get(None)
    client_name = client_name_var.get(None)

    if not uid:
        return json.dumps({"error": "user_id not provided"})
    if not client_name:
        return json.dumps({"error": "client_name not provided"})

    if not is_graph_enabled():
        return json.dumps({
            "entity": entity_name,
            "events": [],
            "count": 0,
            "graph_enabled": False,
        })

    try:
        type_list = [t.strip() for t in event_types.split(",")] if event_types else None

        principal = principal_var.get(None)
        access_entities, access_entity_prefixes = _build_graph_access_filters(principal, uid)

        events = get_biography_timeline_from_graph(
            user_id=uid,
            entity_name=entity_name,
            event_types=type_list,
            start_year=start_year,
            end_year=end_year,
            limit=min(max(1, limit or 50), 200),
            access_entities=access_entities,
            access_entity_prefixes=access_entity_prefixes,
        )

        return json.dumps({
            "entity": entity_name,
            "events": events,
            "count": len(events),
            "graph_enabled": True,
        }, default=str)

    except Exception as e:
        logging.exception(f"Error in graph_biography_timeline: {e}")
        return json.dumps({"error": f"Error getting timeline: {str(e)}"})


# =============================================================================
# CODE INTELLIGENCE TOOLS
# =============================================================================


def get_code_toolkit():
    """Lazy-load code toolkit to avoid import-time failures."""
    from app.code_toolkit import get_code_toolkit as _get_toolkit
    return _get_toolkit()


def _create_code_meta(request_id: str = None, degraded: bool = False, missing: list = None, error: str = None) -> dict:
    """Create metadata dict for code tool responses."""
    return {
        "request_id": request_id or str(uuid.uuid4()),
        "degraded_mode": degraded,
        "missing_sources": missing or [],
        "error": error,
    }


def _create_graph_meta(
    request_id: str | None = None,
    degraded: bool = False,
    missing: list | None = None,
    error: str | None = None,
) -> dict:
    """Create metadata dict for graph query tool responses."""
    return {
        "request_id": request_id or str(uuid.uuid4()),
        "degraded_mode": degraded,
        "missing_sources": missing or [],
        "error": error,
    }


@mcp.tool(description="""Run a read-only Neo4j query against graph data.

Scopes:
- code: CODE_* graph (requires repo_id filter)
- memory: OM_* graph (requires accessEntity filter)
- mem0: __Entity__ graph (requires user_id filter)

Required parameters:
- scope: code | memory | mem0
- query: Read-only Cypher (single-statement; must start with MATCH/OPTIONAL MATCH/WITH and include RETURN)

Optional parameters:
- params: Dict of Cypher parameters
- repo_id: Required when scope=code
- access_entity: Required when scope=memory
- user_id: Optional when scope=mem0 (defaults to current user)
- limit: Max rows (default: 50, max: 200)

Rules enforced:
- Single-statement only (no semicolons)
- Must include explicit labels (CODE_*, OM_*, or __Entity__ depending on scope)
- Must include required filter (repo_id/accessEntity/user_id) using parameters
- If LIMIT is used as a parameter, it must be $limit (max 200 enforced)
- Only read-only Cypher allowed (no CREATE/MERGE/SET/DELETE/DETACH/REMOVE/DROP/CALL/LOAD/FOREACH/APOC/DBMS/SHOW/PROFILE/EXPLAIN/COMMIT/ROLLBACK/TRANSACTION/CONSTRAINT/INDEX)

Returns:
- rows[]: List of result records with nodes/relationships serialized
- meta: Response metadata
""")
async def graph_query(
    scope: str,
    query: str,
    params: dict | None = None,
    repo_id: str | None = None,
    access_entity: str | None = None,
    user_id: str | None = None,
    limit: int = 50,
) -> str:
    """Run a read-only graph query with scope-aware safeguards."""
    scope_error = _check_tool_scope("graph:read")
    if scope_error:
        return scope_error

    principal = principal_var.get(None)
    if not principal:
        return json.dumps({
            "error": "Authentication required",
            "code": "MISSING_AUTH",
        })

    try:
        from tools.graph_query import GraphQueryInput, GraphQueryError, prepare_graph_query, serialize_record
    except Exception as exc:
        return json.dumps({
            "rows": [],
            "meta": _create_graph_meta(
                degraded=True,
                missing=["graph_query"],
                error=str(exc),
            ),
        })

    try:
        prepared = prepare_graph_query(
            GraphQueryInput(
                scope=scope,
                query=query,
                params=params or {},
                repo_id=repo_id,
                access_entity=access_entity,
                user_id=user_id,
                limit=limit,
            ),
            principal,
        )
    except GraphQueryError as exc:
        return json.dumps({
            "rows": [],
            "error": str(exc),
            "code": exc.code,
            "meta": _create_graph_meta(error=str(exc)),
        })

    try:
        from app.graph.neo4j_client import is_neo4j_configured, is_neo4j_healthy, get_neo4j_session

        if not is_neo4j_configured() or not is_neo4j_healthy():
            return json.dumps({
                "rows": [],
                "meta": _create_graph_meta(
                    degraded=True,
                    missing=["neo4j"],
                    error="Graph backend unavailable",
                ),
            })

        rows = []
        with get_neo4j_session() as session:
            result = session.run(prepared.query, prepared.params)
            for idx, record in enumerate(result):
                if idx >= prepared.limit:
                    break
                rows.append(serialize_record(record))

        return json.dumps({
            "rows": rows,
            "meta": _create_graph_meta(),
        }, default=str)
    except Exception as exc:
        logging.exception(f"Error in graph_query: {exc}")
        return json.dumps({
            "rows": [],
            "meta": _create_graph_meta(
                degraded=True,
                error=str(exc),
            ),
        })


@mcp.tool(description="""Index a local codebase into the CODE_* graph and search index.

Required parameters:
- repo_id: Repository ID (required)
- root_path: Filesystem path to the repository root (required)

Optional parameters:
- index_name: OpenSearch index name (default: "code")
- reset: Clear existing repo data before indexing (default: false)
- max_files: Maximum number of files to index
- include_api_boundaries: Enable API boundary detection (default: true)
- enable_zod_schema_aliases: Add SCHEMA_ALIASES edges between local and canonical Zod schemas (default: true)
- ignore_patterns: List of substrings to skip (e.g., ["dist", "build"])
- allow_patterns: List of substrings to always include, even if ignored
- async_mode: Run indexing in the background and return a job_id (default: false)

Returns:
- repo_id: Repository ID
- files_indexed: Files successfully indexed
- files_failed: Files that failed indexing
- symbols_indexed: Symbols added to the graph
- documents_indexed: Documents added to OpenSearch
- call_edges_indexed: Inferred traversal edges (CALLS + GraphQL READS; historical name)
- duration_ms: Total indexing time
- meta: Response metadata

Notes:
- Also ingests OpenAPI JSON/YAML specs (openapi*.json/yml/yaml) into CODE_OPENAPI_DEF and CODE_SCHEMA_FIELD nodes.
- Parses GraphQL documents (*.graphql/*.gql) to add READS edges to schema fields.
- Extracts string path literals into CODE_FIELD_PATH nodes and CONTAINS edges.
""")
async def index_codebase(
    repo_id: str,
    root_path: str,
    index_name: str = "code",
    reset: bool = False,
    max_files: int = None,
    include_api_boundaries: bool = True,
    enable_zod_schema_aliases: bool = True,
    ignore_patterns: list[str] | None = None,
    allow_patterns: list[str] | None = None,
    async_mode: bool = False,
) -> str:
    """Index a local repository for code intelligence."""
    scope_error = _check_tool_scope("graph:write")
    if scope_error:
        return scope_error

    toolkit = get_code_toolkit()

    root = resolve_repo_root(root_path, repo_id=repo_id)
    if not root.exists() or not root.is_dir():
        return json.dumps({
            "error": "root_path not found",
            "hint": "If running in Docker, use the container path (e.g., /usr/src/<repo>) or set OPENMEMORY_REPO_ROOT.",
        })

    if not toolkit.is_available("neo4j"):
        return json.dumps({
            "repo_id": repo_id,
            "files_indexed": 0,
            "files_failed": 0,
            "symbols_indexed": 0,
            "documents_indexed": 0,
            "call_edges_indexed": 0,
            "duration_ms": 0.0,
            "meta": _create_code_meta(
                degraded=True,
                missing=["neo4j"],
                error="Graph backend unavailable",
            ),
        })

    missing_sources = []
    if not toolkit.is_available("opensearch"):
        missing_sources.append("opensearch")
    if not toolkit.is_available("embedding"):
        missing_sources.append("embedding")

    try:
        from indexing.code_indexer import CodeIndexingService

        meta = _create_code_meta(
            degraded=bool(missing_sources),
            missing=missing_sources,
            error="Some backends unavailable" if missing_sources else None,
        )

        indexer = CodeIndexingService(
            root_path=root,
            repo_id=repo_id,
            graph_driver=toolkit.neo4j_driver,
            opensearch_client=toolkit.opensearch_client if toolkit.is_available("opensearch") else None,
            embedding_service=toolkit.embedding_service if toolkit.is_available("embedding") else None,
            index_name=index_name or "code",
            include_api_boundaries=include_api_boundaries,
            enable_zod_schema_aliases=enable_zod_schema_aliases,
            ignore_patterns=ignore_patterns,
            allow_patterns=allow_patterns,
        )

        if async_mode:
            # Use persistent job queue - worker will pick up the job
            from app.database import SessionLocal
            from app.services.job_queue_service import IndexingJobQueueService, QueueFullError

            uid = user_id_var.get("anonymous")
            db = SessionLocal()
            try:
                import os
                max_queued_jobs = int(os.getenv("MAX_QUEUED_JOBS", "100"))
                valkey_client = None
                try:
                    import redis

                    host = os.getenv("VALKEY_HOST", "valkey")
                    port = int(os.getenv("VALKEY_PORT", "6379"))
                    client = redis.Redis(host=host, port=port, socket_timeout=5)
                    client.ping()
                    valkey_client = client
                except Exception:
                    valkey_client = None

                queue_service = IndexingJobQueueService(
                    db=db,
                    valkey_client=valkey_client,
                    max_queued_jobs=max_queued_jobs,
                )
                try:
                    job_uuid = queue_service.create_job(
                        repo_id=repo_id,
                        root_path=str(root),
                        index_name=index_name or "code",
                        requested_by=uid,
                        request={
                            "max_files": max_files,
                            "reset": reset,
                            "include_api_boundaries": include_api_boundaries,
                            "enable_zod_schema_aliases": enable_zod_schema_aliases,
                            "ignore_patterns": ignore_patterns,
                            "allow_patterns": allow_patterns,
                        },
                        meta=meta,
                    )
                    return json.dumps({
                        "repo_id": repo_id,
                        "job_id": str(job_uuid),
                        "status": "queued",
                        "meta": meta,
                    })
                except QueueFullError as e:
                    return json.dumps({
                        "error": str(e),
                        "meta": _create_code_meta(
                            degraded=True,
                            missing=missing_sources,
                            error=str(e),
                        ),
                    })
            finally:
                db.close()

        summary = indexer.index_repository(
            max_files=max_files,
            reset=reset,
        )

        return json.dumps({
            "repo_id": summary.repo_id,
            "files_indexed": summary.files_indexed,
            "files_failed": summary.files_failed,
            "symbols_indexed": summary.symbols_indexed,
            "documents_indexed": summary.documents_indexed,
            "call_edges_indexed": summary.call_edges_indexed,
            "duration_ms": summary.duration_ms,
            "meta": meta,
        }, default=str)

    except Exception as e:
        logging.exception(f"Error in index_codebase: {e}")
        return json.dumps({
            "repo_id": repo_id,
            "files_indexed": 0,
            "files_failed": 0,
            "symbols_indexed": 0,
            "documents_indexed": 0,
            "call_edges_indexed": 0,
            "duration_ms": 0.0,
            "meta": _create_code_meta(
                degraded=True,
                missing=missing_sources,
                error=str(e),
            ),
        })


@mcp.tool(description="""Get the status of a background code indexing job.

Required parameters:
- job_id: Indexing job ID (required)

Returns:
- job_id
- repo_id
- status: queued | running | succeeded | failed
- timestamps and summary/error when available
""")
async def index_codebase_status(job_id: str) -> str:
    """Fetch status for a background code indexing job."""
    scope_error = _check_tool_scope("graph:read")
    if scope_error:
        return scope_error

    import uuid as uuid_mod

    # Try persistent store first
    try:
        job_uuid = uuid_mod.UUID(job_id)
        from app.database import SessionLocal
        from app.services.job_queue_service import IndexingJobQueueService

        db = SessionLocal()
        try:
            queue_service = IndexingJobQueueService(db=db, valkey_client=None)
            job = queue_service.get_job(job_uuid)
            if job:
                if not _can_access_job(job, "admin:read"):
                    return json.dumps({"error": "Access denied"})
                return json.dumps(job, default=str)
        finally:
            db.close()
    except ValueError:
        pass  # Invalid UUID, try legacy store
    except Exception:
        pass  # Persistent store error, try legacy store

    # Fallback to legacy in-memory store
    job = get_index_job(job_id)
    if not job:
        return json.dumps({"error": "job_id not found"})

    return json.dumps(job, default=str)


@mcp.tool(description="""Cancel a running or queued indexing job.

Required parameters:
- job_id: Indexing job ID (required)

Returns:
- job_id
- status: "cancel_requested" on success
- error: Error message if job not found

Examples:
- index_codebase_cancel(job_id="<job-id>")
""")
async def index_codebase_cancel(job_id: str) -> str:
    """Request cancellation of an indexing job."""
    scope_error = _check_tool_scope("graph:write")
    if scope_error:
        return scope_error

    import uuid as uuid_mod
    try:
        job_uuid = uuid_mod.UUID(job_id)
    except ValueError:
        return json.dumps({"error": "Invalid job_id format"})

    # Try persistent store
    from app.database import SessionLocal
    from app.services.job_queue_service import IndexingJobQueueService

    try:
        db = SessionLocal()
        try:
            queue_service = IndexingJobQueueService(db=db, valkey_client=None)
            job = queue_service.get_job(job_uuid)
            if not job:
                return json.dumps({"error": "job_id not found"})

            if not _can_access_job(job, "admin:write"):
                return json.dumps({"error": "Access denied"})

            success = queue_service.cancel_job(job_uuid)
            if success:
                return json.dumps({"job_id": job_id, "status": "cancel_requested"})
            else:
                return json.dumps({"error": "Failed to request cancellation"})
        finally:
            db.close()
    except Exception as e:
        return json.dumps({"error": str(e)})


@mcp.tool(
    description="""Searches code definitions and logic via semantic/hybrid retrieval.

Use when: Discovering implementation details, finding code by concept, function name, or pattern. Good for understanding "How" something is implemented.

NOT for: Finding ALL references of a specific string exhaustively - prefer Grep/search_file_content for strict literal matching. Also not for reading full file context - use Read tool after finding the file.

Parameters:
- query: Search text (required)
- repo_id: Filter by repository
- language: Filter by language (python, typescript, etc.)
- limit: Max results (default: 10, max: 100)
- include_snippet: Include code snippets in results (default: true)
- snippet_max_chars: Max characters for snippets (default: 400, set null for full)
- include_generated: Include generated results without source preference (default: false)

Returns: results[] with symbol info, file paths, line numbers, code snippets, and
generated-source metadata (is_generated, source_tier).

IMPORTANT: Results are snippets only. Use Read tool to see full file context before answering.
Note: Snippets may include class fields/properties, but those are not indexed as symbols. Use the
returned class/file symbol_id for downstream tools.
Note: Snippets are truncated by default to keep results compact; set include_snippet=false or
snippet_max_chars=null to disable truncation.
Note: Results prefer source files by default. Generated/compiled/vendor files (dist/, build/, .d.ts,
codegen output, node_modules) are downranked when source matches exist. Set include_generated to
include them without preference.

Example:
- search_code_hybrid(query="authentication middleware", language="python")
""",
    annotations=ToolAnnotations(readOnlyHint=True, title="Search Code")
)
async def search_code_hybrid(
    query: str,
    repo_id: str = None,
    language: str = None,
    limit: int = 10,
    include_snippet: Optional[bool] = None,
    snippet_max_chars: Optional[int] = None,
    include_generated: Optional[bool] = None,
) -> str:
    """Search code using tri-hybrid retrieval."""
    # Check scope
    scope_error = _check_tool_scope("search:read")
    if scope_error:
        return scope_error

    toolkit = get_code_toolkit()

    # Check if search backend is available
    if not toolkit.is_available("opensearch") and not toolkit.is_available("trihybrid"):
        return json.dumps({
            "results": [],
            "meta": _create_code_meta(
                degraded=True,
                missing=["opensearch"],
                error="Search backend unavailable",
            ),
        })

    if not toolkit.search_tool:
        return json.dumps({
            "results": [],
            "meta": _create_code_meta(
                degraded=True,
                missing=toolkit.get_missing_sources(),
                error=toolkit.get_error("search_tool"),
            ),
        })

    try:
        from tools.search_code_hybrid import SearchCodeHybridInput
        import dataclasses

        input_data = SearchCodeHybridInput(
            query=query,
            repo_id=repo_id,
            language=language,
            limit=min(max(1, limit or 10), 100),
            include_snippet=include_snippet,
            snippet_max_chars=snippet_max_chars,
            include_generated=include_generated,
        )

        result = toolkit.search_tool.search(input_data)
        return json.dumps(dataclasses.asdict(result), default=str)

    except Exception as e:
        logging.exception(f"Error in search_code_hybrid: {e}")
        return json.dumps({
            "results": [],
            "meta": _create_code_meta(
                degraded=True,
                missing=toolkit.get_missing_sources(),
                error=str(e),
            ),
        })


@mcp.tool(
    description="""Explain a code symbol with full call graph context.

Use when: Understanding a function's role - who calls it, what it calls, and its signature.

Parameters:
- symbol_id: SCIP symbol ID (required) - get from search_code_hybrid results
- depth: Analysis depth (default: 2, max: 5)
- include_callers: Show incoming calls (default: true)
- include_callees: Show outgoing calls (default: true)

Returns: Symbol explanation with call graph context.

FALLBACK: If symbol not found, use search_code_hybrid first, then Read the file directly.
""",
    annotations=ToolAnnotations(readOnlyHint=True, title="Explain Code")
)
async def explain_code(
    symbol_id: str,
    depth: int = 2,
    include_callers: bool = True,
    include_callees: bool = True,
) -> str:
    """Explain a code symbol with context."""
    # Check scopes - needs both search:read and graph:read
    scope_error = _check_tool_scope("search:read")
    if scope_error:
        return scope_error
    scope_error = _check_tool_scope("graph:read")
    if scope_error:
        return scope_error

    toolkit = get_code_toolkit()

    if not toolkit.is_available("neo4j"):
        return json.dumps({
            "explanation": None,
            "meta": _create_code_meta(
                degraded=True,
                missing=["neo4j"],
                error="Graph backend unavailable",
            ),
        })

    if not toolkit.explain_tool:
        return json.dumps({
            "explanation": None,
            "meta": _create_code_meta(
                degraded=True,
                missing=toolkit.get_missing_sources(),
                error=toolkit.get_error("explain_tool"),
            ),
        })

    try:
        from tools.explain_code import ExplainCodeConfig
        import dataclasses

        config = ExplainCodeConfig(
            depth=min(max(1, depth or 2), 5),
            include_callers=include_callers,
            include_callees=include_callees,
        )

        result = toolkit.explain_tool.explain(symbol_id, config)
        return json.dumps({
            "explanation": dataclasses.asdict(result) if result else None,
            "meta": _create_code_meta(),
        }, default=str)

    except Exception as e:
        logging.exception(f"Error in explain_code: {e}")
        return json.dumps({
            "explanation": None,
            "meta": _create_code_meta(
                degraded=True,
                error=str(e),
            ),
        })


@mcp.tool(
    description="""Find functions that call a given symbol.

Use when: Tracing who calls a function/method in the codebase.

Parameters:
- repo_id: Repository ID (required)
- symbol_name: Function/method name to find callers for
- symbol_id: SCIP symbol ID (alternative to symbol_name)
- depth: Traversal depth (default: 2, max: 5)
- include_inferred_edges: Override inferred-edge usage (defaults to server config)

Returns: nodes[] and edges[] representing the call graph.

FALLBACK: If "Symbol not found", automatic cascade tries:
1. Grep(pattern=symbol_name)
2. search_code_hybrid(query=symbol_name)
3. Returns suggestions if all fail

NEVER guess callers - use fallback results or admit uncertainty.
""",
    annotations=ToolAnnotations(readOnlyHint=True, title="Find Callers")
)
async def find_callers(
    repo_id: str,
    symbol_name: str = None,
    symbol_id: str = None,
    depth: int = 2,
    use_fallback: bool = True,
    include_inferred_edges: bool | None = None,
) -> str:
    """Find functions that call a given symbol with automatic fallback."""
    # Check scope
    scope_error = _check_tool_scope("graph:read")
    if scope_error:
        return scope_error

    if not symbol_name and not symbol_id:
        return json.dumps({"error": "Either symbol_name or symbol_id is required"})

    toolkit = get_code_toolkit()

    if not toolkit.is_available("neo4j"):
        return json.dumps({
            "nodes": [],
            "edges": [],
            "meta": _create_code_meta(
                degraded=True,
                missing=["neo4j"],
                error="Graph backend unavailable",
            ),
        })

    if not toolkit.callers_tool:
        return json.dumps({
            "nodes": [],
            "edges": [],
            "meta": _create_code_meta(
                degraded=True,
                missing=toolkit.get_missing_sources(),
                error=toolkit.get_error("callers_tool"),
            ),
        })

    # Import at function level to avoid circular imports
    from tools.call_graph import CallGraphInput, SymbolNotFoundError
    import dataclasses

    try:
        input_data = CallGraphInput(
            repo_id=repo_id,
            symbol_id=symbol_id,
            symbol_name=symbol_name,
            depth=min(max(1, depth or 2), 5),
            include_inferred_edges=include_inferred_edges,
        )

        # Use fallback tool if enabled
        if use_fallback:
            try:
                from tools.fallback_find_callers import (
                    FallbackFindCallersTool,
                    GrepTool,
                )

                # Create fallback tool with available dependencies
                fallback_tool = FallbackFindCallersTool(
                    graph_driver=toolkit.callers_tool.graph_driver,
                    grep_tool=GrepTool(),  # Basic grep tool
                    search_tool=toolkit.search_tool if hasattr(toolkit, 'search_tool') else None,
                )

                result = fallback_tool.find(input_data)

                # Add fallback info to response
                response = dataclasses.asdict(result)
                if fallback_tool.fallback_used:
                    response["_fallback_info"] = {
                        "fallback_used": True,
                        "fallback_stage": fallback_tool.fallback_stage,
                        "stage_timings_ms": fallback_tool.get_stage_timings(),
                        "warning": "Results from fallback strategy - verify accuracy",
                    }

                return json.dumps(response, default=str)

            except ImportError:
                logging.warning("FallbackFindCallersTool not available, using basic tool")
                # Fall through to basic tool

        # Basic tool (no fallback)
        result = toolkit.callers_tool.find(input_data)
        return json.dumps(dataclasses.asdict(result), default=str)

    except SymbolNotFoundError as e:
        # Return structured error with suggestions
        logging.info(f"Symbol not found in find_callers: {e}")
        error_response = e.to_dict()
        error_response["nodes"] = []
        error_response["edges"] = []
        error_response["meta"] = _create_code_meta(
            degraded=True,
            error=str(e),
        )
        error_response["_instructions"] = "NEVER guess the caller. Use Grep fallback or ask user for file path."
        return json.dumps(error_response, default=str)

    except Exception as e:
        logging.exception(f"Error in find_callers: {e}")
        return json.dumps({
            "nodes": [],
            "edges": [],
            "meta": _create_code_meta(degraded=True, error=str(e)),
        })


@mcp.tool(
    description="""Find functions called BY a given symbol (outgoing calls).

Use when: Understanding what a function depends on or calls internally.

Parameters:
- repo_id: Repository ID (required)
- symbol_name: Function/method name
- symbol_id: SCIP symbol ID (alternative)
- depth: Traversal depth (default: 2, max: 5)
- include_inferred_edges: Override inferred-edge usage (defaults to server config)

Returns: nodes[] and edges[] representing outgoing call graph.

FALLBACK: If symbol not found, use search_code_hybrid to locate it first.
""",
    annotations=ToolAnnotations(readOnlyHint=True, title="Find Callees")
)
async def find_callees(
    repo_id: str,
    symbol_name: str = None,
    symbol_id: str = None,
    depth: int = 2,
    include_inferred_edges: bool | None = None,
) -> str:
    """Find functions called by a given symbol."""
    # Check scope
    scope_error = _check_tool_scope("graph:read")
    if scope_error:
        return scope_error

    if not symbol_name and not symbol_id:
        return json.dumps({"error": "Either symbol_name or symbol_id is required"})

    toolkit = get_code_toolkit()

    if not toolkit.is_available("neo4j"):
        return json.dumps({
            "nodes": [],
            "edges": [],
            "meta": _create_code_meta(
                degraded=True,
                missing=["neo4j"],
                error="Graph backend unavailable",
            ),
        })

    if not toolkit.callees_tool:
        return json.dumps({
            "nodes": [],
            "edges": [],
            "meta": _create_code_meta(
                degraded=True,
                missing=toolkit.get_missing_sources(),
                error=toolkit.get_error("callees_tool"),
            ),
        })

    try:
        from tools.call_graph import CallGraphInput
        import dataclasses

        input_data = CallGraphInput(
            repo_id=repo_id,
            symbol_id=symbol_id,
            symbol_name=symbol_name,
            depth=min(max(1, depth or 2), 5),
            include_inferred_edges=include_inferred_edges,
        )

        result = toolkit.callees_tool.find(input_data)
        return json.dumps(dataclasses.asdict(result), default=str)

    except Exception as e:
        logging.exception(f"Error in find_callees: {e}")
        return json.dumps({
            "nodes": [],
            "edges": [],
            "meta": _create_code_meta(degraded=True, error=str(e)),
        })


@mcp.tool(description="""Analyze the impact of code changes.

Determine which files and symbols are affected by code changes.
Use SCIP `symbol_id`s from `search_code_hybrid` for symbol-level analysis.

Required parameters:
- repo_id: Repository ID (required)

Optional parameters:
- changed_files: List of changed file paths
- symbol_id: The SCIP symbol ID (e.g., `scip-typescript npm pkg method.`) retrieved from `search_code_hybrid`.
- symbol_name: Symbol name to resolve when symbol_id is unknown (e.g., `firstname`)
- parent_name: Parent type name for disambiguation (e.g., `Producer`)
- symbol_kind: Filter by kind (field|method|class|function|interface|enum|type_alias|property)
- file_path: File path to scope symbol lookup (supports suffix match)
- max_depth: Maximum traversal depth (default: 10, max: 10)
- include_cross_language: Include cross-language dependencies (default: false)
- include_inferred_edges: Override inferred-edge usage (defaults to server config)
- include_field_edges: Include READS/WRITES/HAS_FIELD edges for fields/properties (default: true)
- include_schema_edges: Include SCHEMA_EXPOSES and SCHEMA_ALIASES edges for schema surfaces (default: true)
- include_path_edges: Include CODE_FIELD_PATH string path references (heuristic; may include false positives) (default: true)

Returns:
- affected_files[]: List of affected files with impact scores
- required_files[]: Files that must be read before finalizing an answer (write or schema-alias matches)
- coverage_summary: Counts by evidence type (reads/writes/schema/path/calls/contains)
- coverage_low: True when no reads/writes/schema hits were found
- action_required: Required follow-up action when coverage is shallow
- action_message: Short imperative message for the required action
- resolved_symbol_id/name/kind/file_path/parent_name: Resolved symbol info when symbol_id/name lookup succeeds
- symbol_candidates[]: Candidate symbols when the name is ambiguous
- meta: Response metadata

Note: Results are graph-derived candidates; read any file you cite to confirm behavior.
If coverage_low is true or action_required is set, do not finalize. Find the internal field name (schema/state or mapping code) and rerun impact_analysis.
If you pass file_path and the resolved_symbol_file_path does not match, do not use the results; rerun with the correct file_path or parent_name.

Examples:
- impact_analysis(repo_id="repo-123", changed_files=["src/utils.py"])
- impact_analysis(repo_id="repo-123", symbol_id="scip-typescript npm my-pkg 1.0.0 src/`utils.ts`/processData().", max_depth=5)
- impact_analysis(repo_id="repo-123", symbol_id="scip-typescript npm my-pkg 1.0.0 src/`models.ts`/User#field:email.", include_field_edges=true, include_schema_edges=true)
- impact_analysis(repo_id="repo-123", symbol_name="email", parent_name="User", symbol_kind="field", file_path="src/models.ts", include_field_edges=true, include_schema_edges=true)
- impact_analysis(repo_id="repo-123", symbol_name="email", parent_name="User", symbol_kind="field", file_path="src/models.ts", include_path_edges=true)
""")
async def impact_analysis(
    repo_id: str,
    changed_files: list = None,
    symbol_id: str = None,
    symbol_name: str = None,
    parent_name: str = None,
    symbol_kind: str = None,
    file_path: str = None,
    max_depth: int = 10,
    include_cross_language: bool = False,
    include_inferred_edges: bool | None = None,
    include_field_edges: bool = True,
    include_schema_edges: bool = True,
    include_path_edges: bool = True,
) -> str:
    """Analyze the impact of code changes."""
    # Check scope
    scope_error = _check_tool_scope("graph:read")
    if scope_error:
        return scope_error

    toolkit = get_code_toolkit()

    if not toolkit.is_available("neo4j"):
        return json.dumps({
            "affected_files": [],
            "meta": _create_code_meta(
                degraded=True,
                missing=["neo4j"],
                error="Graph backend unavailable",
            ),
        })

    if not toolkit.impact_tool:
        return json.dumps({
            "affected_files": [],
            "meta": _create_code_meta(
                degraded=True,
                missing=toolkit.get_missing_sources(),
                error=toolkit.get_error("impact_tool"),
            ),
        })

    try:
        from tools.impact_analysis import ImpactInput
        import dataclasses

        input_data = ImpactInput(
            repo_id=repo_id,
            changed_files=changed_files or [],
            symbol_id=symbol_id,
            symbol_name=symbol_name,
            parent_name=parent_name,
            symbol_kind=symbol_kind,
            file_path=file_path,
            include_cross_language=include_cross_language,
            max_depth=min(max(1, max_depth or 10), 10),
            include_inferred_edges=include_inferred_edges,
            include_field_edges=include_field_edges,
            include_schema_edges=include_schema_edges,
            include_path_edges=include_path_edges,
        )

        result = toolkit.impact_tool.analyze(input_data)
        return json.dumps(dataclasses.asdict(result), default=str)

    except Exception as e:
        logging.exception(f"Error in impact_analysis: {e}")
        return json.dumps({
            "affected_files": [],
            "meta": _create_code_meta(degraded=True, error=str(e)),
        })


@mcp.tool(description="""Analyze code changes for ADR (Architecture Decision Record) recommendation.

Detects significant architectural changes that warrant documentation.

Required parameters:
- changes: List of change objects with file_path, change_type, diff, added_lines, removed_lines

Optional parameters:
- min_confidence: Minimum confidence threshold for ADR recommendation (default: 0.7)

Returns:
- should_create_adr: Whether an ADR should be created
- confidence: Confidence score (0.0-1.0)
- triggered_heuristics: List of detected change types (dependency, api_change, etc.)
- reasons: Human-readable reasons for the recommendation
- generated_adr: Generated ADR content (when should_create_adr is True)
- code_links: Links to relevant code sections (optional)
- impact_analysis: Impact analysis results (optional)
- meta: Response metadata

Examples:
- adr_automation(changes=[{"file_path": "requirements.txt", "change_type": "modified", "diff": "+redis>=4.0.0"}])
- adr_automation(changes=[...], min_confidence=0.8)
""")
async def adr_automation(
    changes: list,
    min_confidence: float = 0.7,
) -> str:
    """Analyze code changes for ADR recommendation."""
    # Check scopes - needs both search:read and graph:read
    scope_error = _check_tool_scope("search:read")
    if scope_error:
        return scope_error
    scope_error = _check_tool_scope("graph:read")
    if scope_error:
        return scope_error

    toolkit = get_code_toolkit()

    if not toolkit.adr_tool:
        return json.dumps({
            "should_create_adr": False,
            "confidence": 0.0,
            "triggered_heuristics": [],
            "reasons": [],
            "meta": _create_code_meta(
                degraded=True,
                missing=toolkit.get_missing_sources(),
                error=toolkit.get_error("adr_tool") or "ADR tool not available",
            ),
        }, default=str)

    try:
        result = toolkit.adr_tool.execute({
            "changes": changes,
            "min_confidence": min_confidence,
        })

        # Build response with meta
        response = {
            "should_create_adr": result.get("should_create_adr", False),
            "confidence": result.get("confidence", 0.0),
            "triggered_heuristics": result.get("triggered_heuristics", []),
            "reasons": result.get("reasons", []),
            "meta": _create_code_meta(),
        }

        # Include optional fields when present
        if result.get("should_create_adr") and result.get("generated_adr"):
            response["generated_adr"] = result["generated_adr"]
        if result.get("code_links"):
            response["code_links"] = result["code_links"]
        if result.get("impact_analysis"):
            response["impact_analysis"] = result["impact_analysis"]

        return json.dumps(response, default=str)

    except Exception as e:
        logging.exception(f"Error in adr_automation: {e}")
        return json.dumps({
            "should_create_adr": False,
            "confidence": 0.0,
            "triggered_heuristics": [],
            "reasons": [],
            "meta": _create_code_meta(degraded=True, error=str(e)),
        })


@mcp.tool(description="""Generate test cases for a code symbol or file.

Creates comprehensive test cases based on code analysis.

Required parameters (at least one):
- symbol_id: SCIP symbol ID to generate tests for
- file_path: File path to generate tests for

Optional parameters:
- framework: Test framework to use (default: "pytest")
- include_edge_cases: Include edge case tests (default: true)
- include_error_cases: Include error handling tests (default: true)

Returns:
- symbol_id: The symbol that tests were generated for
- symbol_name: Human-readable symbol name
- test_cases: List of generated test cases with name, description, category, code
- file_content: Rendered test file content
- meta: Response metadata

Examples:
- test_generation(symbol_id="scip-python myapp module/my_function.")
- test_generation(file_path="/path/to/module.py", framework="pytest")
- test_generation(symbol_id="...", include_edge_cases=true, include_error_cases=false)
""")
async def test_generation(
    symbol_id: str = None,
    file_path: str = None,
    framework: str = "pytest",
    include_edge_cases: bool = True,
    include_error_cases: bool = True,
) -> str:
    """Generate test cases for a code symbol or file."""
    # Check scopes - needs both search:read and graph:read
    scope_error = _check_tool_scope("search:read")
    if scope_error:
        return scope_error
    scope_error = _check_tool_scope("graph:read")
    if scope_error:
        return scope_error

    # Validate input - at least one of symbol_id or file_path required
    if not symbol_id and not file_path:
        return json.dumps({
            "error": "Either symbol_id or file_path is required",
        })

    toolkit = get_code_toolkit()

    if not toolkit.test_gen_tool:
        return json.dumps({
            "symbol_id": symbol_id,
            "symbol_name": None,
            "test_cases": [],
            "file_content": "",
            "meta": _create_code_meta(
                degraded=True,
                missing=toolkit.get_missing_sources(),
                error=toolkit.get_error("test_gen_tool") or "Test generation tool not available",
            ),
        })

    try:
        # Build input for the tool (symbol_id takes precedence)
        input_data = {
            "framework": framework,
            "include_edge_cases": include_edge_cases,
            "include_error_cases": include_error_cases,
        }
        if symbol_id:
            input_data["symbol_id"] = symbol_id
        if file_path:
            input_data["file_path"] = file_path

        result = toolkit.test_gen_tool.execute(input_data)

        # Build response with meta
        response = {
            "symbol_id": result.get("symbol_id", symbol_id),
            "symbol_name": result.get("symbol_name"),
            "test_cases": result.get("test_cases", []),
            "file_content": result.get("file_content", ""),
            "meta": _create_code_meta(),
        }

        return json.dumps(response, default=str)

    except Exception as e:
        logging.exception(f"Error in test_generation: {e}")
        return json.dumps({
            "symbol_id": symbol_id,
            "symbol_name": None,
            "test_cases": [],
            "file_content": "",
            "meta": _create_code_meta(degraded=True, error=str(e)),
        })


@mcp.tool(description="""Analyze a pull request for issues, impact, and recommendations.

Provides comprehensive PR analysis including security checks, convention compliance,
impact analysis, and ADR recommendations.

Required parameters:
- repo_id: Repository ID (required, non-empty)
- diff: The PR diff content (required)

Optional parameters:
- pr_number: Pull request number
- title: PR title
- body: PR description body
- check_impact: Run impact analysis (default: true)
- check_adr: Check for ADR recommendations (default: true)
- check_security: Run security checks (default: true)
- check_conventions: Check coding conventions (default: true)

Returns:
- summary: PR summary with files_changed, additions, deletions, languages, main_areas,
          complexity_score, affected_files, suggested_adr, adr_reason
- issues: List of detected issues with severity, category, file_path, line_number,
          message, suggestion
- meta: Response metadata

Examples:
- pr_analysis(repo_id="my-repo", diff="diff --git a/file.py...")
- pr_analysis(repo_id="my-repo", diff="...", pr_number=123, title="Add feature")
- pr_analysis(repo_id="my-repo", diff="...", check_security=true, check_adr=false)
""")
async def pr_analysis(
    repo_id: str,
    diff: str,
    pr_number: int = None,
    title: str = None,
    body: str = None,
    check_impact: bool = True,
    check_adr: bool = True,
    check_security: bool = True,
    check_conventions: bool = True,
) -> str:
    """Analyze a pull request for issues and recommendations."""
    # Check scopes - needs both search:read and graph:read
    scope_error = _check_tool_scope("search:read")
    if scope_error:
        return scope_error
    scope_error = _check_tool_scope("graph:read")
    if scope_error:
        return scope_error

    # Validate required inputs
    if not repo_id:
        return json.dumps({
            "error": "repo_id is required and cannot be empty",
        })

    toolkit = get_code_toolkit()

    if not toolkit.pr_analysis_tool:
        return json.dumps({
            "summary": None,
            "issues": [],
            "meta": _create_code_meta(
                degraded=True,
                missing=toolkit.get_missing_sources(),
                error=toolkit.get_error("pr_analysis_tool") or "PR analysis tool not available",
            ),
        }, default=str)

    try:
        # Build input for the tool
        input_data = {
            "repo_id": repo_id,
            "diff": diff,
            "check_impact": check_impact,
            "check_adr": check_adr,
            "check_security": check_security,
            "check_conventions": check_conventions,
        }
        if pr_number is not None:
            input_data["pr_number"] = pr_number
        if title:
            input_data["title"] = title
        if body:
            input_data["body"] = body

        result = toolkit.pr_analysis_tool.execute(input_data)

        # Build response with meta
        response = {
            "summary": result.get("summary"),
            "issues": result.get("issues", []),
            "meta": _create_code_meta(
                request_id=result.get("request_id"),
            ),
        }

        return json.dumps(response, default=str)

    except Exception as e:
        logging.exception(f"Error in pr_analysis: {e}")
        return json.dumps({
            "summary": None,
            "issues": [],
            "meta": _create_code_meta(degraded=True, error=str(e)),
        })


# =============================================================================
# BUSINESS CONCEPT TOOLS (Separate MCP endpoint - /concepts/claude/sse/{user_id})
# =============================================================================


# DISABLED: Business Concepts tools not exposed to LLM
# @concept_mcp.tool(description="""Extract business concepts from a memory or text.
#
# Requires BUSINESS_CONCEPTS_ENABLED=true environment variable.
#
# Parameters:
# - memory_id: UUID of the memory to extract from (required if content not provided)
# - content: Text content to extract from (required if memory_id not provided)
# - category: Optional category for concept scoping
# - store: Whether to store extracted concepts in graph (default: true)
# - access_entity: Access control scope for stored concepts (defaults to memory scope or user)
#
# Returns:
# - entities: Extracted business entities with types and importance
# - concepts: Extracted business concepts with types and confidence
# - summary: Brief summary of main topics
# - language: Detected language (en, de, mixed)
# - stored_entities: Count of entities stored (if store=true)
# - stored_concepts: Count of concepts stored (if store=true)
# """)
async def extract_business_concepts(
    memory_id: str = None,
    content: str = None,
    category: str = None,
    store: bool = True,
    access_entity: str = None,
) -> str:
    uid = user_id_var.get(None)
    client_name = client_name_var.get(None)

    if not uid:
        return json.dumps({"error": "user_id not provided"})
    if not client_name:
        return json.dumps({"error": "client_name not provided"})

    # Check if concepts are enabled
    from app.config import BusinessConceptsConfig
    if not BusinessConceptsConfig.is_enabled():
        return json.dumps({
            "error": "Business concepts system not enabled. Set BUSINESS_CONCEPTS_ENABLED=true",
            "enabled": False,
        })

    if not content and not memory_id:
        return json.dumps({"error": "Either memory_id or content is required"})

    try:
        # If memory_id provided but no content, fetch content from DB
        memory_access_entity = None
        if memory_id:
            db = SessionLocal()
            try:
                memory = db.query(Memory).filter(Memory.id == uuid.UUID(memory_id)).first()
                if not memory:
                    return json.dumps({"error": f"Memory {memory_id} not found"})
                if not content:
                    content = memory.content
                if isinstance(memory.metadata_, dict):
                    if category is None:
                        category = memory.metadata_.get("category")
                    memory_access_entity = memory.metadata_.get("access_entity")
            finally:
                db.close()

        if memory_access_entity and access_entity and memory_access_entity != access_entity:
            return json.dumps({
                "error": "access_entity does not match memory scope",
                "memory_access_entity": memory_access_entity,
                "access_entity": access_entity,
            })

        # Use provided memory_id or generate a temporary one
        effective_memory_id = memory_id or str(uuid.uuid4())

        from app.utils.concept_extractor import extract_from_memory

        result = extract_from_memory(
            memory_id=effective_memory_id,
            user_id=uid,
            content=content,
            category=category,
            store_in_graph=store,
            access_entity=memory_access_entity or access_entity,
        )

        return json.dumps(result, indent=2, default=str)

    except Exception as e:
        logging.exception(f"Error extracting concepts: {e}")
        return json.dumps({"error": str(e)})


# @concept_mcp.tool(description="""List business concepts with optional filters.
#
# Requires BUSINESS_CONCEPTS_ENABLED=true environment variable.
#
# Parameters:
# - category: Filter by category
# - concept_type: Filter by type (causal, pattern, comparison, trend, contradiction, hypothesis, fact)
# - min_confidence: Minimum confidence threshold (0.0-1.0)
# - limit: Maximum results (default: 50)
# - offset: Pagination offset
#
# Returns:
# - concepts[]: List of concepts with name, type, confidence, category, etc.
# - count: Number of concepts returned
# """)
async def list_business_concepts(
    category: str = None,
    concept_type: str = None,
    min_confidence: float = None,
    limit: int = 50,
    offset: int = 0,
) -> str:
    uid = user_id_var.get(None)
    client_name = client_name_var.get(None)

    if not uid:
        return json.dumps({"error": "user_id not provided"})
    if not client_name:
        return json.dumps({"error": "client_name not provided"})

    from app.config import BusinessConceptsConfig
    if not BusinessConceptsConfig.is_enabled():
        return json.dumps({
            "error": "Business concepts system not enabled",
            "enabled": False,
        })

    try:
        from app.graph.concept_ops import list_concepts
        principal = principal_var.get(None)
        access_entities, access_entity_prefixes = _build_graph_access_filters(principal, uid)

        concepts = list_concepts(
            user_id=uid,
            category=category,
            concept_type=concept_type,
            min_confidence=min_confidence,
            limit=limit,
            offset=offset,
            access_entities=access_entities,
            access_entity_prefixes=access_entity_prefixes,
        )

        return json.dumps({
            "concepts": concepts,
            "count": len(concepts),
            "enabled": True,
        }, indent=2, default=str)

    except Exception as e:
        logging.exception(f"Error listing concepts: {e}")
        return json.dumps({"error": str(e)})


# @concept_mcp.tool(description="""Get a specific business concept by name.
#
# Requires BUSINESS_CONCEPTS_ENABLED=true environment variable.
#
# Parameters:
# - name: Name of the concept (required)
#
# Returns:
# - Concept details including name, type, confidence, evidence, etc.
# """)
async def get_business_concept(
    name: str,
) -> str:
    uid = user_id_var.get(None)
    client_name = client_name_var.get(None)

    if not uid:
        return json.dumps({"error": "user_id not provided"})
    if not client_name:
        return json.dumps({"error": "client_name not provided"})

    from app.config import BusinessConceptsConfig
    if not BusinessConceptsConfig.is_enabled():
        return json.dumps({
            "error": "Business concepts system not enabled",
            "enabled": False,
        })

    try:
        from app.graph.concept_ops import get_concept

        principal = principal_var.get(None)
        access_entities, access_entity_prefixes = _build_graph_access_filters(principal, uid)

        concept = get_concept(
            user_id=uid,
            name=name,
            access_entities=access_entities,
            access_entity_prefixes=access_entity_prefixes,
        )

        if concept:
            return json.dumps(concept, indent=2, default=str)
        else:
            return json.dumps({"error": f"Concept '{name}' not found"})

    except Exception as e:
        logging.exception(f"Error getting concept: {e}")
        return json.dumps({"error": str(e)})


# @concept_mcp.tool(description="""Search business concepts by text query.
#
# Uses semantic (vector) search when embeddings are enabled, falling back to
# full-text search otherwise. Semantic search understands meaning, not just keywords.
#
# Requires BUSINESS_CONCEPTS_ENABLED=true environment variable.
#
# Parameters:
# - query: Search query (required)
# - limit: Maximum results (default: 20)
# - use_semantic: Use semantic search if available (default: true)
# - min_score: Minimum similarity score 0-1 for semantic search (default: 0.5)
#
# Returns:
# - concepts[]: Matching concepts with search/similarity scores
# - count: Number of results
# - search_type: "semantic" or "fulltext"
# """)
async def search_business_concepts(
    query: str,
    limit: int = 20,
    use_semantic: bool = True,
    min_score: float = 0.5,
) -> str:
    uid = user_id_var.get(None)
    client_name = client_name_var.get(None)

    if not uid:
        return json.dumps({"error": "user_id not provided"})
    if not client_name:
        return json.dumps({"error": "client_name not provided"})

    from app.config import BusinessConceptsConfig
    if not BusinessConceptsConfig.is_enabled():
        return json.dumps({
            "error": "Business concepts system not enabled",
            "enabled": False,
        })

    try:
        from app.graph.concept_ops import (
            search_concepts,
            semantic_search_concepts,
            is_vector_search_enabled,
        )
        principal = principal_var.get(None)
        access_entities, access_entity_prefixes = _build_graph_access_filters(principal, uid)

        search_type = "fulltext"

        # Use semantic search if requested and available
        if use_semantic and is_vector_search_enabled():
            concepts = semantic_search_concepts(
                user_id=uid,
                query=query,
                top_k=limit,
                min_score=min_score,
                access_entities=access_entities,
                access_entity_prefixes=access_entity_prefixes,
            )
            search_type = "semantic"
        else:
            concepts = search_concepts(
                user_id=uid,
                query=query,
                limit=limit,
                access_entities=access_entities,
                access_entity_prefixes=access_entity_prefixes,
            )

        return json.dumps({
            "query": query,
            "concepts": concepts,
            "count": len(concepts),
            "search_type": search_type,
            "embedding_enabled": is_vector_search_enabled(),
        }, indent=2, default=str)

    except Exception as e:
        logging.exception(f"Error searching concepts: {e}")
        return json.dumps({"error": str(e)})


# @concept_mcp.tool(description="""Find concepts semantically similar to a given concept.
#
# Uses vector embeddings to find concepts that are semantically related,
# even if they don't share exact keywords. Useful for:
# - Discovering related concepts
# - Finding potential duplicates
# - Building concept clusters
#
# Requires BUSINESS_CONCEPTS_ENABLED=true and BUSINESS_CONCEPTS_EMBEDDING_ENABLED=true.
#
# Parameters:
# - concept_name: Name of the seed concept (required)
# - top_k: Number of similar concepts to return (default: 5)
# - find_duplicates: If true, uses higher threshold to find potential duplicates (default: false)
#
# Returns:
# - similar_concepts[]: List of similar concepts with similarity scores
# - count: Number of results
# - seed_concept: The concept used as seed
# """)
async def find_similar_business_concepts(
    concept_name: str,
    top_k: int = 5,
    find_duplicates: bool = False,
) -> str:
    uid = user_id_var.get(None)
    client_name = client_name_var.get(None)

    if not uid:
        return json.dumps({"error": "user_id not provided"})
    if not client_name:
        return json.dumps({"error": "client_name not provided"})

    from app.config import BusinessConceptsConfig
    if not BusinessConceptsConfig.is_enabled():
        return json.dumps({
            "error": "Business concepts system not enabled",
            "enabled": False,
        })

    try:
        from app.graph.concept_ops import (
            find_similar_concepts,
            find_concept_duplicates,
            is_vector_search_enabled,
        )
        principal = principal_var.get(None)
        access_entities, access_entity_prefixes = _build_graph_access_filters(principal, uid)

        if not is_vector_search_enabled():
            return json.dumps({
                "error": "Vector embeddings not enabled",
                "help": "Set BUSINESS_CONCEPTS_EMBEDDING_ENABLED=true to use this feature",
                "embedding_enabled": False,
            })

        if find_duplicates:
            similar = find_concept_duplicates(
                user_id=uid,
                concept_name=concept_name,
                access_entities=access_entities,
                access_entity_prefixes=access_entity_prefixes,
            )
        else:
            similar = find_similar_concepts(
                user_id=uid,
                concept_name=concept_name,
                top_k=top_k,
                access_entities=access_entities,
                access_entity_prefixes=access_entity_prefixes,
            )

        return json.dumps({
            "seed_concept": concept_name,
            "similar_concepts": similar,
            "count": len(similar),
            "mode": "duplicates" if find_duplicates else "similar",
        }, indent=2, default=str)

    except Exception as e:
        logging.exception(f"Error finding similar concepts: {e}")
        return json.dumps({"error": str(e)})


# @concept_mcp.tool(description="""List business entities (companies, people, products, etc.)
#
# Requires BUSINESS_CONCEPTS_ENABLED=true environment variable.
#
# Parameters:
# - entity_type: Filter by type (company, person, product, market, metric, business_model, technology, strategy)
# - min_importance: Minimum importance threshold (0.0-1.0)
# - limit: Maximum results (default: 50)
#
# Returns:
# - entities[]: List of business entities
# - count: Number of entities returned
# """)
async def list_business_entities(
    entity_type: str = None,
    min_importance: float = None,
    limit: int = 50,
) -> str:
    uid = user_id_var.get(None)
    client_name = client_name_var.get(None)

    if not uid:
        return json.dumps({"error": "user_id not provided"})
    if not client_name:
        return json.dumps({"error": "client_name not provided"})

    from app.config import BusinessConceptsConfig
    if not BusinessConceptsConfig.is_enabled():
        return json.dumps({
            "error": "Business concepts system not enabled",
            "enabled": False,
        })

    try:
        from app.graph.concept_ops import list_business_entities as list_entities
        principal = principal_var.get(None)
        access_entities, access_entity_prefixes = _build_graph_access_filters(principal, uid)

        entities = list_entities(
            user_id=uid,
            entity_type=entity_type,
            min_importance=min_importance,
            limit=limit,
            access_entities=access_entities,
            access_entity_prefixes=access_entity_prefixes,
        )

        return json.dumps({
            "entities": entities,
            "count": len(entities),
        }, indent=2, default=str)

    except Exception as e:
        logging.exception(f"Error listing entities: {e}")
        return json.dumps({"error": str(e)})


# @concept_mcp.tool(description="""Get the business concept network graph for visualization.
#
# Requires BUSINESS_CONCEPTS_ENABLED=true environment variable.
#
# Parameters:
# - concept_name: Optional seed concept (if omitted, returns full network)
# - depth: Traversal depth 1-3 (default: 2)
# - limit: Maximum nodes (default: 50)
#
# Returns:
# - nodes[]: Concept and entity nodes with properties
# - edges[]: Relationships between nodes
# """)
async def get_concept_network(
    concept_name: str = None,
    depth: int = 2,
    limit: int = 50,
) -> str:
    uid = user_id_var.get(None)
    client_name = client_name_var.get(None)

    if not uid:
        return json.dumps({"error": "user_id not provided"})
    if not client_name:
        return json.dumps({"error": "client_name not provided"})

    from app.config import BusinessConceptsConfig
    if not BusinessConceptsConfig.is_enabled():
        return json.dumps({
            "error": "Business concepts system not enabled",
            "enabled": False,
        })

    try:
        from app.graph.concept_ops import get_concept_network as get_network
        principal = principal_var.get(None)
        access_entities, access_entity_prefixes = _build_graph_access_filters(principal, uid)

        result = get_network(
            user_id=uid,
            concept_name=concept_name,
            depth=depth,
            limit=limit,
            access_entities=access_entities,
            access_entity_prefixes=access_entity_prefixes,
        )

        return json.dumps(result, indent=2, default=str)

    except Exception as e:
        logging.exception(f"Error getting concept network: {e}")
        return json.dumps({"error": str(e)})


# @concept_mcp.tool(description="""Find contradictions between business concepts.
#
# Requires BUSINESS_CONCEPTS_ENABLED=true and BUSINESS_CONCEPTS_CONTRADICTION_DETECTION=true.
#
# Parameters:
# - concept_name: Optional concept to analyze (if omitted, finds all contradictions)
# - category: Optional category filter
# - min_severity: Minimum severity threshold 0.0-1.0 (default: 0.5)
#
# Returns:
# - contradictions[]: List of detected contradictions with severity and evidence
# - count: Number of contradictions found
# """)
async def find_concept_contradictions(
    concept_name: str = None,
    category: str = None,
    min_severity: float = 0.5,
) -> str:
    uid = user_id_var.get(None)
    client_name = client_name_var.get(None)

    if not uid:
        return json.dumps({"error": "user_id not provided"})
    if not client_name:
        return json.dumps({"error": "client_name not provided"})

    from app.config import BusinessConceptsConfig
    if not BusinessConceptsConfig.is_enabled():
        return json.dumps({
            "error": "Business concepts system not enabled",
            "enabled": False,
        })

    try:
        from app.graph.convergence_detector import (
            detect_contradictions_for_concept,
            find_all_contradictions,
        )
        principal = principal_var.get(None)
        access_entities, access_entity_prefixes = _build_graph_access_filters(principal, uid)

        if concept_name:
            contradictions = detect_contradictions_for_concept(
                user_id=uid,
                concept_name=concept_name,
                store=True,
                access_entities=access_entities,
                access_entity_prefixes=access_entity_prefixes,
            )
        else:
            contradictions = find_all_contradictions(
                user_id=uid,
                category=category,
                min_severity=min_severity,
                access_entities=access_entities,
                access_entity_prefixes=access_entity_prefixes,
            )

        return json.dumps({
            "concept": concept_name,
            "contradictions": contradictions,
            "count": len(contradictions),
        }, indent=2, default=str)

    except Exception as e:
        logging.exception(f"Error finding contradictions: {e}")
        return json.dumps({"error": str(e)})


# @concept_mcp.tool(description="""Analyze convergence of evidence for a concept.
#
# Requires BUSINESS_CONCEPTS_ENABLED=true.
#
# Analyzes whether a concept has strong convergent evidence from multiple
# independent sources across time and domains.
#
# Parameters:
# - concept_name: Name of the concept to analyze (required)
# - min_evidence: Minimum supporting memories required (default: 3)
#
# Returns:
# - convergence_score: Overall convergence score 0.0-1.0
# - is_strong: Whether meets strong convergence threshold
# - temporal_spread_days: Days between first and last evidence
# - category_diversity: Diversity of evidence sources
# - recommended_confidence: Suggested confidence boost
# - evidence_count: Number of supporting memories
# """)
async def analyze_concept_convergence(
    concept_name: str,
    min_evidence: int = 3,
) -> str:
    uid = user_id_var.get(None)
    client_name = client_name_var.get(None)

    if not uid:
        return json.dumps({"error": "user_id not provided"})
    if not client_name:
        return json.dumps({"error": "client_name not provided"})

    from app.config import BusinessConceptsConfig
    if not BusinessConceptsConfig.is_enabled():
        return json.dumps({
            "error": "Business concepts system not enabled",
            "enabled": False,
        })

    try:
        from app.graph.convergence_detector import ConvergenceDetector

        principal = principal_var.get(None)
        access_entities, access_entity_prefixes = _build_graph_access_filters(principal, uid)
        detector = ConvergenceDetector(
            user_id=uid,
            access_entities=access_entities,
            access_entity_prefixes=access_entity_prefixes,
        )
        result = detector.analyze_concept_convergence(
            concept_name=concept_name,
            min_evidence=min_evidence,
        )

        if result:
            return json.dumps(result.to_dict(), indent=2, default=str)
        else:
            return json.dumps({
                "concept_name": concept_name,
                "error": f"Insufficient evidence (need at least {min_evidence} supporting memories)",
            })

    except Exception as e:
        logging.exception(f"Error analyzing convergence: {e}")
        return json.dumps({"error": str(e)})


# @concept_mcp.tool(description="""Delete a business concept.
#
# Requires BUSINESS_CONCEPTS_ENABLED=true.
#
# Parameters:
# - name: Name of the concept to delete (required)
# - access_entity: Access control scope to target (defaults to `user:<user_id>` if omitted)
#
# Returns:
# - deleted: Whether the concept was deleted
# """)
async def delete_business_concept(
    name: str,
    access_entity: str = None,
) -> str:
    uid = user_id_var.get(None)
    client_name = client_name_var.get(None)

    if not uid:
        return json.dumps({"error": "user_id not provided"})
    if not client_name:
        return json.dumps({"error": "client_name not provided"})

    from app.config import BusinessConceptsConfig
    if not BusinessConceptsConfig.is_enabled():
        return json.dumps({
            "error": "Business concepts system not enabled",
            "enabled": False,
        })

    try:
        from app.graph.concept_ops import delete_concept

        deleted = delete_concept(user_id=uid, name=name, access_entity=access_entity)

        return json.dumps({
            "name": name,
            "deleted": deleted,
        })

    except Exception as e:
        logging.exception(f"Error deleting concept: {e}")
        return json.dumps({"error": str(e)})


@mcp_router.get("/{client_name}/sse/{user_id}")
async def handle_sse(request: Request):
    """Handle SSE connections for a specific user and client.

    Authentication is required - the user_id in the path is IGNORED.
    The authenticated user comes from the JWT token in the Authorization header.
    """
    # Authenticate the request - user_id comes from JWT, not path
    try:
        principal = _extract_principal_from_request(request)
    except Exception as e:
        if HAS_SECURITY and isinstance(e, AuthenticationError):
            return JSONResponse(
                status_code=401,
                content={"error": e.message, "code": e.code},
                headers={"WWW-Authenticate": 'Bearer realm="mcp"'},
            )
        raise

    # Extract and validate DPoP proof if present (binds session to DPoP key)
    dpop_thumbprint = None
    if HAS_SECURITY:
        try:
            auth_header = request.headers.get("authorization", "")
            token = auth_header[7:] if auth_header.lower().startswith("bearer ") else ""
            dpop_thumbprint = await _extract_dpop_thumbprint(request, token)
            if dpop_thumbprint:
                principal.dpop_thumbprint = dpop_thumbprint
        except Exception as e:
            if isinstance(e, AuthenticationError):
                return JSONResponse(
                    status_code=401,
                    content={"error": e.message, "code": e.code},
                    headers={"WWW-Authenticate": 'Bearer realm="mcp"'},
                )
            raise

    # Set context variables from authenticated principal (NOT from path)
    user_token = user_id_var.set(principal.user_id)
    client_name = request.path_params.get("client_name")
    client_token = client_name_var.set(client_name or "")
    org_token = org_id_var.set(principal.org_id)
    principal_token = principal_var.set(principal)

    session_id = None
    try:
        # Handle SSE connection
        async with sse.connect_sse(
            request.scope,
            request.receive,
            request._send,
        ) as (read_stream, write_stream):
            # Capture session_id and create binding
            session_id = sse.get_session_id(request.scope)
            if session_id and HAS_SECURITY:
                store = get_session_binding_store()
                store.create(
                    session_id=session_id,
                    user_id=principal.user_id,
                    org_id=principal.org_id,
                    dpop_thumbprint=dpop_thumbprint,
                )

            await mcp._mcp_server.run(
                read_stream,
                write_stream,
                mcp._mcp_server.create_initialization_options(),
            )
    finally:
        # Clean up session binding on disconnect
        if session_id and HAS_SECURITY:
            get_session_binding_store().delete(session_id)

        # Clean up context variables
        user_id_var.reset(user_token)
        client_name_var.reset(client_token)
        org_id_var.reset(org_token)
        principal_var.reset(principal_token)


@mcp_router.post("/messages/")
async def handle_get_message(request: Request):
    return await handle_post_message(request)


@mcp_router.post("/{client_name}/sse/{user_id}/messages/")
async def handle_mcp_post_message(request: Request):
    return await handle_post_message(request)


async def handle_post_message(request: Request):
    """Handle POST messages for SSE.

    Authentication is required - context vars must be set from authenticated principal.
    Session binding is validated to prevent session hijacking.
    DPoP validation is performed when session was bound with DPoP.
    """
    # Validate session_id from query params (required for session binding)
    if HAS_SECURITY:
        session_id_str = request.query_params.get("session_id")
        if not session_id_str:
            return JSONResponse(
                status_code=400,
                content={"error": "Missing session_id", "code": "MISSING_SESSION_ID"},
            )

        try:
            from uuid import UUID
            session_id = UUID(session_id_str)
        except ValueError:
            return JSONResponse(
                status_code=400,
                content={"error": "Invalid session_id format", "code": "INVALID_SESSION_ID"},
            )

    # Authenticate the request
    try:
        principal = _extract_principal_from_request(request)
    except Exception as e:
        if HAS_SECURITY and isinstance(e, AuthenticationError):
            return JSONResponse(
                status_code=401,
                content={"error": e.message, "code": e.code},
                headers={"WWW-Authenticate": 'Bearer realm="mcp"'},
            )
        raise

    # Extract and validate DPoP proof if present
    dpop_thumbprint = None
    if HAS_SECURITY:
        try:
            # Get token for DPoP ath validation
            auth_header = request.headers.get("authorization", "")
            token = auth_header[7:] if auth_header.lower().startswith("bearer ") else ""
            dpop_thumbprint = await _extract_dpop_thumbprint(request, token)
        except Exception as e:
            if isinstance(e, AuthenticationError):
                return JSONResponse(
                    status_code=401,
                    content={"error": e.message, "code": e.code},
                    headers={"WWW-Authenticate": 'Bearer realm="mcp"'},
                )
            raise

    # Validate session binding (after authentication)
    if HAS_SECURITY:
        store = get_session_binding_store()
        if not store.validate(
            session_id,
            principal.user_id,
            principal.org_id,
            dpop_thumbprint=dpop_thumbprint,
        ):
            return JSONResponse(
                status_code=403,
                content={"error": "Session binding mismatch", "code": "SESSION_BINDING_INVALID"},
            )

    # Set context variables from authenticated principal
    user_token = user_id_var.set(principal.user_id)
    client_name = request.path_params.get("client_name", "")
    client_token = client_name_var.set(client_name)
    org_token = org_id_var.set(principal.org_id)
    principal_token = principal_var.set(principal)

    try:
        body = await request.body()

        # Create a simple receive function that returns the body
        async def receive():
            return {"type": "http.request", "body": body, "more_body": False}

        # Create a simple send function that does nothing
        async def send(message):
            return {}

        # Call handle_post_message with the correct arguments
        await sse.handle_post_message(request.scope, receive, send)

        # Return a success response
        return {"status": "ok"}
    finally:
        user_id_var.reset(user_token)
        client_name_var.reset(client_token)
        org_id_var.reset(org_token)
        principal_var.reset(principal_token)


# =============================================================================
# BUSINESS CONCEPTS MCP ENDPOINT - /concepts/claude/sse/{user_id}
# =============================================================================
# This separate endpoint keeps Business Concept tools isolated from the
# main memory tools to reduce context window bloat in daily use.


@concept_router.get("/{client_name}/sse/{user_id}")
async def handle_concept_sse(request: Request):
    """Handle SSE connections for Business Concepts MCP.

    Authentication is required - the user_id in the path is IGNORED.
    The authenticated user comes from the JWT token in the Authorization header.
    """
    # Authenticate the request - user_id comes from JWT, not path
    try:
        principal = _extract_principal_from_request(request)
    except Exception as e:
        if HAS_SECURITY and isinstance(e, AuthenticationError):
            return JSONResponse(
                status_code=401,
                content={"error": e.message, "code": e.code},
                headers={"WWW-Authenticate": 'Bearer realm="concepts"'},
            )
        raise

    # Extract and validate DPoP proof if present (binds session to DPoP key)
    dpop_thumbprint = None
    if HAS_SECURITY:
        try:
            auth_header = request.headers.get("authorization", "")
            token = auth_header[7:] if auth_header.lower().startswith("bearer ") else ""
            dpop_thumbprint = await _extract_dpop_thumbprint(request, token)
            if dpop_thumbprint:
                principal.dpop_thumbprint = dpop_thumbprint
        except Exception as e:
            if isinstance(e, AuthenticationError):
                return JSONResponse(
                    status_code=401,
                    content={"error": e.message, "code": e.code},
                    headers={"WWW-Authenticate": 'Bearer realm="concepts"'},
                )
            raise

    # Set context variables from authenticated principal (NOT from path)
    user_token = user_id_var.set(principal.user_id)
    client_name = request.path_params.get("client_name")
    client_token = client_name_var.set(client_name or "")
    org_token = org_id_var.set(principal.org_id)
    principal_token = principal_var.set(principal)

    session_id = None
    try:
        # Handle SSE connection using concept_mcp server
        async with concept_sse.connect_sse(
            request.scope,
            request.receive,
            request._send,
        ) as (read_stream, write_stream):
            # Capture session_id and create binding
            session_id = concept_sse.get_session_id(request.scope)
            if session_id and HAS_SECURITY:
                store = get_session_binding_store()
                store.create(
                    session_id=session_id,
                    user_id=principal.user_id,
                    org_id=principal.org_id,
                    dpop_thumbprint=dpop_thumbprint,
                )

            await concept_mcp._mcp_server.run(
                read_stream,
                write_stream,
                concept_mcp._mcp_server.create_initialization_options(),
            )
    finally:
        # Clean up session binding on disconnect
        if session_id and HAS_SECURITY:
            get_session_binding_store().delete(session_id)

        # Clean up context variables
        user_id_var.reset(user_token)
        client_name_var.reset(client_token)
        org_id_var.reset(org_token)
        principal_var.reset(principal_token)


@concept_router.post("/messages/")
async def handle_concept_get_message(request: Request):
    return await handle_concept_post_message(request)


@concept_router.post("/{client_name}/sse/{user_id}/messages/")
async def handle_concept_post_message_route(request: Request):
    return await handle_concept_post_message(request)


async def handle_concept_post_message(request: Request):
    """Handle POST messages for Business Concepts SSE.

    Authentication is required - context vars must be set from authenticated principal.
    Session binding is validated to prevent session hijacking.
    DPoP validation is performed when session was bound with DPoP.
    """
    # Validate session_id from query params (required for session binding)
    if HAS_SECURITY:
        session_id_str = request.query_params.get("session_id")
        if not session_id_str:
            return JSONResponse(
                status_code=400,
                content={"error": "Missing session_id", "code": "MISSING_SESSION_ID"},
            )

        try:
            from uuid import UUID
            session_id = UUID(session_id_str)
        except ValueError:
            return JSONResponse(
                status_code=400,
                content={"error": "Invalid session_id format", "code": "INVALID_SESSION_ID"},
            )

    # Authenticate the request
    try:
        principal = _extract_principal_from_request(request)
    except Exception as e:
        if HAS_SECURITY and isinstance(e, AuthenticationError):
            return JSONResponse(
                status_code=401,
                content={"error": e.message, "code": e.code},
                headers={"WWW-Authenticate": 'Bearer realm="concepts"'},
            )
        raise

    # Extract and validate DPoP proof if present
    dpop_thumbprint = None
    if HAS_SECURITY:
        try:
            # Get token for DPoP ath validation
            auth_header = request.headers.get("authorization", "")
            token = auth_header[7:] if auth_header.lower().startswith("bearer ") else ""
            dpop_thumbprint = await _extract_dpop_thumbprint(request, token)
        except Exception as e:
            if isinstance(e, AuthenticationError):
                return JSONResponse(
                    status_code=401,
                    content={"error": e.message, "code": e.code},
                    headers={"WWW-Authenticate": 'Bearer realm="concepts"'},
                )
            raise

    # Validate session binding (after authentication)
    if HAS_SECURITY:
        store = get_session_binding_store()
        if not store.validate(
            session_id,
            principal.user_id,
            principal.org_id,
            dpop_thumbprint=dpop_thumbprint,
        ):
            return JSONResponse(
                status_code=403,
                content={"error": "Session binding mismatch", "code": "SESSION_BINDING_INVALID"},
            )

    # Set context variables from authenticated principal
    user_token = user_id_var.set(principal.user_id)
    client_name = request.path_params.get("client_name", "")
    client_token = client_name_var.set(client_name)
    org_token = org_id_var.set(principal.org_id)
    principal_token = principal_var.set(principal)

    try:
        body = await request.body()

        # Create a simple receive function that returns the body
        async def receive():
            return {"type": "http.request", "body": body, "more_body": False}

        # Create a simple send function that does nothing
        async def send(message):
            return {}

        # Call handle_post_message with the correct arguments
        await concept_sse.handle_post_message(request.scope, receive, send)

        # Return a success response
        return {"status": "ok"}
    finally:
        user_id_var.reset(user_token)
        client_name_var.reset(client_token)
        org_id_var.reset(org_token)
        principal_var.reset(principal_token)


def setup_mcp_server(app: FastAPI):
    """Setup MCP server with the FastAPI application"""
    mcp._mcp_server.name = "mem0-mcp-server"
    concept_mcp._mcp_server.name = "business-concepts-mcp-server"

    # Include MCP router in the FastAPI app
    app.include_router(mcp_router)

    # Include Business Concepts router (separate endpoint for concept tools)
    app.include_router(concept_router)
