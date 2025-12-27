import logging
import hashlib
from datetime import UTC, datetime
from typing import List, Optional, Set
from uuid import UUID

from app.database import get_db
from app.models import (
    AccessControl,
    App,
    Category,
    Memory,
    MemoryAccessLog,
    MemoryState,
    MemoryStatusHistory,
    User,
)
from app.schemas import MemoryResponse
from app.security.dependencies import get_current_principal, require_scopes
from app.security.types import Principal, Scope
from app.utils.axis_tags import process_memory_input
from app.utils.memory import get_memory_client
from app.utils.permissions import check_memory_access_permissions
from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi_pagination import Page, Params
from fastapi_pagination.ext.sqlalchemy import paginate as sqlalchemy_paginate
from pydantic import BaseModel
from sqlalchemy import func, String, cast, text
from sqlalchemy.orm import Session, joinedload

router = APIRouter(prefix="/api/v1/memories", tags=["memories"])


def get_memory_or_404(db: Session, memory_id: UUID) -> Memory:
    memory = db.query(Memory).filter(Memory.id == memory_id).first()
    if not memory:
        raise HTTPException(status_code=404, detail="Memory not found")
    return memory


def update_memory_state(db: Session, memory_id: UUID, new_state: MemoryState, user_id: UUID):
    memory = get_memory_or_404(db, memory_id)
    old_state = memory.state

    # Update memory state
    memory.state = new_state
    if new_state == MemoryState.archived:
        memory.archived_at = datetime.now(UTC)
    elif new_state == MemoryState.deleted:
        memory.deleted_at = datetime.now(UTC)

    # Record state change
    history = MemoryStatusHistory(
        memory_id=memory_id,
        changed_by=user_id,
        old_state=old_state,
        new_state=new_state
    )
    db.add(history)
    db.commit()
    return memory


def get_accessible_memory_ids(db: Session, app_id: UUID) -> Set[UUID]:
    """
    Get the set of memory IDs that the app has access to based on app-level ACL rules.
    Returns all memory IDs if no specific restrictions are found.
    """
    # Get app-level access controls
    app_access = db.query(AccessControl).filter(
        AccessControl.subject_type == "app",
        AccessControl.subject_id == app_id,
        AccessControl.object_type == "memory"
    ).all()

    # If no app-level rules exist, return None to indicate all memories are accessible
    if not app_access:
        return None

    # Initialize sets for allowed and denied memory IDs
    allowed_memory_ids = set()
    denied_memory_ids = set()

    # Process app-level rules
    for rule in app_access:
        if rule.effect == "allow":
            if rule.object_id:  # Specific memory access
                allowed_memory_ids.add(rule.object_id)
            else:  # All memories access
                return None  # All memories allowed
        elif rule.effect == "deny":
            if rule.object_id:  # Specific memory denied
                denied_memory_ids.add(rule.object_id)
            else:  # All memories denied
                return set()  # No memories accessible

    # Remove denied memories from allowed set
    if allowed_memory_ids:
        allowed_memory_ids -= denied_memory_ids

    return allowed_memory_ids


# List all memories with filtering
@router.get("/", response_model=Page[MemoryResponse])
async def list_memories(
    principal: Principal = Depends(require_scopes(Scope.MEMORIES_READ)),
    app_id: Optional[UUID] = None,
    from_date: Optional[int] = Query(
        None,
        description="Filter memories created after this date (timestamp)",
        examples=[1718505600]
    ),
    to_date: Optional[int] = Query(
        None,
        description="Filter memories created before this date (timestamp)",
        examples=[1718505600]
    ),
    categories: Optional[str] = None,
    params: Params = Depends(),
    search_query: Optional[str] = None,
    sort_column: Optional[str] = Query(None, description="Column to sort by (memory, categories, app_name, created_at)"),
    sort_direction: Optional[str] = Query(None, description="Sort direction (asc or desc)"),
    db: Session = Depends(get_db)
):
    # Get user from principal (user_id from JWT, not from query param)
    user = db.query(User).filter(User.user_id == principal.user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    # Build base query
    query = db.query(Memory).filter(
        Memory.user_id == user.id,
        Memory.state != MemoryState.deleted,
        Memory.state != MemoryState.archived,
        Memory.content.ilike(f"%{search_query}%") if search_query else True
    )

    # Apply filters
    if app_id:
        query = query.filter(Memory.app_id == app_id)

    if from_date:
        from_datetime = datetime.fromtimestamp(from_date, tz=UTC)
        query = query.filter(Memory.created_at >= from_datetime)

    if to_date:
        to_datetime = datetime.fromtimestamp(to_date, tz=UTC)
        query = query.filter(Memory.created_at <= to_datetime)

    # Add joins for app and categories after filtering
    query = query.outerjoin(App, Memory.app_id == App.id)
    query = query.outerjoin(Memory.categories)

    # Apply category filter if provided
    if categories:
        category_list = [c.strip() for c in categories.split(",")]
        query = query.filter(Category.name.in_(category_list))

    # Apply sorting if specified
    if sort_column:
        sort_field = getattr(Memory, sort_column, None)
        if sort_field:
            query = query.order_by(sort_field.desc()) if sort_direction == "desc" else query.order_by(sort_field.asc())

    # Add eager loading for app and categories
    query = query.options(
        joinedload(Memory.app),
        joinedload(Memory.categories)
    ).distinct(Memory.id)

    # Get paginated results with transformer
    return sqlalchemy_paginate(
        query,
        params,
        transformer=lambda items: [
            MemoryResponse(
                id=memory.id,
                content=memory.content,
                created_at=memory.created_at,
                state=memory.state.value,
                app_id=memory.app_id,
                app_name=memory.app.name if memory.app else None,
                categories=[category.name for category in memory.categories],
                metadata_=memory.metadata_
            )
            for memory in items
            if check_memory_access_permissions(db, memory, app_id)
        ]
    )


# Get all categories
@router.get("/categories")
async def get_categories(
    principal: Principal = Depends(require_scopes(Scope.MEMORIES_READ)),
    db: Session = Depends(get_db)
):
    user = db.query(User).filter(User.user_id == principal.user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    # Get unique categories associated with the user's memories
    # Get all memories
    memories = db.query(Memory).filter(Memory.user_id == user.id, Memory.state != MemoryState.deleted, Memory.state != MemoryState.archived).all()
    # Get all categories from memories
    categories = [category for memory in memories for category in memory.categories]
    # Get unique categories
    unique_categories = list(set(categories))

    return {
        "categories": unique_categories,
        "total": len(unique_categories)
    }


class CreateMemoryRequest(BaseModel):
    text: str
    metadata: dict = {}
    infer: bool = True
    app: str = "openmemory"


# Create new memory
@router.post("/")
async def create_memory(
    request: CreateMemoryRequest,
    principal: Principal = Depends(require_scopes(Scope.MEMORIES_WRITE)),
    db: Session = Depends(get_db)
):
    user = db.query(User).filter(User.user_id == principal.user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    # Get or create app
    app_obj = db.query(App).filter(App.name == request.app,
                                   App.owner_id == user.id).first()
    if not app_obj:
        app_obj = App(name=request.app, owner_id=user.id)
        db.add(app_obj)
        db.commit()
        db.refresh(app_obj)

    # Check if app is active
    if not app_obj.is_active:
        raise HTTPException(status_code=403, detail=f"App {request.app} is currently paused on OpenMemory. Cannot create new memories.")

    # Log what we're about to do
    logging.info(f"Creating memory for user_id: {principal.user_id} with app: {request.app}")
    
    clean_text, axis_metadata = process_memory_input(request.text)

    # Try to get memory client safely
    try:
        memory_client = get_memory_client()
        if not memory_client:
            raise Exception("Memory client is not available")
    except Exception as client_error:
        logging.warning(f"Memory client unavailable: {client_error}. Creating memory in database only.")
        # Return a json response with the error
        return {
            "error": str(client_error)
        }

    # Try to save to Qdrant via memory_client
    try:
        combined_metadata = {
            "source_app": "openmemory",
            "mcp_client": request.app,
            **request.metadata,
            **axis_metadata,
        }
        qdrant_response = memory_client.add(
            clean_text,
            user_id=principal.user_id,  # Use JWT user_id to match search
            metadata=combined_metadata,
            infer=request.infer
        )
        
        # Log the response for debugging
        logging.info(f"Qdrant response: {qdrant_response}")
        
        # Process Qdrant response
        if isinstance(qdrant_response, dict) and 'results' in qdrant_response:
            created_memories = []
            
            for result in qdrant_response['results']:
                if result['event'] == 'ADD':
                    # Get the Qdrant-generated ID
                    memory_id = UUID(result['id'])
                    
                    # Check if memory already exists
                    existing_memory = db.query(Memory).filter(Memory.id == memory_id).first()
                    
                    if existing_memory:
                        # Update existing memory
                        existing_memory.state = MemoryState.active
                        existing_memory.content = result['memory']
                        existing_memory.metadata_ = combined_metadata
                        memory = existing_memory
                    else:
                        # Create memory with the EXACT SAME ID from Qdrant
                        memory = Memory(
                            id=memory_id,  # Use the same ID that Qdrant generated
                            user_id=user.id,
                            app_id=app_obj.id,
                            content=result['memory'],
                            metadata_=combined_metadata,
                            state=MemoryState.active
                        )
                        db.add(memory)
                    
                    # Create history entry
                    history = MemoryStatusHistory(
                        memory_id=memory_id,
                        changed_by=user.id,
                        old_state=MemoryState.deleted if existing_memory else MemoryState.deleted,
                        new_state=MemoryState.active
                    )
                    db.add(history)
                    
                    created_memories.append(memory)
            
            # Commit all changes at once
            if created_memories:
                db.commit()
                for memory in created_memories:
                    db.refresh(memory)
                
                # Return the first memory (for API compatibility)
                # but all memories are now saved to the database
                return created_memories[0]
    except Exception as qdrant_error:
        logging.warning(f"Qdrant operation failed: {qdrant_error}.")
        # Return a json response with the error
        return {
            "error": str(qdrant_error)
        }




# Get memory by ID
@router.get("/{memory_id}")
async def get_memory(
    memory_id: UUID,
    principal: Principal = Depends(require_scopes(Scope.MEMORIES_READ)),
    db: Session = Depends(get_db)
):
    memory = get_memory_or_404(db, memory_id)
    # Verify user owns this memory
    user = db.query(User).filter(User.user_id == principal.user_id).first()
    if not user or memory.user_id != user.id:
        raise HTTPException(status_code=404, detail="Memory not found")

    # Extract AXIS metadata from metadata_ dict for frontend
    metadata = {}
    if memory.metadata_:
        metadata = {
            "vault": memory.metadata_.get("vault"),
            "layer": memory.metadata_.get("layer"),
            "circuit": memory.metadata_.get("circuit"),
            "vector": memory.metadata_.get("vector"),
            "re": memory.metadata_.get("re"),
            "tags": memory.metadata_.get("tags"),
        }
        # Remove None values
        metadata = {k: v for k, v in metadata.items() if v is not None}

    return {
        "id": memory.id,
        "text": memory.content,
        "created_at": int(memory.created_at.timestamp()),
        "state": memory.state.value,
        "app_id": memory.app_id,
        "app_name": memory.app.name if memory.app else None,
        "categories": [category.name for category in memory.categories],
        "metadata_": memory.metadata_,
        "metadata": metadata if metadata else None
    }


class DeleteMemoriesRequest(BaseModel):
    memory_ids: List[UUID]

# Delete multiple memories
@router.delete("/")
async def delete_memories(
    request: DeleteMemoriesRequest,
    principal: Principal = Depends(require_scopes(Scope.MEMORIES_DELETE)),
    db: Session = Depends(get_db)
):
    user = db.query(User).filter(User.user_id == principal.user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    # Get memory client to delete from vector store
    try:
        memory_client = get_memory_client()
        if not memory_client:
            raise HTTPException(
                status_code=503,
                detail="Memory client is not available"
            )
    except HTTPException:
        raise
    except Exception as client_error:
        logging.error(f"Memory client initialization failed: {client_error}")
        raise HTTPException(
            status_code=503,
            detail=f"Memory service unavailable: {str(client_error)}"
        )

    # Delete from vector store then mark as deleted in database
    for memory_id in request.memory_ids:
        try:
            memory_client.delete(str(memory_id))
        except Exception as delete_error:
            logging.warning(f"Failed to delete memory {memory_id} from vector store: {delete_error}")

        update_memory_state(db, memory_id, MemoryState.deleted, user.id)

    return {"message": f"Successfully deleted {len(request.memory_ids)} memories"}


# Archive memories
@router.post("/actions/archive")
async def archive_memories(
    memory_ids: List[UUID],
    principal: Principal = Depends(require_scopes(Scope.MEMORIES_WRITE)),
    db: Session = Depends(get_db)
):
    user = db.query(User).filter(User.user_id == principal.user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    for memory_id in memory_ids:
        update_memory_state(db, memory_id, MemoryState.archived, user.id)
    return {"message": f"Successfully archived {len(memory_ids)} memories"}


class PauseMemoriesRequest(BaseModel):
    memory_ids: Optional[List[UUID]] = None
    category_ids: Optional[List[UUID]] = None
    app_id: Optional[UUID] = None
    all_for_app: bool = False
    global_pause: bool = False
    state: Optional[MemoryState] = None

# Pause access to memories
@router.post("/actions/pause")
async def pause_memories(
    request: PauseMemoriesRequest,
    principal: Principal = Depends(require_scopes(Scope.MEMORIES_WRITE)),
    db: Session = Depends(get_db)
):

    global_pause = request.global_pause
    all_for_app = request.all_for_app
    app_id = request.app_id
    memory_ids = request.memory_ids
    category_ids = request.category_ids
    state = request.state or MemoryState.paused

    user = db.query(User).filter(User.user_id == principal.user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    user_id = user.id
    
    if global_pause:
        # Pause all memories
        memories = db.query(Memory).filter(
            Memory.state != MemoryState.deleted,
            Memory.state != MemoryState.archived
        ).all()
        for memory in memories:
            update_memory_state(db, memory.id, state, user_id)
        return {"message": "Successfully paused all memories"}

    if app_id:
        # Pause all memories for an app
        memories = db.query(Memory).filter(
            Memory.app_id == app_id,
            Memory.user_id == user.id,
            Memory.state != MemoryState.deleted,
            Memory.state != MemoryState.archived
        ).all()
        for memory in memories:
            update_memory_state(db, memory.id, state, user_id)
        return {"message": f"Successfully paused all memories for app {app_id}"}
    
    if all_for_app and memory_ids:
        # Pause all memories for an app
        memories = db.query(Memory).filter(
            Memory.user_id == user.id,
            Memory.state != MemoryState.deleted,
            Memory.id.in_(memory_ids)
        ).all()
        for memory in memories:
            update_memory_state(db, memory.id, state, user_id)
        return {"message": "Successfully paused all memories"}

    if memory_ids:
        # Pause specific memories
        for memory_id in memory_ids:
            update_memory_state(db, memory_id, state, user_id)
        return {"message": f"Successfully paused {len(memory_ids)} memories"}

    if category_ids:
        # Pause memories by category
        memories = db.query(Memory).join(Memory.categories).filter(
            Category.id.in_(category_ids),
            Memory.state != MemoryState.deleted,
            Memory.state != MemoryState.archived
        ).all()
        for memory in memories:
            update_memory_state(db, memory.id, state, user_id)
        return {"message": f"Successfully paused memories in {len(category_ids)} categories"}

    raise HTTPException(status_code=400, detail="Invalid pause request parameters")


# Get memory access logs
@router.get("/{memory_id}/access-log")
async def get_memory_access_log(
    memory_id: UUID,
    principal: Principal = Depends(require_scopes(Scope.MEMORIES_READ)),
    page: int = Query(1, ge=1),
    page_size: int = Query(10, ge=1, le=100),
    db: Session = Depends(get_db)
):
    query = db.query(MemoryAccessLog).filter(MemoryAccessLog.memory_id == memory_id)
    total = query.count()
    logs = query.order_by(MemoryAccessLog.accessed_at.desc()).offset((page - 1) * page_size).limit(page_size).all()

    # Get app name
    for log in logs:
        app = db.query(App).filter(App.id == log.app_id).first()
        log.app_name = app.name if app else None

    return {
        "total": total,
        "page": page,
        "page_size": page_size,
        "logs": logs
    }


class UpdateMemoryRequest(BaseModel):
    memory_content: Optional[str] = None
    vault: Optional[str] = None
    layer: Optional[str] = None
    circuit: Optional[int] = None
    vector: Optional[str] = None
    entity: Optional[str] = None
    tags: Optional[dict] = None

    @classmethod
    def validate_vault(cls, v):
        if v is not None:
            valid_vaults = ["SOV", "WLT", "SIG", "FRC", "DIR", "FGP", "Q"]
            if v not in valid_vaults:
                raise ValueError(f"vault must be one of {valid_vaults}")
        return v

    @classmethod
    def validate_layer(cls, v):
        if v is not None:
            valid_layers = ["somatic", "emotional", "narrative", "cognitive",
                          "values", "identity", "relational", "goals",
                          "resources", "context", "temporal", "meta"]
            if v not in valid_layers:
                raise ValueError(f"layer must be one of {valid_layers}")
        return v

    @classmethod
    def validate_circuit(cls, v):
        if v is not None and (v < 1 or v > 8):
            raise ValueError("circuit must be between 1 and 8")
        return v

    @classmethod
    def validate_vector(cls, v):
        if v is not None:
            valid_vectors = ["say", "want", "do"]
            if v not in valid_vectors:
                raise ValueError(f"vector must be one of {valid_vectors}")
        return v

# Update a memory
@router.put("/{memory_id}")
async def update_memory(
    memory_id: UUID,
    request: UpdateMemoryRequest,
    principal: Principal = Depends(require_scopes(Scope.MEMORIES_WRITE)),
    db: Session = Depends(get_db)
):
    """Update memory content and/or metadata with best-effort sync to vector + graph stores."""
    from app.graph.graph_ops import (
        project_memory_to_graph,
        is_graph_enabled,
        is_mem0_graph_enabled,
        update_tag_edges_on_memory_add,
        delete_similarity_edges_for_memory,
        project_similarity_edges_for_memory,
        get_entities_for_memory_from_graph,
        refresh_co_mention_edges_for_entities,
    )
    from app.graph.entity_bridge import bridge_entities_to_om_graph
    from app.utils.structured_memory import validate_update_fields, apply_metadata_updates

    user = db.query(User).filter(User.user_id == principal.user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    memory = get_memory_or_404(db, memory_id)
    # Verify user owns this memory
    if memory.user_id != user.id:
        raise HTTPException(status_code=404, detail="Memory not found")

    sync_warnings: List[str] = []
    content_updated = request.memory_content is not None

    # --- Graph: collect old entities for precise co-mention refresh (best-effort) ---
    old_entities: List[str] = []
    if is_graph_enabled():
        try:
            old_entities = get_entities_for_memory_from_graph(
                memory_id=str(memory_id),
                user_id=principal.user_id,
            )
        except Exception as e:
            sync_warnings.append(f"graph_old_entities_failed:{e}")

    # --- SQL: apply validated metadata updates ---
    current_metadata = memory.metadata_ or {}
    try:
        validated_fields = validate_update_fields(
            vault=request.vault,
            layer=request.layer,
            circuit=request.circuit,
            vector=request.vector,
            entity=request.entity,
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    updated_metadata = apply_metadata_updates(
        current_metadata=current_metadata,
        validated_fields=validated_fields,
        add_tags=request.tags,
        remove_tags=None,
    )

    new_content = request.memory_content if request.memory_content is not None else memory.content

    # Update indexed fields if provided (keeps SQL filters fast + consistent)
    if "vault_full" in validated_fields:
        memory.vault = validated_fields["vault_full"]
    if "layer" in validated_fields:
        memory.layer = validated_fields["layer"]
    if "vector" in validated_fields:
        memory.axis_vector = validated_fields["vector"]

    memory.content = new_content
    memory.metadata_ = updated_metadata
    memory.updated_at = datetime.now(UTC)
    db.commit()
    db.refresh(memory)

    # --- Vector store: update embedding + payload (best-effort) ---
    vector_store_updated = False
    try:
        memory_client = get_memory_client()
        if memory_client:
            existing_payload = {}
            try:
                existing = memory_client.vector_store.get(vector_id=str(memory_id))
                if existing and getattr(existing, "payload", None):
                    existing_payload = dict(existing.payload)
            except Exception as e:
                sync_warnings.append(f"vector_store_get_failed:{e}")

            # Merge payload safely (preserve session ids + mem0 core keys)
            payload = dict(existing_payload)
            payload["user_id"] = principal.user_id
            payload["data"] = new_content
            payload["hash"] = hashlib.md5(new_content.encode()).hexdigest()
            payload["created_at"] = (
                existing_payload.get("created_at")
                or (memory.created_at.isoformat() if memory.created_at else None)
                or payload.get("created_at")
            )
            payload["updated_at"] = memory.updated_at.isoformat() if memory.updated_at else datetime.now(UTC).isoformat()

            reserved = {"user_id", "data", "hash", "created_at", "updated_at"}
            for k, v in (updated_metadata or {}).items():
                if k in reserved:
                    continue
                payload[k] = v

            if content_updated:
                embeddings = memory_client.embedding_model.embed(new_content, "update")
                memory_client.vector_store.update(
                    vector_id=str(memory_id),
                    vector=embeddings,
                    payload=payload,
                )
                vector_store_updated = True
            else:
                # Metadata-only update (Qdrant): update payload without touching vectors.
                # Other vector stores may not support payload-only updates here.
                vs = getattr(memory_client, "vector_store", None)
                if vs and hasattr(vs, "client") and hasattr(vs, "collection_name"):
                    vs.client.set_payload(
                        collection_name=vs.collection_name,
                        payload=payload,
                        points=[str(memory_id)],
                    )
                    vector_store_updated = True
                else:
                    sync_warnings.append("vector_store_payload_only_update_not_supported")
        else:
            sync_warnings.append("memory_client_unavailable")
    except Exception as e:
        sync_warnings.append(f"vector_store_update_failed:{e}")

    # --- Neo4j (OM_*): re-project + refresh derived edges (best-effort) ---
    graph_projected = False
    entity_bridge_ran = False
    co_mentions_refreshed = False
    similarity_refreshed = False
    tag_edges_updated = False

    if is_graph_enabled():
        try:
            project_memory_to_graph(
                memory_id=str(memory_id),
                user_id=principal.user_id,
                content=new_content,
                metadata=updated_metadata,
                created_at=memory.created_at.isoformat() if memory.created_at else None,
                updated_at=memory.updated_at.isoformat() if memory.updated_at else None,
                state=memory.state.value if memory.state else "active",
            )
            graph_projected = True
        except Exception as e:
            sync_warnings.append(f"neo4j_projection_failed:{e}")

        # Bridge multi-entity extraction (uses Mem0 Graph extraction; does not require vector-store changes)
        if is_mem0_graph_enabled():
            try:
                bridge_entities_to_om_graph(
                    memory_id=str(memory_id),
                    user_id=principal.user_id,
                    content=new_content,
                    existing_entity=updated_metadata.get("re"),
                )
                entity_bridge_ran = True
            except Exception as e:
                sync_warnings.append(f"entity_bridge_failed:{e}")

        # Update tag co-occurrence edges only if tags were part of the request
        if request.tags is not None:
            try:
                update_tag_edges_on_memory_add(str(memory_id), principal.user_id)
                tag_edges_updated = True
            except Exception as e:
                sync_warnings.append(f"tag_edges_update_failed:{e}")

        # Similarity edges depend on embeddings; only refresh when content changed.
        if content_updated:
            try:
                delete_similarity_edges_for_memory(str(memory_id), principal.user_id)
                project_similarity_edges_for_memory(str(memory_id), principal.user_id)
                similarity_refreshed = True
            except Exception as e:
                sync_warnings.append(f"similarity_refresh_failed:{e}")

        # Co-mention refresh: recompute counts for affected entity pairs (old ∪ new)
        try:
            new_entities = get_entities_for_memory_from_graph(
                memory_id=str(memory_id),
                user_id=principal.user_id,
            )
            affected_entities = sorted({*(old_entities or []), *(new_entities or [])})
            if affected_entities:
                refresh_co_mention_edges_for_entities(
                    user_id=principal.user_id,
                    entity_names=affected_entities,
                )
                co_mentions_refreshed = True
        except Exception as e:
            sync_warnings.append(f"co_mentions_refresh_failed:{e}")

    return {
        "message": "Memory updated successfully",
        "id": str(memory_id),
        "sync": {
            "vector_store_updated": vector_store_updated,
            "graph_projected": graph_projected,
            "entity_bridge_ran": entity_bridge_ran,
            "tag_edges_updated": tag_edges_updated,
            "similarity_refreshed": similarity_refreshed,
            "co_mentions_refreshed": co_mentions_refreshed,
        },
        "warnings": sync_warnings,
    }

class FilterMemoriesRequest(BaseModel):
    page: int = 1
    size: int = 10
    search_query: Optional[str] = None
    app_ids: Optional[List[UUID]] = None
    category_ids: Optional[List[UUID]] = None
    sort_column: Optional[str] = None
    sort_direction: Optional[str] = None
    from_date: Optional[int] = None
    to_date: Optional[int] = None
    show_archived: Optional[bool] = False
    vaults: Optional[List[str]] = None
    layers: Optional[List[str]] = None

@router.post("/filter", response_model=Page[MemoryResponse])
async def filter_memories(
    request: FilterMemoriesRequest,
    principal: Principal = Depends(require_scopes(Scope.MEMORIES_READ)),
    db: Session = Depends(get_db)
):
    user = db.query(User).filter(User.user_id == principal.user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    # Build base query
    query = db.query(Memory).filter(
        Memory.user_id == user.id,
        Memory.state != MemoryState.deleted,
    )

    # Filter archived memories based on show_archived parameter
    if not request.show_archived:
        query = query.filter(Memory.state != MemoryState.archived)

    # Apply search filter
    if request.search_query:
        query = query.filter(Memory.content.ilike(f"%{request.search_query}%"))

    # Apply app filter
    if request.app_ids:
        query = query.filter(Memory.app_id.in_(request.app_ids))

    # Apply vault/layer filters from indexed columns (database-agnostic)
    # Note: Memory model has vault/layer as indexed columns for fast queries
    if request.vaults:
        query = query.filter(Memory.vault.in_(request.vaults))
    if request.layers:
        query = query.filter(Memory.layer.in_(request.layers))

    # Add joins for app and categories
    query = query.outerjoin(App, Memory.app_id == App.id)

    # Apply category filter
    if request.category_ids:
        query = query.join(Memory.categories).filter(Category.id.in_(request.category_ids))
    else:
        query = query.outerjoin(Memory.categories)

    # Apply date filters
    if request.from_date:
        from_datetime = datetime.fromtimestamp(request.from_date, tz=UTC)
        query = query.filter(Memory.created_at >= from_datetime)

    if request.to_date:
        to_datetime = datetime.fromtimestamp(request.to_date, tz=UTC)
        query = query.filter(Memory.created_at <= to_datetime)

    # Apply sorting
    if request.sort_column and request.sort_direction:
        sort_direction = request.sort_direction.lower()
        if sort_direction not in ['asc', 'desc']:
            raise HTTPException(status_code=400, detail="Invalid sort direction")

        sort_mapping = {
            'memory': Memory.content,
            'app_name': App.name,
            'created_at': Memory.created_at
        }

        if request.sort_column not in sort_mapping:
            raise HTTPException(status_code=400, detail="Invalid sort column")

        sort_field = sort_mapping[request.sort_column]
        if sort_direction == 'desc':
            query = query.order_by(sort_field.desc())
        else:
            query = query.order_by(sort_field.asc())
    else:
        # Default sorting
        query = query.order_by(Memory.created_at.desc())

    # Add eager loading for categories and make the query distinct
    query = query.options(
        joinedload(Memory.categories)
    ).distinct(Memory.id)

    # Use fastapi-pagination's paginate function
    return sqlalchemy_paginate(
        query,
        Params(page=request.page, size=request.size),
        transformer=lambda items: [
            MemoryResponse(
                id=memory.id,
                content=memory.content,
                created_at=memory.created_at,
                state=memory.state.value,
                app_id=memory.app_id,
                app_name=memory.app.name if memory.app else None,
                categories=[category.name for category in memory.categories],
                metadata_=memory.metadata_
            )
            for memory in items
        ]
    )


@router.get("/{memory_id}/related", response_model=Page[MemoryResponse])
async def get_related_memories(
    memory_id: UUID,
    principal: Principal = Depends(require_scopes(Scope.MEMORIES_READ)),
    params: Params = Depends(),
    db: Session = Depends(get_db)
):
    # Validate user from JWT
    user = db.query(User).filter(User.user_id == principal.user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    # Get the source memory
    memory = get_memory_or_404(db, memory_id)
    # Verify ownership
    if memory.user_id != user.id:
        raise HTTPException(status_code=404, detail="Memory not found")
    
    # Extract category IDs from the source memory
    category_ids = [category.id for category in memory.categories]
    
    if not category_ids:
        return Page.create([], total=0, params=params)
    
    # Build query for related memories
    query = db.query(Memory).distinct(Memory.id).filter(
        Memory.user_id == user.id,
        Memory.id != memory_id,
        Memory.state != MemoryState.deleted
    ).join(Memory.categories).filter(
        Category.id.in_(category_ids)
    ).options(
        joinedload(Memory.categories),
        joinedload(Memory.app)
    ).order_by(
        func.count(Category.id).desc(),
        Memory.created_at.desc()
    ).group_by(Memory.id)
    
    # ⚡ Force page size to be 5
    params = Params(page=params.page, size=5)
    
    return sqlalchemy_paginate(
        query,
        params,
        transformer=lambda items: [
            MemoryResponse(
                id=memory.id,
                content=memory.content,
                created_at=memory.created_at,
                state=memory.state.value,
                app_id=memory.app_id,
                app_name=memory.app.name if memory.app else None,
                categories=[category.name for category in memory.categories],
                metadata_=memory.metadata_
            )
            for memory in items
        ]
    )


# =============================================================================
# Graph-based Memory Endpoints
# =============================================================================

@router.get("/{memory_id}/graph")
async def get_memory_graph_context(
    memory_id: UUID,
    principal: Principal = Depends(require_scopes(Scope.MEMORIES_READ)),
    db: Session = Depends(get_db),
):
    """Get graph context for a memory: similar memories, subgraph, etc."""
    from app.graph.graph_ops import (
        get_similar_memories_from_graph,
        get_memory_subgraph_from_graph,
    )

    # Verify memory exists and belongs to user
    user = db.query(User).filter(User.user_id == principal.user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    memory = db.query(Memory).filter(
        Memory.id == memory_id,
        Memory.user_id == user.id
    ).first()

    if not memory:
        raise HTTPException(status_code=404, detail="Memory not found")

    # Get similar memories via OM_SIMILAR edges
    similar = get_similar_memories_from_graph(
        memory_id=str(memory_id),
        user_id=principal.user_id,
        min_score=0.5,
        limit=10
    )

    # Get subgraph (entities, dimensions, related memories)
    subgraph = get_memory_subgraph_from_graph(
        memory_id=str(memory_id),
        user_id=principal.user_id,
        depth=2,
        related_limit=20
    )

    return {
        "memory_id": str(memory_id),
        "similar_memories": similar,
        "subgraph": subgraph,
    }


@router.get("/{memory_id}/similar")
async def get_similar_memories(
    memory_id: UUID,
    principal: Principal = Depends(require_scopes(Scope.MEMORIES_READ)),
    min_score: float = Query(default=0.5, ge=0.0, le=1.0),
    limit: int = Query(default=10, ge=1, le=50),
    db: Session = Depends(get_db),
):
    """Get semantically similar memories via pre-computed OM_SIMILAR edges."""
    from app.graph.graph_ops import get_similar_memories_from_graph

    user = db.query(User).filter(User.user_id == principal.user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    memory = db.query(Memory).filter(
        Memory.id == memory_id,
        Memory.user_id == user.id
    ).first()

    if not memory:
        raise HTTPException(status_code=404, detail="Memory not found")

    similar = get_similar_memories_from_graph(
        memory_id=str(memory_id),
        user_id=principal.user_id,
        min_score=min_score,
        limit=limit
    )

    return {"memory_id": str(memory_id), "similar_memories": similar}
