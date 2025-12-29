import { useState, useCallback } from 'react';
import api from '@/lib/api';
import {
  Memory,
  MemoryMetadata,
  Client,
  Category,
  MemoryUpdateRequest,
  SimilarMemory,
  MemorySubgraph,
  StructuredCategory,
  MemoryScope,
  ArtifactType,
  SourceType,
} from '@/components/types';
import { useDispatch, useSelector } from 'react-redux';
import { AppDispatch, RootState } from '@/store/store';
import { setAccessLogs, setMemoriesSuccess, setSelectedMemory, setRelatedMemories } from '@/store/memoriesSlice';

// Define the new simplified memory type
export interface SimpleMemory {
  id: string;
  text: string;
  created_at: string;
  state: string;
  categories: string[];
  app_name: string;
  metadata?: {
    category?: StructuredCategory;
    scope?: MemoryScope;
    artifact_type?: ArtifactType;
    artifact_ref?: string;
    entity?: string;
    source?: SourceType;
    evidence?: string[];
    tags?: Record<string, any>;
  };
}

// Define the shape of the API response item
interface ApiMemoryItem {
  id: string;
  content: string;
  created_at: string;
  state: string;
  app_id: string;
  categories: string[];
  metadata_?: Record<string, any>;
  app_name: string;
}

// Define the shape of the API response
interface ApiResponse {
  items: ApiMemoryItem[];
  total: number;
  page: number;
  size: number;
  pages: number;
}

interface AccessLogEntry {
  id: string;
  app_name: string;
  accessed_at: string;
}

interface AccessLogResponse {
  total: number;
  page: number;
  page_size: number;
  logs: AccessLogEntry[];
}

interface RelatedMemoryItem {
  id: string;
  content: string;
  created_at: number;
  state: string;
  app_id: string;
  app_name: string;
  categories: string[];
  metadata_: Record<string, any>;
}

interface RelatedMemoriesResponse {
  items: RelatedMemoryItem[];
  total: number;
  page: number;
  size: number;
  pages: number;
}

interface SimilarMemoriesResponse {
  memory_id: string;
  similar_memories: SimilarMemory[];
}

interface MemoryGraphContextResponse {
  memory_id: string;
  similar_memories: SimilarMemory[];
  subgraph: MemorySubgraph;
}

interface UseMemoriesApiReturn {
  fetchMemories: (
    query?: string,
    page?: number,
    size?: number,
    filters?: {
      apps?: string[];
      categories?: string[];
      sortColumn?: string;
      sortDirection?: 'asc' | 'desc';
      showArchived?: boolean;
      scopes?: string[];
      artifactTypes?: string[];
      entities?: string[];
      sources?: string[];
    }
  ) => Promise<{ memories: Memory[]; total: number; pages: number }>;
  fetchMemoryById: (memoryId: string) => Promise<void>;
  fetchAccessLogs: (memoryId: string, page?: number, pageSize?: number) => Promise<void>;
  fetchRelatedMemories: (memoryId: string) => Promise<void>;
  createMemory: (text: string) => Promise<void>;
  deleteMemories: (memoryIds: string[]) => Promise<void>;
  updateMemory: (memoryId: string, content: string) => Promise<void>;
  updateMemoryWithMetadata: (
    memoryId: string,
    updates: {
      content?: string;
      category?: StructuredCategory;
      scope?: MemoryScope;
      artifact_type?: ArtifactType;
      artifact_ref?: string;
      entity?: string;
      source?: SourceType;
      evidence?: string[];
      tags?: Record<string, any>;
    }
  ) => Promise<void>;
  updateMemoryState: (memoryIds: string[], state: string) => Promise<void>;
  fetchSimilarMemories: (memoryId: string, minScore?: number, limit?: number) => Promise<SimilarMemory[]>;
  fetchMemoryGraphContext: (memoryId: string) => Promise<MemoryGraphContextResponse | null>;
  isLoading: boolean;
  error: string | null;
  hasUpdates: number;
  memories: Memory[];
  selectedMemory: SimpleMemory | null;
}

export const useMemoriesApi = (): UseMemoriesApiReturn => {
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  const [hasUpdates, setHasUpdates] = useState<number>(0);
  const dispatch = useDispatch<AppDispatch>();
  const user_id = useSelector((state: RootState) => state.profile.userId);
  const memories = useSelector((state: RootState) => state.memories.memories);
  const selectedMemory = useSelector((state: RootState) => state.memories.selectedMemory);

  const URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8765";

  const fetchMemories = useCallback(async (
    query?: string,
    page: number = 1,
    size: number = 10,
    filters?: {
      apps?: string[];
      categories?: string[];
      sortColumn?: string;
      sortDirection?: 'asc' | 'desc';
      showArchived?: boolean;
      scopes?: string[];
      artifactTypes?: string[];
      entities?: string[];
      sources?: string[];
    }
  ): Promise<{ memories: Memory[], total: number, pages: number }> => {
    setIsLoading(true);
    setError(null);
    try {
      const response = await api.post<ApiResponse>(
        `${URL}/api/v1/memories/filter`,
        {
          user_id: user_id,
          page: page,
          size: size,
          search_query: query,
          app_ids: filters?.apps,
          sort_column: filters?.sortColumn?.toLowerCase(),
          sort_direction: filters?.sortDirection,
          show_archived: filters?.showArchived,
          memory_categories: filters?.categories,
          scopes: filters?.scopes,
          artifact_types: filters?.artifactTypes,
          entities: filters?.entities,
          sources: filters?.sources
        }
      );

      const adaptedMemories: Memory[] = response.data.items.map((item: ApiMemoryItem) => ({
        id: item.id,
        memory: item.content,
        created_at: new Date(item.created_at).getTime(),
        state: item.state as "active" | "paused" | "archived" | "deleted",
        metadata: (item.metadata_ || {}) as MemoryMetadata,
        categories: item.categories as Category[],
        client: 'api' as const,
        app_name: item.app_name
      }));
      setIsLoading(false);
      dispatch(setMemoriesSuccess(adaptedMemories));
      return {
        memories: adaptedMemories,
        total: response.data.total,
        pages: response.data.pages
      };
    } catch (err: any) {
      const errorMessage = err.message || 'Failed to fetch memories';
      setError(errorMessage);
      setIsLoading(false);
      throw new Error(errorMessage);
    }
  }, [user_id, dispatch]);

  const createMemory = async (text: string): Promise<void> => {
    try {
      const memoryData = {
        user_id: user_id,
        text: text,
        infer: false,
        app: "openmemory",
      }
      await api.post<ApiMemoryItem>(`${URL}/api/v1/memories/`, memoryData);
    } catch (err: any) {
      const errorMessage = err.message || 'Failed to create memory';
      setError(errorMessage);
      setIsLoading(false);
      throw new Error(errorMessage);
    }
  };

  const deleteMemories = async (memory_ids: string[]) => {
    try {
      await api.delete(`${URL}/api/v1/memories/`, {
        data: { memory_ids, user_id }
      });
      dispatch(setMemoriesSuccess(memories.filter((memory: Memory) => !memory_ids.includes(memory.id))));
    } catch (err: any) {
      const errorMessage = err.message || 'Failed to delete memories';
      setError(errorMessage);
      setIsLoading(false);
      throw new Error(errorMessage);
    }
  };

  const fetchMemoryById = async (memoryId: string): Promise<void> => {
    if (memoryId === "") {
      return;
    }
    setIsLoading(true);
    setError(null);
    try {
      const response = await api.get<SimpleMemory>(
        `${URL}/api/v1/memories/${memoryId}?user_id=${user_id}`
      );
      setIsLoading(false);
      dispatch(setSelectedMemory(response.data));
    } catch (err: any) {
      const errorMessage = err.message || 'Failed to fetch memory';
      setError(errorMessage);
      setIsLoading(false);
      throw new Error(errorMessage);
    }
  };

  const fetchAccessLogs = async (memoryId: string, page: number = 1, pageSize: number = 10): Promise<void> => {
    if (memoryId === "") {
      return;
    }
    setIsLoading(true);
    setError(null);
    try {
      const response = await api.get<AccessLogResponse>(
        `${URL}/api/v1/memories/${memoryId}/access-log?page=${page}&page_size=${pageSize}`
      );
      setIsLoading(false);
      dispatch(setAccessLogs(response.data.logs));
    } catch (err: any) {
      const errorMessage = err.message || 'Failed to fetch access logs';
      setError(errorMessage);
      setIsLoading(false);
      throw new Error(errorMessage);
    }
  };

  const fetchRelatedMemories = async (memoryId: string): Promise<void> => {
    if (memoryId === "") {
      return;
    }
    setIsLoading(true);
    setError(null);
    try {
      const response = await api.get<RelatedMemoriesResponse>(
        `${URL}/api/v1/memories/${memoryId}/related?user_id=${user_id}`
      );

      const adaptedMemories: Memory[] = response.data.items.map((item: RelatedMemoryItem) => ({
        id: item.id,
        memory: item.content,
        created_at: item.created_at,
        state: item.state as "active" | "paused" | "archived" | "deleted",
        metadata: item.metadata_,
        categories: item.categories as Category[],
        client: 'api',
        app_name: item.app_name
      }));

      setIsLoading(false);
      dispatch(setRelatedMemories(adaptedMemories));
    } catch (err: any) {
      const errorMessage = err.message || 'Failed to fetch related memories';
      setError(errorMessage);
      setIsLoading(false);
      throw new Error(errorMessage);
    }
  };

  const updateMemory = async (memoryId: string, content: string): Promise<void> => {
    if (memoryId === "") {
      return;
    }
    setIsLoading(true);
    setError(null);
    try {
      await api.put(`${URL}/api/v1/memories/${memoryId}`, {
        memory_id: memoryId,
        memory_content: content,
        user_id: user_id
      });
      setIsLoading(false);
      setHasUpdates(hasUpdates + 1);
    } catch (err: any) {
      const errorMessage = err.message || 'Failed to update memory';
      setError(errorMessage);
      setIsLoading(false);
      throw new Error(errorMessage);
    }
  };

  const updateMemoryWithMetadata = async (
    memoryId: string,
    updates: {
      content?: string;
      category?: StructuredCategory;
      scope?: MemoryScope;
      artifact_type?: ArtifactType;
      artifact_ref?: string;
      entity?: string;
      source?: SourceType;
      evidence?: string[];
      tags?: Record<string, any>;
    }
  ): Promise<void> => {
    if (memoryId === "") {
      return;
    }
    setIsLoading(true);
    setError(null);
    try {
      const payload: MemoryUpdateRequest = {
        user_id: user_id,
      };
      if (updates.content !== undefined) payload.memory_content = updates.content;
      if (updates.category !== undefined) payload.category = updates.category;
      if (updates.scope !== undefined) payload.scope = updates.scope;
      if (updates.artifact_type !== undefined) payload.artifact_type = updates.artifact_type;
      if (updates.artifact_ref !== undefined) payload.artifact_ref = updates.artifact_ref;
      if (updates.entity !== undefined) payload.entity = updates.entity;
      if (updates.source !== undefined) payload.source = updates.source;
      if (updates.evidence !== undefined) payload.evidence = updates.evidence;
      if (updates.tags !== undefined) payload.tags = updates.tags;

      await api.put(`${URL}/api/v1/memories/${memoryId}`, payload);
      setIsLoading(false);
      setHasUpdates(hasUpdates + 1);
    } catch (err: any) {
      const errorMessage = err.message || 'Failed to update memory';
      setError(errorMessage);
      setIsLoading(false);
      throw new Error(errorMessage);
    }
  };

  const fetchSimilarMemories = async (
    memoryId: string,
    minScore: number = 0.5,
    limit: number = 10
  ): Promise<SimilarMemory[]> => {
    if (memoryId === "") {
      return [];
    }
    setIsLoading(true);
    setError(null);
    try {
      const response = await api.get<SimilarMemoriesResponse>(
        `${URL}/api/v1/memories/${memoryId}/similar?user_id=${user_id}&min_score=${minScore}&limit=${limit}`
      );
      setIsLoading(false);
      return response.data.similar_memories || [];
    } catch (err: any) {
      const errorMessage = err.message || 'Failed to fetch similar memories';
      setError(errorMessage);
      setIsLoading(false);
      return [];
    }
  };

  const fetchMemoryGraphContext = async (
    memoryId: string
  ): Promise<MemoryGraphContextResponse | null> => {
    if (memoryId === "") {
      return null;
    }
    setIsLoading(true);
    setError(null);
    try {
      const response = await api.get<MemoryGraphContextResponse>(
        `${URL}/api/v1/memories/${memoryId}/graph?user_id=${user_id}`
      );
      setIsLoading(false);
      return response.data;
    } catch (err: any) {
      const errorMessage = err.message || 'Failed to fetch memory graph context';
      setError(errorMessage);
      setIsLoading(false);
      return null;
    }
  };

  const updateMemoryState = async (memoryIds: string[], state: string): Promise<void> => {
    if (memoryIds.length === 0) {
      return;
    }
    setIsLoading(true);
    setError(null);
    try {
      await api.post(`${URL}/api/v1/memories/actions/pause`, {
        memory_ids: memoryIds,
        all_for_app: true,
        state: state,
        user_id: user_id
      });
      dispatch(setMemoriesSuccess(memories.map((memory: Memory) => {
        if (memoryIds.includes(memory.id)) {
          return { ...memory, state: state as "active" | "paused" | "archived" | "deleted" };
        }
        return memory;
      })));

      // If archive, delete the memory
      if (state === "archived") {
        dispatch(setMemoriesSuccess(memories.filter((memory: Memory) => !memoryIds.includes(memory.id))));
      }

      // if selected memory, update it
      if (selectedMemory?.id && memoryIds.includes(selectedMemory.id)) {
        dispatch(setSelectedMemory({ ...selectedMemory, state: state as "active" | "paused" | "archived" | "deleted" }));
      }

      setIsLoading(false);
      setHasUpdates(hasUpdates + 1);
    } catch (err: any) {
      const errorMessage = err.message || 'Failed to update memory state';
      setError(errorMessage);
      setIsLoading(false);
      throw new Error(errorMessage);
    }
  };

  return {
    fetchMemories,
    fetchMemoryById,
    fetchAccessLogs,
    fetchRelatedMemories,
    createMemory,
    deleteMemories,
    updateMemory,
    updateMemoryWithMetadata,
    updateMemoryState,
    fetchSimilarMemories,
    fetchMemoryGraphContext,
    isLoading,
    error,
    hasUpdates,
    memories,
    selectedMemory
  };
};
