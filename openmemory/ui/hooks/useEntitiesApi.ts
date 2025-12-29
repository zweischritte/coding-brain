import { useState, useCallback } from 'react';
import api from '@/lib/api';
import {
  Entity,
  EntityNetwork,
  EntityRelation,
} from '@/components/types';

interface EntityListResponse {
  entities: { key: string; count: number }[];
  total: number;
}

interface EntityDetailResponse {
  name: string;
  memory_count: number;
  network: EntityNetwork | null;
  relations: EntityRelation[];
}

interface EntityMemoriesResponse {
  entity: string;
  memories: {
    id: string;
    content: string;
    category?: string;
    scope?: string;
    artifact_type?: string;
    artifact_ref?: string;
    matchedEntities: number;
    entityNames: string[];
  }[];
}

interface EntityRelationsResponse {
  entity: string;
  relations: EntityRelation[];
}

interface PathResponse {
  entity_a: string;
  entity_b: string;
  nodes: any[];
  relationships: any[];
}

interface CentralityEntity {
  entity: string;
  connections: number;
  mentionCount: number;
}

interface DuplicateGroup {
  canonical: string;
  variants: { name: string; memories: number }[];
  total_memories: number;
}

interface UseEntitiesApiReturn {
  listEntities: (limit?: number, minMemories?: number) => Promise<Entity[]>;
  getEntity: (entityName: string) => Promise<EntityDetailResponse | null>;
  getEntityNetwork: (entityName: string, minCount?: number, limit?: number) => Promise<EntityNetwork | null>;
  getEntityRelations: (
    entityName: string,
    params?: { relationTypes?: string; category?: string; direction?: string; limit?: number }
  ) => Promise<EntityRelation[]>;
  getEntityMemories: (entityName: string, limit?: number) => Promise<any[]>;
  getCentrality: (limit?: number) => Promise<CentralityEntity[]>;
  findPath: (entityA: string, entityB: string, maxHops?: number) => Promise<PathResponse | null>;
  findDuplicates: () => Promise<DuplicateGroup[]>;
  mergeDuplicates: (
    canonical: string,
    variants: string[],
    dryRun?: boolean
  ) => Promise<any>;
  isLoading: boolean;
  error: string | null;
}

export const useEntitiesApi = (): UseEntitiesApiReturn => {
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  const URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8765";

  const listEntities = useCallback(async (
    limit: number = 50,
    minMemories: number = 1
  ): Promise<Entity[]> => {
    setIsLoading(true);
    setError(null);
    try {
      const response = await api.get<EntityListResponse>(
        `${URL}/api/v1/entities?limit=${limit}&min_memories=${minMemories}`
      );
      setIsLoading(false);
      return response.data.entities.map(e => ({
        name: e.key,
        memory_count: e.count,
      }));
    } catch (err: any) {
      const errorMessage = err.message || 'Failed to fetch entities';
      setError(errorMessage);
      setIsLoading(false);
      return [];
    }
  }, [URL]);

  const getEntity = useCallback(async (
    entityName: string
  ): Promise<EntityDetailResponse | null> => {
    setIsLoading(true);
    setError(null);
    try {
      const response = await api.get<EntityDetailResponse>(
        `${URL}/api/v1/entities/${encodeURIComponent(entityName)}`
      );
      setIsLoading(false);
      return response.data;
    } catch (err: any) {
      const errorMessage = err.message || 'Failed to fetch entity';
      setError(errorMessage);
      setIsLoading(false);
      return null;
    }
  }, [URL]);

  const getEntityNetwork = useCallback(async (
    entityName: string,
    minCount: number = 1,
    limit: number = 50
  ): Promise<EntityNetwork | null> => {
    setIsLoading(true);
    setError(null);
    try {
      const response = await api.get<EntityNetwork>(
        `${URL}/api/v1/entities/${encodeURIComponent(entityName)}/network?min_count=${minCount}&limit=${limit}`
      );
      setIsLoading(false);
      return response.data;
    } catch (err: any) {
      const errorMessage = err.message || 'Failed to fetch entity network';
      setError(errorMessage);
      setIsLoading(false);
      return null;
    }
  }, [URL]);

  const getEntityRelations = useCallback(async (
    entityName: string,
    params?: { relationTypes?: string; category?: string; direction?: string; limit?: number }
  ): Promise<EntityRelation[]> => {
    setIsLoading(true);
    setError(null);
    try {
      const queryParams = new URLSearchParams();
      if (params?.relationTypes) queryParams.append('relation_types', params.relationTypes);
      if (params?.category) queryParams.append('category', params.category);
      if (params?.direction) queryParams.append('direction', params.direction);
      if (params?.limit) queryParams.append('limit', params.limit.toString());

      const response = await api.get<EntityRelationsResponse>(
        `${URL}/api/v1/entities/${encodeURIComponent(entityName)}/relations?${queryParams.toString()}`
      );
      setIsLoading(false);
      return response.data.relations || [];
    } catch (err: any) {
      const errorMessage = err.message || 'Failed to fetch entity relations';
      setError(errorMessage);
      setIsLoading(false);
      return [];
    }
  }, [URL]);

  const getEntityMemories = useCallback(async (
    entityName: string,
    limit: number = 20
  ): Promise<any[]> => {
    setIsLoading(true);
    setError(null);
    try {
      const response = await api.get<EntityMemoriesResponse>(
        `${URL}/api/v1/entities/${encodeURIComponent(entityName)}/memories?limit=${limit}`
      );
      setIsLoading(false);
      return response.data.memories || [];
    } catch (err: any) {
      const errorMessage = err.message || 'Failed to fetch entity memories';
      setError(errorMessage);
      setIsLoading(false);
      return [];
    }
  }, [URL]);

  const getCentrality = useCallback(async (
    limit: number = 20
  ): Promise<CentralityEntity[]> => {
    setIsLoading(true);
    setError(null);
    try {
      const response = await api.get<{ entities: CentralityEntity[] }>(
        `${URL}/api/v1/entities/analytics/centrality?limit=${limit}`
      );
      setIsLoading(false);
      return response.data.entities || [];
    } catch (err: any) {
      const errorMessage = err.message || 'Failed to fetch centrality';
      setError(errorMessage);
      setIsLoading(false);
      return [];
    }
  }, [URL]);

  const findPath = useCallback(async (
    entityA: string,
    entityB: string,
    maxHops: number = 6
  ): Promise<PathResponse | null> => {
    setIsLoading(true);
    setError(null);
    try {
      const response = await api.get<PathResponse>(
        `${URL}/api/v1/entities/path/${encodeURIComponent(entityA)}/${encodeURIComponent(entityB)}?max_hops=${maxHops}`
      );
      setIsLoading(false);
      return response.data;
    } catch (err: any) {
      const errorMessage = err.message || 'Failed to find path';
      setError(errorMessage);
      setIsLoading(false);
      return null;
    }
  }, [URL]);

  const findDuplicates = useCallback(async (): Promise<DuplicateGroup[]> => {
    setIsLoading(true);
    setError(null);
    try {
      const response = await api.get<{ duplicates: DuplicateGroup[] }>(
        `${URL}/api/v1/entities/normalization/duplicates`
      );
      setIsLoading(false);
      return response.data.duplicates || [];
    } catch (err: any) {
      const errorMessage = err.message || 'Failed to find duplicates';
      setError(errorMessage);
      setIsLoading(false);
      return [];
    }
  }, [URL]);

  const mergeDuplicates = useCallback(async (
    canonical: string,
    variants: string[],
    dryRun: boolean = true
  ): Promise<any> => {
    setIsLoading(true);
    setError(null);
    try {
      const response = await api.post(
        `${URL}/api/v1/entities/normalization/merge`,
        {
          canonical,
          variants,
          dry_run: dryRun,
        }
      );
      setIsLoading(false);
      return response.data;
    } catch (err: any) {
      const errorMessage = err.message || 'Failed to merge duplicates';
      setError(errorMessage);
      setIsLoading(false);
      throw new Error(errorMessage);
    }
  }, [URL]);

  return {
    listEntities,
    getEntity,
    getEntityNetwork,
    getEntityRelations,
    getEntityMemories,
    getCentrality,
    findPath,
    findDuplicates,
    mergeDuplicates,
    isLoading,
    error,
  };
};
