import { useState, useCallback } from 'react';
import axios from 'axios';
import { useSelector } from 'react-redux';
import { RootState } from '@/store/store';
import {
  GraphStats,
  AggregationBucket,
  TagPair,
} from '@/components/types';

interface GraphHealthResponse {
  neo4j_metadata_projection: boolean;
  neo4j_similarity_edges: boolean;
  neo4j_gds: boolean;
  mem0_graph_memory: boolean;
}

interface AggregateResponse {
  dimension: string;
  buckets: AggregationBucket[];
}

interface TagCooccurrenceResponse {
  pairs: TagPair[];
}

interface TimelineEvent {
  id: string;
  name: string;
  event_type: string;
  start_date?: string;
  end_date?: string;
  description?: string;
  entity?: string;
  memory_ids?: string[];
}

interface TimelineResponse {
  events: TimelineEvent[];
}

interface FullTextSearchResult {
  id: string;
  content: string;
  category?: string;
  scope?: string;
  artifact_type?: string;
  artifact_ref?: string;
  entity?: string;
  createdAt?: string;
  searchScore: number;
}

interface UseGraphApiReturn {
  getStats: () => Promise<GraphStats>;
  getHealth: () => Promise<GraphHealthResponse>;
  aggregate: (dimension: string, limit?: number) => Promise<AggregationBucket[]>;
  getTagCooccurrence: (limit?: number, minCount?: number) => Promise<TagPair[]>;
  getRelatedTags: (tagKey: string, minCount?: number, limit?: number) => Promise<any[]>;
  getTimeline: (params?: {
    entityName?: string;
    eventTypes?: string;
    startYear?: number;
    endYear?: number;
    limit?: number;
  }) => Promise<TimelineEvent[]>;
  searchMemories: (query: string, limit?: number) => Promise<FullTextSearchResult[]>;
  searchEntities: (query: string, limit?: number) => Promise<any[]>;
  isLoading: boolean;
  error: string | null;
}

export const useGraphApi = (): UseGraphApiReturn => {
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  const user_id = useSelector((state: RootState) => state.profile.userId);

  const URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8765";

  const getStats = useCallback(async (): Promise<GraphStats> => {
    setIsLoading(true);
    setError(null);
    try {
      const response = await axios.get<GraphStats>(
        `${URL}/api/v1/graph/stats?user_id=${user_id}`
      );
      setIsLoading(false);
      return response.data;
    } catch (err: any) {
      const errorMessage = err.message || 'Failed to fetch graph stats';
      setError(errorMessage);
      setIsLoading(false);
      return { enabled: false, message: errorMessage };
    }
  }, [user_id, URL]);

  const getHealth = useCallback(async (): Promise<GraphHealthResponse> => {
    setIsLoading(true);
    setError(null);
    try {
      const response = await axios.get<GraphHealthResponse>(
        `${URL}/api/v1/graph/health`
      );
      setIsLoading(false);
      return response.data;
    } catch (err: any) {
      const errorMessage = err.message || 'Failed to fetch graph health';
      setError(errorMessage);
      setIsLoading(false);
      return {
        neo4j_metadata_projection: false,
        neo4j_similarity_edges: false,
        neo4j_gds: false,
        mem0_graph_memory: false,
      };
    }
  }, [URL]);

  const aggregate = useCallback(async (
    dimension: string,
    limit: number = 20
  ): Promise<AggregationBucket[]> => {
    setIsLoading(true);
    setError(null);
    try {
      const response = await axios.get<AggregateResponse>(
        `${URL}/api/v1/graph/aggregate/${dimension}?user_id=${user_id}&limit=${limit}`
      );
      setIsLoading(false);
      return response.data.buckets;
    } catch (err: any) {
      const errorMessage = err.message || 'Failed to fetch aggregation';
      setError(errorMessage);
      setIsLoading(false);
      return [];
    }
  }, [user_id, URL]);

  const getTagCooccurrence = useCallback(async (
    limit: number = 20,
    minCount: number = 2
  ): Promise<TagPair[]> => {
    setIsLoading(true);
    setError(null);
    try {
      const response = await axios.get<TagCooccurrenceResponse>(
        `${URL}/api/v1/graph/tags/cooccurrence?user_id=${user_id}&limit=${limit}&min_count=${minCount}`
      );
      setIsLoading(false);
      return response.data.pairs;
    } catch (err: any) {
      const errorMessage = err.message || 'Failed to fetch tag cooccurrence';
      setError(errorMessage);
      setIsLoading(false);
      return [];
    }
  }, [user_id, URL]);

  const getRelatedTags = useCallback(async (
    tagKey: string,
    minCount: number = 1,
    limit: number = 20
  ): Promise<any[]> => {
    setIsLoading(true);
    setError(null);
    try {
      const response = await axios.get(
        `${URL}/api/v1/graph/tags/${encodeURIComponent(tagKey)}/related?user_id=${user_id}&min_count=${minCount}&limit=${limit}`
      );
      setIsLoading(false);
      return response.data.related || [];
    } catch (err: any) {
      const errorMessage = err.message || 'Failed to fetch related tags';
      setError(errorMessage);
      setIsLoading(false);
      return [];
    }
  }, [user_id, URL]);

  const getTimeline = useCallback(async (params?: {
    entityName?: string;
    eventTypes?: string;
    startYear?: number;
    endYear?: number;
    limit?: number;
  }): Promise<TimelineEvent[]> => {
    setIsLoading(true);
    setError(null);
    try {
      const queryParams = new URLSearchParams({ user_id });
      if (params?.entityName) queryParams.append('entity_name', params.entityName);
      if (params?.eventTypes) queryParams.append('event_types', params.eventTypes);
      if (params?.startYear) queryParams.append('start_year', params.startYear.toString());
      if (params?.endYear) queryParams.append('end_year', params.endYear.toString());
      if (params?.limit) queryParams.append('limit', params.limit.toString());

      const response = await axios.get<TimelineResponse>(
        `${URL}/api/v1/graph/timeline?${queryParams.toString()}`
      );
      setIsLoading(false);
      return response.data.events || [];
    } catch (err: any) {
      const errorMessage = err.message || 'Failed to fetch timeline';
      setError(errorMessage);
      setIsLoading(false);
      return [];
    }
  }, [user_id, URL]);

  const searchMemories = useCallback(async (
    query: string,
    limit: number = 20
  ): Promise<FullTextSearchResult[]> => {
    setIsLoading(true);
    setError(null);
    try {
      const response = await axios.get(
        `${URL}/api/v1/graph/search/memories?query=${encodeURIComponent(query)}&user_id=${user_id}&limit=${limit}`
      );
      setIsLoading(false);
      return response.data.results || [];
    } catch (err: any) {
      const errorMessage = err.message || 'Failed to search memories';
      setError(errorMessage);
      setIsLoading(false);
      return [];
    }
  }, [user_id, URL]);

  const searchEntities = useCallback(async (
    query: string,
    limit: number = 20
  ): Promise<any[]> => {
    setIsLoading(true);
    setError(null);
    try {
      const response = await axios.get(
        `${URL}/api/v1/graph/search/entities?query=${encodeURIComponent(query)}&user_id=${user_id}&limit=${limit}`
      );
      setIsLoading(false);
      return response.data.results || [];
    } catch (err: any) {
      const errorMessage = err.message || 'Failed to search entities';
      setError(errorMessage);
      setIsLoading(false);
      return [];
    }
  }, [user_id, URL]);

  return {
    getStats,
    getHealth,
    aggregate,
    getTagCooccurrence,
    getRelatedTags,
    getTimeline,
    searchMemories,
    searchEntities,
    isLoading,
    error,
  };
};
