import { useState, useCallback } from 'react';
import { useDispatch } from 'react-redux';
import { AppDispatch } from '@/store/store';
import {
  Category,
  setCategoriesLoading,
  setCategoriesSuccess,
  setCategoriesError,
  setSortingState,
  setSelectedApps,
  setSelectedCategories,
  setScopes,
  setArtifactTypes,
  setSources
} from '@/store/filtersSlice';

export interface UseFiltersApiReturn {
  fetchCategories: () => Promise<void>;
  isLoading: boolean;
  error: string | null;
  updateApps: (apps: string[]) => void;
  updateCategories: (categories: string[]) => void;
  updateScopes: (scopes: string[]) => void;
  updateArtifactTypes: (artifactTypes: string[]) => void;
  updateSources: (sources: string[]) => void;
  updateSort: (column: string, direction: 'asc' | 'desc') => void;
}

export const useFiltersApi = (): UseFiltersApiReturn => {
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  const dispatch = useDispatch<AppDispatch>();

  const fetchCategories = useCallback(async (): Promise<void> => {
    setIsLoading(true);
    dispatch(setCategoriesLoading());
    try {
      const structuredCategories: Category[] = [
        { id: "decision", name: "decision", description: "Decisions", created_at: "", updated_at: "" },
        { id: "convention", name: "convention", description: "Conventions", created_at: "", updated_at: "" },
        { id: "architecture", name: "architecture", description: "Architecture", created_at: "", updated_at: "" },
        { id: "dependency", name: "dependency", description: "Dependencies", created_at: "", updated_at: "" },
        { id: "workflow", name: "workflow", description: "Workflows", created_at: "", updated_at: "" },
        { id: "testing", name: "testing", description: "Testing", created_at: "", updated_at: "" },
        { id: "security", name: "security", description: "Security", created_at: "", updated_at: "" },
        { id: "performance", name: "performance", description: "Performance", created_at: "", updated_at: "" },
        { id: "runbook", name: "runbook", description: "Runbooks", created_at: "", updated_at: "" },
        { id: "glossary", name: "glossary", description: "Glossary", created_at: "", updated_at: "" },
      ];

      dispatch(setCategoriesSuccess({
        categories: structuredCategories,
        total: structuredCategories.length
      }));
      setIsLoading(false);
    } catch (err: any) {
      const errorMessage = err.message || 'Failed to fetch categories';
      setError(errorMessage);
      dispatch(setCategoriesError(errorMessage));
      setIsLoading(false);
      throw new Error(errorMessage);
    }
  }, [dispatch]);

  const updateApps = useCallback((apps: string[]) => {
    dispatch(setSelectedApps(apps));
  }, [dispatch]);

  const updateCategories = useCallback((categories: string[]) => {
    dispatch(setSelectedCategories(categories));
  }, [dispatch]);

  const updateScopes = useCallback((scopes: string[]) => {
    dispatch(setScopes(scopes));
  }, [dispatch]);

  const updateArtifactTypes = useCallback((artifactTypes: string[]) => {
    dispatch(setArtifactTypes(artifactTypes));
  }, [dispatch]);

  const updateSources = useCallback((sources: string[]) => {
    dispatch(setSources(sources));
  }, [dispatch]);

  const updateSort = useCallback((column: string, direction: 'asc' | 'desc') => {
    dispatch(setSortingState({ column, direction }));
  }, [dispatch]);

  return {
    fetchCategories,
    isLoading,
    error,
    updateApps,
    updateCategories,
    updateScopes,
    updateArtifactTypes,
    updateSources,
    updateSort
  };
};
