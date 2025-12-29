import { createSlice, PayloadAction } from '@reduxjs/toolkit';

export interface Category {
  id: string;
  name: string;
  description: string;
  updated_at: string;
  created_at: string;
}

export interface FiltersState {
  apps: {
    selectedApps: string[];
    selectedCategories: string[];
    scopes: string[];
    artifactTypes: string[];
    entities: string[];
    sources: string[];
    sortColumn: string;
    sortDirection: 'asc' | 'desc';
    showArchived: boolean;
  };
  categories: {
    items: Category[];
    total: number;
    isLoading: boolean;
    error: string | null;
  };
}

const initialState: FiltersState = {
  apps: {
    selectedApps: [],
    selectedCategories: [],
    scopes: [],
    artifactTypes: [],
    entities: [],
    sources: [],
    sortColumn: 'created_at',
    sortDirection: 'desc',
    showArchived: false,
  },
  categories: {
    items: [],
    total: 0,
    isLoading: false,
    error: null
  }
};

const filtersSlice = createSlice({
  name: 'filters',
  initialState,
  reducers: {
    setCategoriesLoading: (state) => {
      state.categories.isLoading = true;
      state.categories.error = null;
    },
    setCategoriesSuccess: (state, action: PayloadAction<{ categories: Category[]; total: number }>) => {
      state.categories.items = action.payload.categories;
      state.categories.total = action.payload.total;
      state.categories.isLoading = false;
      state.categories.error = null;
    },
    setCategoriesError: (state, action: PayloadAction<string>) => {
      state.categories.isLoading = false;
      state.categories.error = action.payload;
    },
    setSelectedApps: (state, action: PayloadAction<string[]>) => {
      state.apps.selectedApps = action.payload;
    },
    setSelectedCategories: (state, action: PayloadAction<string[]>) => {
      state.apps.selectedCategories = action.payload;
    },
    setScopes: (state, action: PayloadAction<string[]>) => {
      state.apps.scopes = action.payload;
    },
    setArtifactTypes: (state, action: PayloadAction<string[]>) => {
      state.apps.artifactTypes = action.payload;
    },
    setEntities: (state, action: PayloadAction<string[]>) => {
      state.apps.entities = action.payload;
    },
    setSources: (state, action: PayloadAction<string[]>) => {
      state.apps.sources = action.payload;
    },
    setShowArchived: (state, action: PayloadAction<boolean>) => {
      state.apps.showArchived = action.payload;
    },
    clearFilters: (state) => {
      state.apps.selectedApps = [];
      state.apps.selectedCategories = [];
      state.apps.scopes = [];
      state.apps.artifactTypes = [];
      state.apps.entities = [];
      state.apps.sources = [];
      state.apps.showArchived = false;
    },
    setSortingState: (state, action: PayloadAction<{ column: string; direction: 'asc' | 'desc' }>) => {
      state.apps.sortColumn = action.payload.column;
      state.apps.sortDirection = action.payload.direction;
    },
  },
});

export const {
  setCategoriesLoading,
  setCategoriesSuccess,
  setCategoriesError,
  setSelectedApps,
  setSelectedCategories,
  setScopes,
  setArtifactTypes,
  setEntities,
  setSources,
  setShowArchived,
  clearFilters,
  setSortingState
} = filtersSlice.actions;

export default filtersSlice.reducer; 
