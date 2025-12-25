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
    vaults: string[];
    layers: string[];
    vectors: string[];
    circuits: number[];
    entities: string[];
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
    vaults: [],
    layers: [],
    vectors: [],
    circuits: [],
    entities: [],
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
    setVaults: (state, action: PayloadAction<string[]>) => {
      state.apps.vaults = action.payload;
    },
    setLayers: (state, action: PayloadAction<string[]>) => {
      state.apps.layers = action.payload;
    },
    setVectors: (state, action: PayloadAction<string[]>) => {
      state.apps.vectors = action.payload;
    },
    setCircuits: (state, action: PayloadAction<number[]>) => {
      state.apps.circuits = action.payload;
    },
    setEntities: (state, action: PayloadAction<string[]>) => {
      state.apps.entities = action.payload;
    },
    setShowArchived: (state, action: PayloadAction<boolean>) => {
      state.apps.showArchived = action.payload;
    },
    clearFilters: (state) => {
      state.apps.selectedApps = [];
      state.apps.selectedCategories = [];
      state.apps.vaults = [];
      state.apps.layers = [];
      state.apps.vectors = [];
      state.apps.circuits = [];
      state.apps.entities = [];
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
  setVaults,
  setLayers,
  setVectors,
  setCircuits,
  setEntities,
  setShowArchived,
  clearFilters,
  setSortingState
} = filtersSlice.actions;

export default filtersSlice.reducer; 
