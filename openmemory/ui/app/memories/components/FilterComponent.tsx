"use client";

import { useEffect, useState } from "react";
import { Filter, X, ChevronDown, SortAsc, SortDesc } from "lucide-react";
import { useDispatch, useSelector } from "react-redux";

import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from "@/components/ui/dialog";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Checkbox } from "@/components/ui/checkbox";
import { Label } from "@/components/ui/label";
import { Input } from "@/components/ui/input";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
  DropdownMenuLabel,
  DropdownMenuSeparator,
  DropdownMenuGroup,
} from "@/components/ui/dropdown-menu";
import { RootState } from "@/store/store";
import { useAppsApi } from "@/hooks/useAppsApi";
import { useFiltersApi } from "@/hooks/useFiltersApi";
import { useEntitiesApi } from "@/hooks/useEntitiesApi";
import {
  setSelectedApps,
  setSelectedCategories,
  setScopes,
  setArtifactTypes,
  setEntities,
  setSources,
  clearFilters,
} from "@/store/filtersSlice";
import { useMemoriesApi } from "@/hooks/useMemoriesApi";

const columns = [
  {
    label: "Memory",
    value: "memory",
  },
  {
    label: "App Name",
    value: "app_name",
  },
  {
    label: "Created On",
    value: "created_at",
  },
];

export default function FilterComponent() {
  const dispatch = useDispatch();
  const { fetchApps } = useAppsApi();
  const { fetchCategories, updateSort } = useFiltersApi();
  const { fetchMemories } = useMemoriesApi();
  const { listEntities } = useEntitiesApi();
  const [isOpen, setIsOpen] = useState(false);
  const [tempSelectedApps, setTempSelectedApps] = useState<string[]>([]);
  const [tempSelectedCategories, setTempSelectedCategories] = useState<
    string[]
  >([]);
  const [tempSelectedScopes, setTempSelectedScopes] = useState<string[]>([]);
  const [tempSelectedArtifactTypes, setTempSelectedArtifactTypes] = useState<string[]>([]);
  const [tempSelectedSources, setTempSelectedSources] = useState<string[]>([]);
  const [tempSelectedEntities, setTempSelectedEntities] = useState<string[]>([]);
  const [showArchived, setShowArchived] = useState(false);

  // Search queries for filtering long lists
  const [appSearchQuery, setAppSearchQuery] = useState("");
  const [categorySearchQuery, setCategorySearchQuery] = useState("");
  const [entitySearchQuery, setEntitySearchQuery] = useState("");

  // Available entities from API
  const [availableEntities, setAvailableEntities] = useState<string[]>([]);

  const scopeOptions = [
    { value: "session", label: "Session" },
    { value: "user", label: "User" },
    { value: "team", label: "Team" },
    { value: "project", label: "Project" },
    { value: "org", label: "Org" },
    { value: "enterprise", label: "Enterprise" },
  ];

  const artifactTypeOptions = [
    { value: "repo", label: "Repo" },
    { value: "service", label: "Service" },
    { value: "module", label: "Module" },
    { value: "component", label: "Component" },
    { value: "api", label: "API" },
    { value: "db", label: "Database" },
    { value: "infra", label: "Infra" },
    { value: "file", label: "File" },
  ];

  const sourceOptions = [
    { value: "user", label: "User" },
    { value: "inference", label: "Inference" },
  ];

  const apps = useSelector((state: RootState) => state.apps.apps);
  const categories = useSelector(
    (state: RootState) => state.filters.categories.items
  );
  const filters = useSelector((state: RootState) => state.filters.apps);

  useEffect(() => {
    fetchApps();
    fetchCategories();
    // Load entities
    const loadEntities = async () => {
      try {
        const entities = await listEntities(100);
        setAvailableEntities(entities.map(e => e.name));
      } catch (err) {
        console.error("Failed to load entities:", err);
      }
    };
    loadEntities();
  }, [fetchApps, fetchCategories, listEntities]);

  useEffect(() => {
    // Initialize temporary selections with current active filters when dialog opens
    if (isOpen) {
      setTempSelectedApps(filters.selectedApps);
      setTempSelectedCategories(filters.selectedCategories);
      setTempSelectedScopes(filters.scopes || []);
      setTempSelectedArtifactTypes(filters.artifactTypes || []);
      setTempSelectedSources(filters.sources || []);
      setTempSelectedEntities(filters.entities || []);
      setShowArchived(filters.showArchived || false);
      // Reset search queries when dialog opens
      setAppSearchQuery("");
      setCategorySearchQuery("");
      setEntitySearchQuery("");
    }
  }, [isOpen, filters]);

  useEffect(() => {
    handleClearFilters();
  }, []);

  const toggleAppFilter = (app: string) => {
    setTempSelectedApps((prev) =>
      prev.includes(app) ? prev.filter((a) => a !== app) : [...prev, app]
    );
  };

  const toggleCategoryFilter = (category: string) => {
    setTempSelectedCategories((prev) =>
      prev.includes(category)
        ? prev.filter((c) => c !== category)
        : [...prev, category]
    );
  };

  const toggleAllApps = (checked: boolean) => {
    setTempSelectedApps(checked ? apps.map((app) => app.id) : []);
  };

  const toggleAllCategories = (checked: boolean) => {
    setTempSelectedCategories(checked ? categories.map((cat) => cat.name) : []);
  };

  const handleClearFilters = async () => {
    setTempSelectedApps([]);
    setTempSelectedCategories([]);
    setTempSelectedScopes([]);
    setTempSelectedArtifactTypes([]);
    setTempSelectedSources([]);
    setTempSelectedEntities([]);
    setShowArchived(false);
    setAppSearchQuery("");
    setCategorySearchQuery("");
    setEntitySearchQuery("");
    dispatch(clearFilters());
    await fetchMemories();
  };

  const handleApplyFilters = async () => {
    try {
      // Get app IDs for selected app names
      const selectedAppIds = apps
        .filter((app) => tempSelectedApps.includes(app.id))
        .map((app) => app.id);

      // Update the global state with temporary selections
      dispatch(setSelectedApps(tempSelectedApps));
      dispatch(setSelectedCategories(tempSelectedCategories));
      dispatch(setScopes(tempSelectedScopes));
      dispatch(setArtifactTypes(tempSelectedArtifactTypes));
      dispatch(setEntities(tempSelectedEntities));
      dispatch(setSources(tempSelectedSources));
      dispatch({ type: "filters/setShowArchived", payload: showArchived });

      await fetchMemories(undefined, 1, 10, {
        apps: selectedAppIds,
        categories: tempSelectedCategories,
        sortColumn: filters.sortColumn,
        sortDirection: filters.sortDirection,
        showArchived: showArchived,
        scopes: tempSelectedScopes.length ? tempSelectedScopes : undefined,
        artifactTypes: tempSelectedArtifactTypes.length ? tempSelectedArtifactTypes : undefined,
        entities: tempSelectedEntities.length ? tempSelectedEntities : undefined,
        sources: tempSelectedSources.length ? tempSelectedSources : undefined,
      });
      setIsOpen(false);
    } catch (error) {
      console.error("Failed to apply filters:", error);
    }
  };

  const handleDialogChange = (open: boolean) => {
    setIsOpen(open);
    if (!open) {
      // Reset temporary selections to active filters when dialog closes without applying
      setTempSelectedApps(filters.selectedApps);
      setTempSelectedCategories(filters.selectedCategories);
      setTempSelectedScopes(filters.scopes || []);
      setTempSelectedArtifactTypes(filters.artifactTypes || []);
      setTempSelectedSources(filters.sources || []);
      setTempSelectedEntities(filters.entities || []);
      setShowArchived(filters.showArchived || false);
    }
  };

  const setSorting = async (column: string) => {
    const newDirection =
      filters.sortColumn === column && filters.sortDirection === "asc"
        ? "desc"
        : "asc";
    updateSort(column, newDirection);

    // Get app IDs for selected app names
    const selectedAppIds = apps
      .filter((app) => tempSelectedApps.includes(app.id))
      .map((app) => app.id);

    try {
      await fetchMemories(undefined, 1, 10, {
        apps: selectedAppIds,
        categories: tempSelectedCategories,
        sortColumn: column,
        sortDirection: newDirection,
        scopes: tempSelectedScopes.length ? tempSelectedScopes : undefined,
        artifactTypes: tempSelectedArtifactTypes.length ? tempSelectedArtifactTypes : undefined,
        sources: tempSelectedSources.length ? tempSelectedSources : undefined,
        entities: tempSelectedEntities.length ? tempSelectedEntities : undefined,
      });
    } catch (error) {
      console.error("Failed to apply sorting:", error);
    }
  };

  const hasActiveFilters =
    filters.selectedApps.length > 0 ||
    filters.selectedCategories.length > 0 ||
    filters.scopes.length > 0 ||
    filters.artifactTypes.length > 0 ||
    filters.entities.length > 0 ||
    filters.sources.length > 0 ||
    filters.showArchived;

  const hasTempFilters =
    tempSelectedApps.length > 0 ||
    tempSelectedCategories.length > 0 ||
    tempSelectedScopes.length > 0 ||
    tempSelectedArtifactTypes.length > 0 ||
    tempSelectedSources.length > 0 ||
    tempSelectedEntities.length > 0 ||
    showArchived;

  const activeFilterCount =
    filters.selectedApps.length +
    filters.selectedCategories.length +
    filters.scopes.length +
    filters.artifactTypes.length +
    filters.sources.length +
    filters.entities.length +
    (filters.showArchived ? 1 : 0);

  return (
    <div className="flex items-center gap-2">
      <Dialog open={isOpen} onOpenChange={handleDialogChange}>
        <DialogTrigger asChild>
          <Button
            variant="outline"
            className={`h-9 px-4 border-zinc-700/50 bg-zinc-900 hover:bg-zinc-800 ${
              hasActiveFilters ? "border-primary" : ""
            }`}
          >
            <Filter
              className={`h-4 w-4 ${hasActiveFilters ? "text-primary" : ""}`}
            />
            Filter
            {hasActiveFilters && (
              <Badge className="ml-2 bg-primary hover:bg-primary/80 text-xs">
                {activeFilterCount}
              </Badge>
            )}
          </Button>
        </DialogTrigger>
        <DialogContent className="sm:max-w-[560px] bg-zinc-900 border-zinc-800 text-zinc-100">
          <DialogHeader>
            <DialogTitle className="text-zinc-100 flex justify-between items-center">
              <span>Filters</span>
            </DialogTitle>
          </DialogHeader>
          <Tabs defaultValue="metadata" className="w-full">
            <TabsList className="grid grid-cols-5 bg-zinc-800">
              <TabsTrigger
                value="metadata"
                className="data-[state=active]:bg-zinc-700"
              >
                Metadata
              </TabsTrigger>
              <TabsTrigger
                value="entities"
                className="data-[state=active]:bg-zinc-700"
              >
                Entities
              </TabsTrigger>
              <TabsTrigger
                value="apps"
                className="data-[state=active]:bg-zinc-700"
              >
                Apps
              </TabsTrigger>
              <TabsTrigger
                value="categories"
                className="data-[state=active]:bg-zinc-700"
              >
                Categories
              </TabsTrigger>
              <TabsTrigger
                value="status"
                className="data-[state=active]:bg-zinc-700"
              >
                Status
              </TabsTrigger>
            </TabsList>
            <TabsContent value="apps" className="mt-4">
              <div className="space-y-3">
                <Input
                  placeholder="Search apps..."
                  value={appSearchQuery}
                  onChange={(e) => setAppSearchQuery(e.target.value)}
                  className="bg-zinc-800 border-zinc-700 text-white"
                />
                <div className="max-h-[250px] overflow-y-auto pr-2 space-y-2">
                  <div className="flex items-center space-x-2 sticky top-0 bg-zinc-900 pb-2 z-10">
                    <Checkbox
                      id="select-all-apps"
                      checked={
                        apps.length > 0 && tempSelectedApps.length === apps.length
                      }
                      onCheckedChange={(checked) =>
                        toggleAllApps(checked as boolean)
                      }
                      className="border-zinc-600 data-[state=checked]:bg-primary data-[state=checked]:border-primary"
                    />
                    <Label
                      htmlFor="select-all-apps"
                      className="text-sm font-normal text-zinc-300 cursor-pointer"
                    >
                      Select All
                    </Label>
                  </div>
                  {apps
                    .filter((app) =>
                      app.name.toLowerCase().includes(appSearchQuery.toLowerCase())
                    )
                    .map((app) => (
                      <div key={app.id} className="flex items-center space-x-2">
                        <Checkbox
                          id={`app-${app.id}`}
                          checked={tempSelectedApps.includes(app.id)}
                          onCheckedChange={() => toggleAppFilter(app.id)}
                          className="border-zinc-600 data-[state=checked]:bg-primary data-[state=checked]:border-primary"
                        />
                        <Label
                          htmlFor={`app-${app.id}`}
                          className="text-sm font-normal text-zinc-300 cursor-pointer"
                        >
                          {app.name}
                        </Label>
                      </div>
                    ))}
                </div>
              </div>
            </TabsContent>
            <TabsContent value="categories" className="mt-4">
              <div className="space-y-3">
                <Input
                  placeholder="Search categories..."
                  value={categorySearchQuery}
                  onChange={(e) => setCategorySearchQuery(e.target.value)}
                  className="bg-zinc-800 border-zinc-700 text-white"
                />
                <div className="max-h-[250px] overflow-y-auto pr-2 space-y-2">
                  <div className="flex items-center space-x-2 sticky top-0 bg-zinc-900 pb-2 z-10">
                    <Checkbox
                      id="select-all-categories"
                      checked={
                        categories.length > 0 &&
                        tempSelectedCategories.length === categories.length
                      }
                      onCheckedChange={(checked) =>
                        toggleAllCategories(checked as boolean)
                      }
                      className="border-zinc-600 data-[state=checked]:bg-primary data-[state=checked]:border-primary"
                    />
                    <Label
                      htmlFor="select-all-categories"
                      className="text-sm font-normal text-zinc-300 cursor-pointer"
                    >
                      Select All
                    </Label>
                  </div>
                  {categories
                    .filter((category) =>
                      category.name.toLowerCase().includes(categorySearchQuery.toLowerCase())
                    )
                    .map((category) => (
                      <div
                        key={category.name}
                        className="flex items-center space-x-2"
                      >
                        <Checkbox
                          id={`category-${category.name}`}
                          checked={tempSelectedCategories.includes(category.name)}
                          onCheckedChange={() =>
                            toggleCategoryFilter(category.name)
                          }
                          className="border-zinc-600 data-[state=checked]:bg-primary data-[state=checked]:border-primary"
                        />
                        <Label
                          htmlFor={`category-${category.name}`}
                          className="text-sm font-normal text-zinc-300 cursor-pointer"
                        >
                          {category.name}
                        </Label>
                      </div>
                    ))}
                </div>
              </div>
            </TabsContent>
            <TabsContent value="status" className="mt-4">
              <div className="space-y-3">
                <div className="flex items-center space-x-2">
                  <Checkbox
                    id="show-archived"
                    checked={showArchived}
                    onCheckedChange={(checked) =>
                      setShowArchived(checked as boolean)
                    }
                    className="border-zinc-600 data-[state=checked]:bg-primary data-[state=checked]:border-primary"
                  />
                  <Label
                    htmlFor="show-archived"
                    className="text-sm font-normal text-zinc-300 cursor-pointer"
                  >
                    Show Archived Memories
                  </Label>
                </div>
              </div>
            </TabsContent>
            <TabsContent value="metadata" className="mt-4">
              <div className="max-h-[350px] overflow-y-auto pr-2 space-y-4">
                {/* Scopes */}
                <div className="space-y-2">
                  <Label className="text-sm font-medium text-zinc-300">
                    Scope
                  </Label>
                  <div className="space-y-2">
                    <div className="flex items-center space-x-2">
                      <Checkbox
                        id="select-all-scopes"
                        checked={
                          scopeOptions.length > 0 &&
                          tempSelectedScopes.length === scopeOptions.length
                        }
                        onCheckedChange={(checked) =>
                          setTempSelectedScopes(
                            checked ? scopeOptions.map(s => s.value) : []
                          )
                        }
                        className="border-zinc-600 data-[state=checked]:bg-primary data-[state=checked]:border-primary"
                      />
                      <Label
                        htmlFor="select-all-scopes"
                        className="text-sm font-normal text-zinc-300 cursor-pointer"
                      >
                        Select All
                      </Label>
                    </div>
                    {scopeOptions.map((scope) => (
                      <div key={scope.value} className="flex items-center space-x-2">
                        <Checkbox
                          id={`scope-${scope.value}`}
                          checked={tempSelectedScopes.includes(scope.value)}
                          onCheckedChange={() =>
                            setTempSelectedScopes((prev) =>
                              prev.includes(scope.value)
                                ? prev.filter((s) => s !== scope.value)
                                : [...prev, scope.value]
                            )
                          }
                          className="border-zinc-600 data-[state=checked]:bg-primary data-[state=checked]:border-primary"
                        />
                        <Label
                          htmlFor={`scope-${scope.value}`}
                          className="text-sm font-normal text-zinc-300 cursor-pointer"
                        >
                          {scope.label}
                        </Label>
                      </div>
                    ))}
                  </div>
                </div>

                {/* Artifact Types */}
                <div className="space-y-2">
                  <Label className="text-sm font-medium text-zinc-300">
                    Artifact Type
                  </Label>
                  <div className="space-y-2">
                    <div className="flex items-center space-x-2">
                      <Checkbox
                        id="select-all-artifact-types"
                        checked={
                          artifactTypeOptions.length > 0 &&
                          tempSelectedArtifactTypes.length === artifactTypeOptions.length
                        }
                        onCheckedChange={(checked) =>
                          setTempSelectedArtifactTypes(
                            checked ? artifactTypeOptions.map(t => t.value) : []
                          )
                        }
                        className="border-zinc-600 data-[state=checked]:bg-primary data-[state=checked]:border-primary"
                      />
                      <Label
                        htmlFor="select-all-artifact-types"
                        className="text-sm font-normal text-zinc-300 cursor-pointer"
                      >
                        Select All
                      </Label>
                    </div>
                    {artifactTypeOptions.map((artifactType) => (
                      <div key={artifactType.value} className="flex items-center space-x-2">
                        <Checkbox
                          id={`artifact-type-${artifactType.value}`}
                          checked={tempSelectedArtifactTypes.includes(artifactType.value)}
                          onCheckedChange={() =>
                            setTempSelectedArtifactTypes((prev) =>
                              prev.includes(artifactType.value)
                                ? prev.filter((t) => t !== artifactType.value)
                                : [...prev, artifactType.value]
                            )
                          }
                          className="border-zinc-600 data-[state=checked]:bg-primary data-[state=checked]:border-primary"
                        />
                        <Label
                          htmlFor={`artifact-type-${artifactType.value}`}
                          className="text-sm font-normal text-zinc-300 cursor-pointer"
                        >
                          {artifactType.label}
                        </Label>
                      </div>
                    ))}
                  </div>
                </div>

                {/* Sources */}
                <div className="space-y-2">
                  <Label className="text-sm font-medium text-zinc-300">
                    Source
                  </Label>
                  <div className="space-y-2">
                    {sourceOptions.map((source) => (
                      <div key={source.value} className="flex items-center space-x-2">
                        <Checkbox
                          id={`source-${source.value}`}
                          checked={tempSelectedSources.includes(source.value)}
                          onCheckedChange={() =>
                            setTempSelectedSources((prev) =>
                              prev.includes(source.value)
                                ? prev.filter((s) => s !== source.value)
                                : [...prev, source.value]
                            )
                          }
                          className="border-zinc-600 data-[state=checked]:bg-primary data-[state=checked]:border-primary"
                        />
                        <Label
                          htmlFor={`source-${source.value}`}
                          className="text-sm font-normal text-zinc-300 cursor-pointer"
                        >
                          {source.label}
                        </Label>
                      </div>
                    ))}
                  </div>
                </div>
              </div>
            </TabsContent>

            {/* Entities Tab */}
            <TabsContent value="entities" className="mt-4">
              <div className="space-y-3">
                <Input
                  placeholder="Search entities..."
                  value={entitySearchQuery}
                  onChange={(e) => setEntitySearchQuery(e.target.value)}
                  className="bg-zinc-800 border-zinc-700 text-white"
                />
                <div className="max-h-[250px] overflow-y-auto pr-2 space-y-2">
                  {availableEntities.length === 0 ? (
                    <p className="text-sm text-zinc-500">No entities found</p>
                  ) : (
                    availableEntities
                      .filter((entity) =>
                        entity.toLowerCase().includes(entitySearchQuery.toLowerCase())
                      )
                      .map((entity) => (
                        <div key={entity} className="flex items-center space-x-2">
                          <Checkbox
                            id={`entity-${entity}`}
                            checked={tempSelectedEntities.includes(entity)}
                            onCheckedChange={() =>
                              setTempSelectedEntities((prev) =>
                                prev.includes(entity)
                                  ? prev.filter((e) => e !== entity)
                                  : [...prev, entity]
                              )
                            }
                            className="border-zinc-600 data-[state=checked]:bg-primary data-[state=checked]:border-primary"
                          />
                          <Label
                            htmlFor={`entity-${entity}`}
                            className="text-sm font-normal text-zinc-300 cursor-pointer"
                          >
                            {entity}
                          </Label>
                        </div>
                      ))
                  )}
                </div>
              </div>
            </TabsContent>
          </Tabs>
          <div className="flex justify-end mt-4 gap-3">
            {/* Clear all button */}
            {hasTempFilters && (
              <Button
                onClick={handleClearFilters}
                className="bg-zinc-800 hover:bg-zinc-700 text-zinc-300"
              >
                Clear All
              </Button>
            )}
            {/* Apply filters button */}
            <Button
              onClick={handleApplyFilters}
              className="bg-primary hover:bg-primary/80 text-white"
            >
              Apply Filters
            </Button>
          </div>
        </DialogContent>
      </Dialog>

      <DropdownMenu>
        <DropdownMenuTrigger asChild>
          <Button
            variant="outline"
            className="h-9 px-4 border-zinc-700/50 bg-zinc-900 hover:bg-zinc-800"
          >
            {filters.sortDirection === "asc" ? (
              <SortAsc className="h-4 w-4" />
            ) : (
              <SortDesc className="h-4 w-4" />
            )}
            Sort: {columns.find((c) => c.value === filters.sortColumn)?.label}
            <ChevronDown className="h-4 w-4 ml-2" />
          </Button>
        </DropdownMenuTrigger>
        <DropdownMenuContent className="w-56 bg-zinc-900 border-zinc-800 text-zinc-100">
          <DropdownMenuLabel>Sort by</DropdownMenuLabel>
          <DropdownMenuSeparator className="bg-zinc-800" />
          <DropdownMenuGroup>
            {columns.map((column) => (
              <DropdownMenuItem
                key={column.value}
                onClick={() => setSorting(column.value)}
                className="cursor-pointer flex justify-between items-center"
              >
                {column.label}
                {filters.sortColumn === column.value &&
                  (filters.sortDirection === "asc" ? (
                    <SortAsc className="h-4 w-4 text-primary" />
                  ) : (
                    <SortDesc className="h-4 w-4 text-primary" />
                  ))}
              </DropdownMenuItem>
            ))}
          </DropdownMenuGroup>
        </DropdownMenuContent>
      </DropdownMenu>
    </div>
  );
}
