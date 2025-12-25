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
  setVaults,
  setLayers,
  setVectors,
  setCircuits,
  setEntities,
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
  const [tempSelectedVaults, setTempSelectedVaults] = useState<string[]>([]);
  const [tempSelectedLayers, setTempSelectedLayers] = useState<string[]>([]);
  const [tempSelectedVectors, setTempSelectedVectors] = useState<string[]>([]);
  const [tempSelectedCircuits, setTempSelectedCircuits] = useState<number[]>([]);
  const [tempSelectedEntities, setTempSelectedEntities] = useState<string[]>([]);
  const [showArchived, setShowArchived] = useState(false);

  // Search queries for filtering long lists
  const [appSearchQuery, setAppSearchQuery] = useState("");
  const [categorySearchQuery, setCategorySearchQuery] = useState("");
  const [entitySearchQuery, setEntitySearchQuery] = useState("");

  // Available entities from API
  const [availableEntities, setAvailableEntities] = useState<string[]>([]);

  // AXIS Protocol Vaults
  const vaultOptions = [
    { value: "SOV", label: "SOV - Sovereignty Core" },
    { value: "WLT", label: "WLT - Wealth & Work" },
    { value: "SIG", label: "SIG - Signal Processing" },
    { value: "FRC", label: "FRC - Force Health" },
    { value: "DIR", label: "DIR - Direction System" },
    { value: "FGP", label: "FGP - Fingerprint Evolution" },
    { value: "Q", label: "Q - Questions" },
  ];

  // AXIS Protocol Layers
  const layerOptions = [
    { value: "somatic", label: "Somatic" },
    { value: "emotional", label: "Emotional" },
    { value: "narrative", label: "Narrative" },
    { value: "cognitive", label: "Cognitive" },
    { value: "values", label: "Values" },
    { value: "identity", label: "Identity" },
    { value: "relational", label: "Relational" },
    { value: "goals", label: "Goals" },
    { value: "resources", label: "Resources" },
    { value: "context", label: "Context" },
    { value: "temporal", label: "Temporal" },
    { value: "meta", label: "Meta" },
  ];

  // AXIS Protocol Vectors
  const vectorOptions = [
    { value: "say", label: "Say - What I express" },
    { value: "want", label: "Want - What I desire" },
    { value: "do", label: "Do - What I act on" },
  ];

  // AXIS Protocol Circuits (1-8)
  const circuitOptions = [
    { value: 1, label: "1 - Survival" },
    { value: 2, label: "2 - Emotional-Territorial" },
    { value: 3, label: "3 - Semantic-Symbolic" },
    { value: 4, label: "4 - Social-Sexual" },
    { value: 5, label: "5 - Neurosomatic" },
    { value: 6, label: "6 - Neuroelectric" },
    { value: 7, label: "7 - Neurogenetic" },
    { value: 8, label: "8 - Quantum" },
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
      setTempSelectedVaults(filters.vaults || []);
      setTempSelectedLayers(filters.layers || []);
      setTempSelectedVectors(filters.vectors || []);
      setTempSelectedCircuits(filters.circuits || []);
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
    setTempSelectedVaults([]);
    setTempSelectedLayers([]);
    setTempSelectedVectors([]);
    setTempSelectedCircuits([]);
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
      // Get category IDs for selected category names
      const selectedCategoryIds = categories
        .filter((cat) => tempSelectedCategories.includes(cat.name))
        .map((cat) => cat.id);

      // Get app IDs for selected app names
      const selectedAppIds = apps
        .filter((app) => tempSelectedApps.includes(app.id))
        .map((app) => app.id);

      // Update the global state with temporary selections
      dispatch(setSelectedApps(tempSelectedApps));
      dispatch(setSelectedCategories(tempSelectedCategories));
      dispatch(setVaults(tempSelectedVaults));
      dispatch(setLayers(tempSelectedLayers));
      dispatch(setVectors(tempSelectedVectors));
      dispatch(setCircuits(tempSelectedCircuits));
      dispatch(setEntities(tempSelectedEntities));
      dispatch({ type: "filters/setShowArchived", payload: showArchived });

      await fetchMemories(undefined, 1, 10, {
        apps: selectedAppIds,
        categories: selectedCategoryIds,
        sortColumn: filters.sortColumn,
        sortDirection: filters.sortDirection,
        showArchived: showArchived,
        vaults: tempSelectedVaults.length ? tempSelectedVaults : undefined,
        layers: tempSelectedLayers.length ? tempSelectedLayers : undefined,
        vectors: tempSelectedVectors.length ? tempSelectedVectors : undefined,
        circuits: tempSelectedCircuits.length ? tempSelectedCircuits : undefined,
        entities: tempSelectedEntities.length ? tempSelectedEntities : undefined,
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
      setTempSelectedVaults(filters.vaults || []);
      setTempSelectedLayers(filters.layers || []);
      setTempSelectedVectors(filters.vectors || []);
      setTempSelectedCircuits(filters.circuits || []);
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

    // Get category IDs for selected category names
    const selectedCategoryIds = categories
      .filter((cat) => tempSelectedCategories.includes(cat.name))
      .map((cat) => cat.id);

    // Get app IDs for selected app names
    const selectedAppIds = apps
      .filter((app) => tempSelectedApps.includes(app.id))
      .map((app) => app.id);

    try {
      await fetchMemories(undefined, 1, 10, {
        apps: selectedAppIds,
        categories: selectedCategoryIds,
        sortColumn: column,
        sortDirection: newDirection,
        vaults: tempSelectedVaults.length ? tempSelectedVaults : undefined,
        layers: tempSelectedLayers.length ? tempSelectedLayers : undefined,
      });
    } catch (error) {
      console.error("Failed to apply sorting:", error);
    }
  };

  const hasActiveFilters =
    filters.selectedApps.length > 0 ||
    filters.selectedCategories.length > 0 ||
    filters.vaults.length > 0 ||
    filters.layers.length > 0 ||
    filters.vectors.length > 0 ||
    filters.circuits.length > 0 ||
    filters.entities.length > 0 ||
    filters.showArchived;

  const hasTempFilters =
    tempSelectedApps.length > 0 ||
    tempSelectedCategories.length > 0 ||
    tempSelectedVaults.length > 0 ||
    tempSelectedLayers.length > 0 ||
    tempSelectedVectors.length > 0 ||
    tempSelectedCircuits.length > 0 ||
    tempSelectedEntities.length > 0 ||
    showArchived;

  const activeFilterCount =
    filters.selectedApps.length +
    filters.selectedCategories.length +
    filters.vaults.length +
    filters.layers.length +
    filters.vectors.length +
    filters.circuits.length +
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
          <Tabs defaultValue="axis" className="w-full">
            <TabsList className="grid grid-cols-5 bg-zinc-800">
              <TabsTrigger
                value="axis"
                className="data-[state=active]:bg-zinc-700"
              >
                AXIS
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
            <TabsContent value="axis" className="mt-4">
              <div className="max-h-[350px] overflow-y-auto pr-2 space-y-4">
                {/* Vaults */}
                <div className="space-y-2">
                  <Label className="text-sm font-medium text-zinc-300">
                    Vaults
                  </Label>
                  <div className="space-y-2">
                    <div className="flex items-center space-x-2">
                      <Checkbox
                        id="select-all-vaults"
                        checked={
                          vaultOptions.length > 0 &&
                          tempSelectedVaults.length === vaultOptions.length
                        }
                        onCheckedChange={(checked) =>
                          setTempSelectedVaults(
                            checked ? vaultOptions.map(v => v.value) : []
                          )
                        }
                        className="border-zinc-600 data-[state=checked]:bg-primary data-[state=checked]:border-primary"
                      />
                      <Label
                        htmlFor="select-all-vaults"
                        className="text-sm font-normal text-zinc-300 cursor-pointer"
                      >
                        Select All
                      </Label>
                    </div>
                    {vaultOptions.map((vault) => (
                      <div key={vault.value} className="flex items-center space-x-2">
                        <Checkbox
                          id={`vault-${vault.value}`}
                          checked={tempSelectedVaults.includes(vault.value)}
                          onCheckedChange={() =>
                            setTempSelectedVaults((prev) =>
                              prev.includes(vault.value)
                                ? prev.filter((v) => v !== vault.value)
                                : [...prev, vault.value]
                            )
                          }
                          className="border-zinc-600 data-[state=checked]:bg-primary data-[state=checked]:border-primary"
                        />
                        <Label
                          htmlFor={`vault-${vault.value}`}
                          className="text-sm font-normal text-zinc-300 cursor-pointer"
                        >
                          {vault.label}
                        </Label>
                      </div>
                    ))}
                  </div>
                </div>

                {/* Layers */}
                <div className="space-y-2">
                  <Label className="text-sm font-medium text-zinc-300">
                    Layers
                  </Label>
                  <div className="space-y-2">
                    <div className="flex items-center space-x-2">
                      <Checkbox
                        id="select-all-layers"
                        checked={
                          layerOptions.length > 0 &&
                          tempSelectedLayers.length === layerOptions.length
                        }
                        onCheckedChange={(checked) =>
                          setTempSelectedLayers(
                            checked ? layerOptions.map(l => l.value) : []
                          )
                        }
                        className="border-zinc-600 data-[state=checked]:bg-primary data-[state=checked]:border-primary"
                      />
                      <Label
                        htmlFor="select-all-layers"
                        className="text-sm font-normal text-zinc-300 cursor-pointer"
                      >
                        Select All
                      </Label>
                    </div>
                    {layerOptions.map((layer) => (
                      <div key={layer.value} className="flex items-center space-x-2">
                        <Checkbox
                          id={`layer-${layer.value}`}
                          checked={tempSelectedLayers.includes(layer.value)}
                          onCheckedChange={() =>
                            setTempSelectedLayers((prev) =>
                              prev.includes(layer.value)
                                ? prev.filter((l) => l !== layer.value)
                                : [...prev, layer.value]
                            )
                          }
                          className="border-zinc-600 data-[state=checked]:bg-primary data-[state=checked]:border-primary"
                        />
                        <Label
                          htmlFor={`layer-${layer.value}`}
                          className="text-sm font-normal text-zinc-300 cursor-pointer"
                        >
                          {layer.label}
                        </Label>
                      </div>
                    ))}
                  </div>
                </div>

                {/* Vectors */}
                <div className="space-y-2">
                  <Label className="text-sm font-medium text-zinc-300">
                    Vectors (Say/Want/Do)
                  </Label>
                  <div className="space-y-2">
                    {vectorOptions.map((vector) => (
                      <div key={vector.value} className="flex items-center space-x-2">
                        <Checkbox
                          id={`vector-${vector.value}`}
                          checked={tempSelectedVectors.includes(vector.value)}
                          onCheckedChange={() =>
                            setTempSelectedVectors((prev) =>
                              prev.includes(vector.value)
                                ? prev.filter((v) => v !== vector.value)
                                : [...prev, vector.value]
                            )
                          }
                          className="border-zinc-600 data-[state=checked]:bg-primary data-[state=checked]:border-primary"
                        />
                        <Label
                          htmlFor={`vector-${vector.value}`}
                          className="text-sm font-normal text-zinc-300 cursor-pointer"
                        >
                          {vector.label}
                        </Label>
                      </div>
                    ))}
                  </div>
                </div>

                {/* Circuits */}
                <div className="space-y-2">
                  <Label className="text-sm font-medium text-zinc-300">
                    Circuits
                  </Label>
                  <div className="grid grid-cols-2 gap-2">
                    {circuitOptions.map((circuit) => (
                      <div key={circuit.value} className="flex items-center space-x-2">
                        <Checkbox
                          id={`circuit-${circuit.value}`}
                          checked={tempSelectedCircuits.includes(circuit.value)}
                          onCheckedChange={() =>
                            setTempSelectedCircuits((prev) =>
                              prev.includes(circuit.value)
                                ? prev.filter((c) => c !== circuit.value)
                                : [...prev, circuit.value]
                            )
                          }
                          className="border-zinc-600 data-[state=checked]:bg-primary data-[state=checked]:border-primary"
                        />
                        <Label
                          htmlFor={`circuit-${circuit.value}`}
                          className="text-sm font-normal text-zinc-300 cursor-pointer"
                        >
                          {circuit.label}
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
