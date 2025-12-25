"use client";

import { useState, useEffect } from "react";
import { useRouter } from "next/navigation";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Skeleton } from "@/components/ui/skeleton";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { useEntitiesApi } from "@/hooks/useEntitiesApi";
import { useGraphApi } from "@/hooks/useGraphApi";
import { Entity, AggregationBucket } from "@/components/types";
import {
  Users,
  Search,
  Network,
  ExternalLink,
  ArrowUpDown,
  Database,
  GitBranch,
} from "lucide-react";
import "@/styles/animation.css";

export default function EntitiesPage() {
  const router = useRouter();
  const { listEntities, isLoading: entitiesLoading } = useEntitiesApi();
  const { aggregate, getStats, isLoading: graphLoading } = useGraphApi();

  const [entities, setEntities] = useState<Entity[]>([]);
  const [filteredEntities, setFilteredEntities] = useState<Entity[]>([]);
  const [searchQuery, setSearchQuery] = useState("");
  const [sortBy, setSortBy] = useState<"name" | "count">("count");
  const [sortDir, setSortDir] = useState<"asc" | "desc">("desc");
  const [graphStats, setGraphStats] = useState<any>(null);
  const [vaultDistribution, setVaultDistribution] = useState<AggregationBucket[]>([]);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const loadData = async () => {
      try {
        const [entityList, stats, vaults] = await Promise.all([
          listEntities(200, 1),
          getStats(),
          aggregate("vault", 10),
        ]);
        setEntities(entityList);
        setFilteredEntities(entityList);
        setGraphStats(stats);
        setVaultDistribution(vaults);
        setError(null);
      } catch (err) {
        setError("Failed to load entities");
      }
    };

    loadData();
  }, []);

  useEffect(() => {
    let filtered = entities.filter((e) =>
      e.name.toLowerCase().includes(searchQuery.toLowerCase())
    );

    filtered.sort((a, b) => {
      if (sortBy === "name") {
        return sortDir === "asc"
          ? a.name.localeCompare(b.name)
          : b.name.localeCompare(a.name);
      } else {
        return sortDir === "asc"
          ? a.memory_count - b.memory_count
          : b.memory_count - a.memory_count;
      }
    });

    setFilteredEntities(filtered);
  }, [entities, searchQuery, sortBy, sortDir]);

  const handleEntityClick = (name: string) => {
    router.push(`/entity/${encodeURIComponent(name)}`);
  };

  const toggleSort = (column: "name" | "count") => {
    if (sortBy === column) {
      setSortDir(sortDir === "asc" ? "desc" : "asc");
    } else {
      setSortBy(column);
      setSortDir(column === "count" ? "desc" : "asc");
    }
  };

  const isLoading = entitiesLoading || graphLoading;

  return (
    <div className="container mx-auto py-6 px-4">
      <div className="mb-6 animate-fade-slide-down">
        <h1 className="text-2xl font-bold text-white flex items-center gap-2">
          <Users className="h-6 w-6 text-primary" />
          Entities
        </h1>
        <p className="text-zinc-400 text-sm mt-1">
          People, places, and concepts extracted from your memories
        </p>
      </div>

      {/* Stats Cards */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4 mb-6 animate-fade-slide-down delay-1">
        <Card className="bg-zinc-900 border-zinc-800">
          <CardContent className="pt-4">
            <div className="flex items-center gap-3">
              <div className="p-2 rounded-lg bg-primary/20">
                <Users className="h-5 w-5 text-primary" />
              </div>
              <div>
                <p className="text-2xl font-bold text-white">
                  {isLoading ? (
                    <Skeleton className="h-7 w-12 bg-zinc-800" />
                  ) : (
                    entities.length
                  )}
                </p>
                <p className="text-xs text-zinc-400">Total Entities</p>
              </div>
            </div>
          </CardContent>
        </Card>

        <Card className="bg-zinc-900 border-zinc-800">
          <CardContent className="pt-4">
            <div className="flex items-center gap-3">
              <div className="p-2 rounded-lg bg-blue-900/30">
                <Database className="h-5 w-5 text-blue-400" />
              </div>
              <div>
                <p className="text-2xl font-bold text-white">
                  {isLoading ? (
                    <Skeleton className="h-7 w-12 bg-zinc-800" />
                  ) : (
                    graphStats?.memoryCount ?? 0
                  )}
                </p>
                <p className="text-xs text-zinc-400">Total Memories</p>
              </div>
            </div>
          </CardContent>
        </Card>

        <Card className="bg-zinc-900 border-zinc-800">
          <CardContent className="pt-4">
            <div className="flex items-center gap-3">
              <div className="p-2 rounded-lg bg-green-900/30">
                <GitBranch className="h-5 w-5 text-green-400" />
              </div>
              <div>
                <p className="text-2xl font-bold text-white">
                  {isLoading ? (
                    <Skeleton className="h-7 w-12 bg-zinc-800" />
                  ) : (
                    graphStats?.similarityEdges ?? 0
                  )}
                </p>
                <p className="text-xs text-zinc-400">Similarity Links</p>
              </div>
            </div>
          </CardContent>
        </Card>

        <Card className="bg-zinc-900 border-zinc-800">
          <CardContent className="pt-4">
            <div className="flex items-center gap-3">
              <div className="p-2 rounded-lg bg-amber-900/30">
                <Network className="h-5 w-5 text-amber-400" />
              </div>
              <div>
                <p className="text-2xl font-bold text-white">
                  {isLoading ? (
                    <Skeleton className="h-7 w-12 bg-zinc-800" />
                  ) : (
                    graphStats?.coMentionEdges ?? 0
                  )}
                </p>
                <p className="text-xs text-zinc-400">Co-Mentions</p>
              </div>
            </div>
          </CardContent>
        </Card>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-4 gap-6">
        {/* Main Entity List */}
        <div className="lg:col-span-3 animate-fade-slide-down delay-2">
          <Card className="bg-zinc-900 border-zinc-800">
            <CardHeader className="pb-3">
              <div className="flex items-center justify-between">
                <CardTitle className="text-white text-lg">All Entities</CardTitle>
                <div className="relative w-64">
                  <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 h-4 w-4 text-zinc-500" />
                  <Input
                    placeholder="Search entities..."
                    value={searchQuery}
                    onChange={(e) => setSearchQuery(e.target.value)}
                    className="pl-9 bg-zinc-800 border-zinc-700 text-white"
                  />
                </div>
              </div>
            </CardHeader>
            <CardContent>
              {isLoading ? (
                <div className="space-y-2">
                  {[1, 2, 3, 4, 5].map((i) => (
                    <Skeleton key={i} className="h-12 w-full bg-zinc-800" />
                  ))}
                </div>
              ) : error ? (
                <p className="text-zinc-400 text-center py-8">{error}</p>
              ) : filteredEntities.length === 0 ? (
                <p className="text-zinc-400 text-center py-8">
                  {searchQuery
                    ? "No entities match your search"
                    : "No entities found. Add memories with entity references to see them here."}
                </p>
              ) : (
                <Table>
                  <TableHeader>
                    <TableRow className="border-zinc-800 hover:bg-transparent">
                      <TableHead
                        className="text-zinc-400 cursor-pointer hover:text-white"
                        onClick={() => toggleSort("name")}
                      >
                        <div className="flex items-center gap-1">
                          Entity Name
                          <ArrowUpDown className="h-4 w-4" />
                        </div>
                      </TableHead>
                      <TableHead
                        className="text-zinc-400 cursor-pointer hover:text-white w-[120px]"
                        onClick={() => toggleSort("count")}
                      >
                        <div className="flex items-center gap-1">
                          Memories
                          <ArrowUpDown className="h-4 w-4" />
                        </div>
                      </TableHead>
                      <TableHead className="w-[50px]"></TableHead>
                    </TableRow>
                  </TableHeader>
                  <TableBody>
                    {filteredEntities.map((entity) => (
                      <TableRow
                        key={entity.name}
                        className="border-zinc-800 hover:bg-zinc-800/50 cursor-pointer"
                        onClick={() => handleEntityClick(entity.name)}
                      >
                        <TableCell className="font-medium text-white">
                          <div className="flex items-center gap-2">
                            <div className="w-8 h-8 rounded-full bg-primary/20 flex items-center justify-center">
                              <span className="text-primary text-sm font-semibold">
                                {entity.name.charAt(0).toUpperCase()}
                              </span>
                            </div>
                            {entity.name}
                          </div>
                        </TableCell>
                        <TableCell>
                          <Badge
                            variant="secondary"
                            className="bg-zinc-800 text-zinc-300"
                          >
                            {entity.memory_count}
                          </Badge>
                        </TableCell>
                        <TableCell>
                          <ExternalLink className="h-4 w-4 text-zinc-500" />
                        </TableCell>
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
              )}
            </CardContent>
          </Card>
        </div>

        {/* Sidebar - Vault Distribution */}
        <div className="animate-fade-slide-down delay-3">
          <Card className="bg-zinc-900 border-zinc-800">
            <CardHeader>
              <CardTitle className="text-white text-lg">By Vault</CardTitle>
            </CardHeader>
            <CardContent>
              {isLoading ? (
                <div className="space-y-2">
                  {[1, 2, 3].map((i) => (
                    <Skeleton key={i} className="h-8 w-full bg-zinc-800" />
                  ))}
                </div>
              ) : vaultDistribution.length === 0 ? (
                <p className="text-zinc-400 text-sm text-center py-4">
                  No vault data available
                </p>
              ) : (
                <div className="space-y-2">
                  {vaultDistribution.map((vault) => (
                    <div
                      key={vault.key}
                      className="flex items-center justify-between p-2 rounded-lg bg-zinc-800/50"
                    >
                      <span className="text-zinc-200 text-sm">{vault.key}</span>
                      <Badge
                        variant="outline"
                        className="border-primary text-primary"
                      >
                        {vault.count}
                      </Badge>
                    </div>
                  ))}
                </div>
              )}
            </CardContent>
          </Card>
        </div>
      </div>
    </div>
  );
}
