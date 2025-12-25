"use client";

import { useState, useEffect } from "react";
import { useParams, useRouter } from "next/navigation";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
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
import { EntityNetworkGraph } from "@/components/EntityNetworkGraph";
import { EntityRelation } from "@/components/types";
import { cn } from "@/lib/utils";
import {
  ArrowLeft,
  User,
  Link,
  FileText,
  ArrowRight,
  ArrowLeftRight,
  ExternalLink,
} from "lucide-react";

interface EntityDetailResponse {
  name: string;
  memory_count: number;
  network: {
    entity: string;
    connections: { entity: string; count: number; memoryIds?: string[] }[];
    total_connections: number;
  } | null;
  relations: EntityRelation[];
}

export default function EntityPage() {
  const params = useParams();
  const router = useRouter();
  const entityName = decodeURIComponent(params.name as string);

  const { getEntity, getEntityMemories, isLoading } = useEntitiesApi();
  const [entityData, setEntityData] = useState<EntityDetailResponse | null>(null);
  const [memories, setMemories] = useState<any[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [isNetworkExpanded, setIsNetworkExpanded] = useState(false);

  useEffect(() => {
    const loadEntityData = async () => {
      try {
        const [entity, entityMemories] = await Promise.all([
          getEntity(entityName),
          getEntityMemories(entityName, 20),
        ]);
        setEntityData(entity);
        setMemories(entityMemories);
        setError(null);
      } catch (err) {
        setError("Failed to load entity data");
      }
    };

    if (entityName) {
      loadEntityData();
    }
  }, [entityName]);

  const handleEntityClick = (name: string) => {
    router.push(`/entity/${encodeURIComponent(name)}`);
  };

  const handleMemoryClick = (id: string) => {
    router.push(`/memory/${id}`);
  };

  if (isLoading && !entityData) {
    return (
      <div className="container mx-auto px-4 py-8">
        <div className="mb-6">
          <Skeleton className="h-8 w-48 bg-zinc-800" />
        </div>
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          <Skeleton className="h-[400px] bg-zinc-800" />
          <Skeleton className="h-[400px] bg-zinc-800" />
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="container mx-auto px-4 py-8">
        <div className="text-center py-12">
          <p className="text-zinc-400 mb-4">{error}</p>
          <Button onClick={() => router.back()} variant="outline">
            <ArrowLeft className="mr-2 h-4 w-4" />
            Go Back
          </Button>
        </div>
      </div>
    );
  }

  return (
    <div className="container mx-auto px-4 py-8">
      {/* Header */}
      <div className="flex items-center gap-4 mb-6">
        <Button
          variant="ghost"
          size="sm"
          onClick={() => router.back()}
          className="text-zinc-400 hover:text-white"
        >
          <ArrowLeft className="h-4 w-4 mr-2" />
          Back
        </Button>
      <div className="flex items-center gap-3">
        <User className="h-6 w-6 text-primary" />
        <h1 className="text-2xl font-bold text-white">{entityName}</h1>
        {entityData && (
            <Badge variant="secondary" className="bg-zinc-800 text-zinc-300">
              {entityData.memory_count} memories
            </Badge>
          )}
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Network Graph */}
        <EntityNetworkGraph
          entityName={entityName}
          onEntityClick={handleEntityClick}
          limit={30}
          expanded={isNetworkExpanded}
          onExpandedChange={setIsNetworkExpanded}
        />

        {/* Relations Card */}
        <Card className={cn("bg-zinc-900 border-zinc-800", isNetworkExpanded && "lg:col-span-2")}>
          <CardHeader>
            <CardTitle className="text-white flex items-center gap-2 text-lg">
              <Link className="h-5 w-5" />
              Relations
              {entityData?.relations && (
                <Badge variant="secondary" className="ml-2 bg-zinc-800 text-zinc-300">
                  {entityData.relations.length}
                </Badge>
              )}
            </CardTitle>
          </CardHeader>
          <CardContent>
            {entityData?.relations && entityData.relations.length > 0 ? (
              <div className="space-y-2 max-h-[300px] overflow-y-auto">
                {entityData.relations.map((rel, idx) => (
                  <div
                    key={idx}
                    className="flex items-center gap-2 p-2 rounded-lg bg-zinc-800/50 hover:bg-zinc-800 transition-colors cursor-pointer"
                    onClick={() => handleEntityClick(rel.target)}
                  >
                    {rel.direction === "outgoing" ? (
                      <ArrowRight className="h-4 w-4 text-green-500" />
                    ) : rel.direction === "incoming" ? (
                      <ArrowLeft className="h-4 w-4 text-blue-500" />
                    ) : (
                      <ArrowLeftRight className="h-4 w-4 text-zinc-400" />
                    )}
                    <Badge
                      variant="outline"
                      className="text-xs border-zinc-600 text-zinc-300"
                    >
                      {rel.type}
                    </Badge>
                    <span className="text-zinc-200 text-sm flex-1">
                      {rel.target}
                    </span>
                    <Badge
                      variant="secondary"
                      className="text-xs bg-zinc-700 text-zinc-300"
                    >
                      {rel.count}x
                    </Badge>
                  </div>
                ))}
              </div>
            ) : (
              <p className="text-zinc-400 text-sm text-center py-8">
                No typed relations found for this entity.
              </p>
            )}
          </CardContent>
        </Card>
      </div>

      {/* Related Memories */}
      <Card className="mt-6 bg-zinc-900 border-zinc-800">
        <CardHeader>
          <CardTitle className="text-white flex items-center gap-2 text-lg">
            <FileText className="h-5 w-5" />
            Related Memories
            <Badge variant="secondary" className="ml-2 bg-zinc-800 text-zinc-300">
              {memories.length}
            </Badge>
          </CardTitle>
        </CardHeader>
        <CardContent>
          {memories.length > 0 ? (
            <Table>
              <TableHeader>
                <TableRow className="border-zinc-800 hover:bg-transparent">
                  <TableHead className="text-zinc-400">Content</TableHead>
                  <TableHead className="text-zinc-400 w-[100px]">Vault</TableHead>
                  <TableHead className="text-zinc-400 w-[100px]">Layer</TableHead>
                  <TableHead className="text-zinc-400 w-[100px]">Entities</TableHead>
                  <TableHead className="w-[50px]"></TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {memories.map((memory) => (
                  <TableRow
                    key={memory.id}
                    className="border-zinc-800 hover:bg-zinc-800/50 cursor-pointer"
                    onClick={() => handleMemoryClick(memory.id)}
                  >
                    <TableCell className="text-zinc-200">
                      <p className="line-clamp-2">{memory.content}</p>
                    </TableCell>
                    <TableCell>
                      {memory.vault && (
                        <Badge
                          variant="secondary"
                          className="bg-zinc-800 text-zinc-300 border border-zinc-700"
                        >
                          {memory.vault}
                        </Badge>
                      )}
                    </TableCell>
                    <TableCell>
                      {memory.layer && (
                        <Badge
                          variant="secondary"
                          className="bg-zinc-800 text-zinc-300 border border-zinc-700"
                        >
                          {memory.layer}
                        </Badge>
                      )}
                    </TableCell>
                    <TableCell>
                      <Badge
                        variant="outline"
                        className="border-primary text-primary"
                      >
                        {memory.matchedEntities}
                      </Badge>
                    </TableCell>
                    <TableCell>
                      <ExternalLink className="h-4 w-4 text-zinc-500" />
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          ) : (
            <p className="text-zinc-400 text-sm text-center py-8">
              No memories found for this entity.
            </p>
          )}
        </CardContent>
      </Card>
    </div>
  );
}
