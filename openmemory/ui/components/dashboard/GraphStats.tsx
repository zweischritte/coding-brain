"use client";

import { useState, useEffect } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Skeleton } from "@/components/ui/skeleton";
import { useGraphApi } from "@/hooks/useGraphApi";
import {
  Network,
  Users,
  GitBranch,
  Tag,
  Database,
  AlertCircle,
} from "lucide-react";

interface StatItemProps {
  icon: React.ReactNode;
  label: string;
  value: number | string;
  subtitle?: string;
}

function StatItem({ icon, label, value, subtitle }: StatItemProps) {
  return (
    <div className="flex items-center gap-3 p-3 rounded-lg bg-zinc-800/50">
      <div className="p-2 rounded-lg bg-zinc-700/50">{icon}</div>
      <div>
        <p className="text-xs text-zinc-400">{label}</p>
        <p className="text-lg font-semibold text-white">{value}</p>
        {subtitle && <p className="text-xs text-zinc-500">{subtitle}</p>}
      </div>
    </div>
  );
}

export function GraphStats() {
  const { getStats, getHealth, isLoading } = useGraphApi();
  const [stats, setStats] = useState<any>(null);
  const [health, setHealth] = useState<any>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const loadData = async () => {
      try {
        const [statsData, healthData] = await Promise.all([
          getStats(),
          getHealth(),
        ]);
        setStats(statsData);
        setHealth(healthData);
        setError(null);
      } catch (err) {
        setError("Failed to load graph stats");
      }
    };

    loadData();
  }, []);

  if (isLoading && !stats) {
    return (
      <Card className="bg-zinc-900 border-zinc-800 h-full">
        <CardHeader>
          <CardTitle className="text-white flex items-center gap-2 text-lg">
            <Network className="h-5 w-5" />
            Graph Statistics
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-3">
          {[1, 2, 3, 4].map((i) => (
            <Skeleton key={i} className="h-16 w-full bg-zinc-800" />
          ))}
        </CardContent>
      </Card>
    );
  }

  if (!stats?.enabled) {
    return (
      <Card className="bg-zinc-900 border-zinc-800 h-full">
        <CardHeader>
          <CardTitle className="text-white flex items-center gap-2 text-lg">
            <Network className="h-5 w-5" />
            Graph Statistics
          </CardTitle>
        </CardHeader>
        <CardContent className="flex flex-col items-center justify-center py-8 text-center">
          <AlertCircle className="h-8 w-8 text-zinc-500 mb-3" />
          <p className="text-zinc-400 text-sm">
            Graph features not available
          </p>
          <p className="text-zinc-500 text-xs mt-1">
            {stats?.message || "Neo4j is not configured"}
          </p>
        </CardContent>
      </Card>
    );
  }

  return (
    <Card className="bg-zinc-900 border-zinc-800 h-full">
      <CardHeader className="pb-3">
        <div className="flex items-center justify-between">
          <CardTitle className="text-white flex items-center gap-2 text-lg">
            <Network className="h-5 w-5 text-primary" />
            Graph Statistics
          </CardTitle>
          <div className="flex gap-1">
            {health?.neo4j_metadata_projection && (
              <Badge variant="outline" className="text-xs border-green-600 text-green-500">
                Neo4j
              </Badge>
            )}
            {health?.neo4j_gds && (
              <Badge variant="outline" className="text-xs border-blue-600 text-blue-500">
                GDS
              </Badge>
            )}
          </div>
        </div>
      </CardHeader>
      <CardContent className="grid grid-cols-2 gap-3">
        <StatItem
          icon={<Database className="h-4 w-4 text-primary" />}
          label="Memories"
          value={stats.memoryCount ?? 0}
        />
        <StatItem
          icon={<Users className="h-4 w-4 text-blue-400" />}
          label="Entities"
          value={stats.entityCount ?? 0}
        />
        <StatItem
          icon={<GitBranch className="h-4 w-4 text-green-400" />}
          label="Similarity Edges"
          value={stats.similarityEdges ?? 0}
          subtitle="Pre-computed"
        />
        <StatItem
          icon={<Tag className="h-4 w-4 text-amber-400" />}
          label="Co-Mention Edges"
          value={stats.coMentionEdges ?? 0}
          subtitle="Entity connections"
        />
      </CardContent>
    </Card>
  );
}
