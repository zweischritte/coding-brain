"use client";

import { useState, useEffect } from "react";
import { useRouter } from "next/navigation";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Skeleton } from "@/components/ui/skeleton";
import { useMemoriesApi } from "@/hooks/useMemoriesApi";
import { SimilarMemory } from "@/components/types";
import { GitBranch, ExternalLink } from "lucide-react";

interface SimilarMemoriesProps {
  memoryId: string;
  minScore?: number;
  limit?: number;
}

export function SimilarMemories({
  memoryId,
  minScore = 0.5,
  limit = 5,
}: SimilarMemoriesProps) {
  const router = useRouter();
  const { fetchSimilarMemories, isLoading } = useMemoriesApi();
  const [similarMemories, setSimilarMemories] = useState<SimilarMemory[]>([]);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const loadSimilarMemories = async () => {
      if (!memoryId) return;

      try {
        const memories = await fetchSimilarMemories(memoryId, minScore, limit);
        setSimilarMemories(memories);
        setError(null);
      } catch (err) {
        setError("Failed to load similar memories");
        setSimilarMemories([]);
      }
    };

    loadSimilarMemories();
  }, [memoryId, minScore, limit]);

  const handleMemoryClick = (id: string) => {
    router.push(`/memory/${id}`);
  };

  if (isLoading) {
    return (
      <Card className="bg-zinc-900 border-zinc-800">
        <CardHeader>
          <CardTitle className="text-white flex items-center gap-2 text-lg">
            <GitBranch className="h-5 w-5" />
            Similar Memories
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-3">
          {[1, 2, 3].map((i) => (
            <div key={i} className="space-y-2">
              <Skeleton className="h-4 w-full bg-zinc-800" />
              <Skeleton className="h-4 w-3/4 bg-zinc-800" />
            </div>
          ))}
        </CardContent>
      </Card>
    );
  }

  if (error) {
    return (
      <Card className="bg-zinc-900 border-zinc-800">
        <CardHeader>
          <CardTitle className="text-white flex items-center gap-2 text-lg">
            <GitBranch className="h-5 w-5" />
            Similar Memories
          </CardTitle>
        </CardHeader>
        <CardContent>
          <p className="text-zinc-400 text-sm">{error}</p>
        </CardContent>
      </Card>
    );
  }

  if (similarMemories.length === 0) {
    return (
      <Card className="bg-zinc-900 border-zinc-800">
        <CardHeader>
          <CardTitle className="text-white flex items-center gap-2 text-lg">
            <GitBranch className="h-5 w-5" />
            Similar Memories
          </CardTitle>
        </CardHeader>
        <CardContent>
          <p className="text-zinc-400 text-sm">
            No similar memories found. Try adding more memories to build connections.
          </p>
        </CardContent>
      </Card>
    );
  }

  return (
    <Card className="bg-zinc-900 border-zinc-800">
      <CardHeader>
        <CardTitle className="text-white flex items-center gap-2 text-lg">
          <GitBranch className="h-5 w-5" />
          Similar Memories
          <Badge variant="secondary" className="ml-auto bg-zinc-800 text-zinc-300">
            {similarMemories.length}
          </Badge>
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-3">
        {similarMemories.map((memory) => (
          <div
            key={memory.id}
            className="p-3 rounded-lg bg-zinc-800/50 hover:bg-zinc-800 transition-colors cursor-pointer group"
            onClick={() => handleMemoryClick(memory.id)}
          >
            <div className="flex items-start justify-between gap-2">
              <p className="text-sm text-zinc-200 line-clamp-2 flex-1">
                {memory.content}
              </p>
              <ExternalLink className="h-4 w-4 text-zinc-500 group-hover:text-zinc-300 flex-shrink-0 mt-0.5" />
            </div>
            <div className="flex items-center gap-2 mt-2">
              <Badge
                variant="outline"
                className="text-xs bg-zinc-900 border-zinc-700 text-primary"
              >
                {Math.round(memory.similarity_score * 100)}% match
              </Badge>
              {memory.category && (
                <Badge
                  variant="secondary"
                  className="text-xs bg-zinc-700 text-zinc-300"
                >
                  {memory.category}
                </Badge>
              )}
              {memory.scope && (
                <Badge
                  variant="secondary"
                  className="text-xs bg-zinc-700 text-zinc-300"
                >
                  {memory.scope}
                </Badge>
              )}
            </div>
          </div>
        ))}
      </CardContent>
    </Card>
  );
}
