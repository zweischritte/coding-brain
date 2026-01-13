"use client";

import { useEffect, useRef, useState, useCallback } from "react";
import dynamic from "next/dynamic";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Skeleton } from "@/components/ui/skeleton";
import { useEntitiesApi } from "@/hooks/useEntitiesApi";
import { EntityNetwork, EntityRelation } from "@/components/types";
import { cn } from "@/lib/utils";
import { Network, Maximize2, Minimize2, RefreshCw } from "lucide-react";

// Dynamically import ForceGraph2D to avoid SSR issues
const ForceGraph2D = dynamic(() => import("react-force-graph-2d"), {
  ssr: false,
  loading: () => (
    <div className="flex items-center justify-center h-full">
      <Skeleton className="h-64 w-full bg-zinc-800" />
    </div>
  ),
});

// Custom hook to track container dimensions
function useContainerDimensions(element: HTMLDivElement | null) {
  const [dimensions, setDimensions] = useState({ width: 0, height: 0 });

  useEffect(() => {
    if (!element) return;

    const updateDimensions = () => {
      const { width, height } = element.getBoundingClientRect();
      setDimensions({ width: Math.round(width), height: Math.round(height) });
    };

    updateDimensions();

    if (typeof ResizeObserver !== "undefined") {
      const resizeObserver = new ResizeObserver(updateDimensions);
      resizeObserver.observe(element);
      return () => resizeObserver.disconnect();
    }

    // Fallback for environments without ResizeObserver
    window.addEventListener("resize", updateDimensions);
    return () => window.removeEventListener("resize", updateDimensions);
  }, [element]);

  return dimensions;
}

interface EntityNetworkGraphProps {
  entityName: string;
  onEntityClick?: (entityName: string) => void;
  minCount?: number;
  limit?: number;
  height?: number;
  expanded?: boolean;
  onExpandedChange?: (expanded: boolean) => void;
}

interface GraphNode {
  id: string;
  name: string;
  val: number;
  color: string;
  isCenter: boolean;
}

interface GraphLink {
  source: string;
  target: string;
  value: number;
  type: "co_mention" | "relation";
  relationType?: string; // For typed relations (e.g., "works_at", "owns")
}

interface GraphData {
  nodes: GraphNode[];
  links: GraphLink[];
}

const NODE_COLORS = {
  center: "#a855f7", // Purple for central entity
  connected: "#3b82f6", // Blue for connected entities
  hover: "#f59e0b", // Amber for hover
};

const LINK_COLORS = {
  co_mention: "rgba(100, 100, 100, 0.5)", // Gray for co-mentions
  relation: "rgba(34, 197, 94, 0.7)", // Green for typed relations
};

export function EntityNetworkGraph({
  entityName,
  onEntityClick,
  minCount = 1,
  limit = 30,
  height = 400,
  expanded,
  onExpandedChange,
}: EntityNetworkGraphProps) {
  const { getEntityNetwork, getEntityRelations } = useEntitiesApi();
  const [network, setNetwork] = useState<EntityNetwork | null>(null);
  const [relations, setRelations] = useState<EntityRelation[]>([]);
  const [graphData, setGraphData] = useState<GraphData>({ nodes: [], links: [] });
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true); // Start with loading=true
  const [uncontrolledExpanded, setUncontrolledExpanded] = useState(false);
  const isExpanded = expanded ?? uncontrolledExpanded;
  const setIsExpanded = onExpandedChange ?? setUncontrolledExpanded;
  const [hoveredNode, setHoveredNode] = useState<string | null>(null);

  const graphRef = useRef<any>(null);
  const [containerElement, setContainerElement] = useState<HTMLDivElement | null>(null);
  const { width: containerWidth } = useContainerDimensions(containerElement);
  const cardClassName = cn("bg-zinc-900 border-zinc-800", isExpanded && "lg:col-span-2");

  // Store API functions in refs to avoid re-creating loadNetwork
  const getEntityNetworkRef = useRef(getEntityNetwork);
  getEntityNetworkRef.current = getEntityNetwork;
  const getEntityRelationsRef = useRef(getEntityRelations);
  getEntityRelationsRef.current = getEntityRelations;

  // Load network data (co-mentions + typed relations)
  const loadNetwork = useCallback(async () => {
    if (!entityName) return;

    setLoading(true);
    setError(null);

    try {
      // Fetch both co-mentions and typed relations in parallel
      const [networkData, relationsData] = await Promise.all([
        getEntityNetworkRef.current(entityName, minCount, limit),
        getEntityRelationsRef.current(entityName, { direction: "both", limit }),
      ]);

      setNetwork(networkData);
      setRelations(relationsData);

      // Build graph data from both sources
      const nodes: GraphNode[] = [
        {
          id: entityName,
          name: entityName,
          val: 20, // Larger central node
          color: NODE_COLORS.center,
          isCenter: true,
        },
      ];

      const links: GraphLink[] = [];
      const nodeSet = new Set<string>([entityName]);

      // Add nodes and links from co-mentions
      if (networkData && networkData.connections) {
        networkData.connections.forEach((conn) => {
          if (!nodeSet.has(conn.entity)) {
            nodes.push({
              id: conn.entity,
              name: conn.entity,
              val: Math.min(15, 5 + conn.count * 2), // Size based on connection count
              color: NODE_COLORS.connected,
              isCenter: false,
            });
            nodeSet.add(conn.entity);
          }

          links.push({
            source: entityName,
            target: conn.entity,
            value: conn.count,
            type: "co_mention",
          });
        });
      }

      // Add nodes and links from typed relations
      if (relationsData && relationsData.length > 0) {
        relationsData.forEach((rel) => {
          if (!nodeSet.has(rel.target)) {
            nodes.push({
              id: rel.target,
              name: rel.target,
              val: Math.min(15, 5 + rel.count * 2),
              color: NODE_COLORS.connected,
              isCenter: false,
            });
            nodeSet.add(rel.target);
          }

          // Determine link direction based on relation direction
          const source = rel.direction === "outgoing" ? entityName : rel.target;
          const target = rel.direction === "outgoing" ? rel.target : entityName;

          links.push({
            source,
            target,
            value: rel.count,
            type: "relation",
            relationType: rel.type,
          });
        });
      }

      setGraphData({ nodes, links });
    } catch (err) {
      setError("Failed to load entity network");
      setNetwork(null);
      setRelations([]);
      setGraphData({ nodes: [], links: [] });
    } finally {
      setLoading(false);
    }
  }, [entityName, minCount, limit]);

  useEffect(() => {
    loadNetwork();
  }, [loadNetwork]);

  // Center graph when data or size changes
  useEffect(() => {
    if (!graphRef.current || graphData.nodes.length === 0 || containerWidth <= 0) return;
    const timeout = setTimeout(() => {
      graphRef.current?.zoomToFit(400, 50);
    }, 250);
    return () => clearTimeout(timeout);
  }, [graphData, containerWidth, isExpanded]);

  const handleNodeClick = (node: any) => {
    if (onEntityClick && node?.name) {
      onEntityClick(String(node.name));
    }
  };

  const handleNodeHover = (node: any | null) => {
    setHoveredNode(node?.id ? String(node.id) : null);
  };

  const nodeCanvasObject = useCallback(
    (node: any, ctx: CanvasRenderingContext2D, globalScale: number) => {
      const label = node.name;
      const fontSize = 12 / globalScale;
      ctx.font = `${fontSize}px Inter, sans-serif`;

      // Draw node circle
      const radius = Math.sqrt(node.val) * 1.5;
      ctx.beginPath();
      ctx.arc(node.x, node.y, radius, 0, 2 * Math.PI);
      ctx.fillStyle =
        hoveredNode === node.id
          ? NODE_COLORS.hover
          : node.isCenter
          ? NODE_COLORS.center
          : NODE_COLORS.connected;
      ctx.fill();

      // Draw label
      ctx.textAlign = "center";
      ctx.textBaseline = "middle";
      ctx.fillStyle = "#fff";
      ctx.fillText(label, node.x, node.y + radius + fontSize + 2);
    },
    [hoveredNode]
  );

  if (loading) {
    return (
      <Card className={cardClassName}>
        <CardHeader>
          <CardTitle className="text-white flex items-center gap-2 text-lg">
            <Network className="h-5 w-5" />
            Entity Network
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div
            ref={setContainerElement}
            className="bg-zinc-950 rounded-lg overflow-hidden"
            style={{ height: isExpanded ? 600 : height }}
          >
            <Skeleton className="h-full w-full bg-zinc-800" />
          </div>
        </CardContent>
      </Card>
    );
  }

  if (error) {
    return (
      <Card className={cardClassName}>
        <CardHeader>
          <CardTitle className="text-white flex items-center gap-2 text-lg">
            <Network className="h-5 w-5" />
            Entity Network
          </CardTitle>
        </CardHeader>
        <CardContent className="flex flex-col items-center gap-4">
          <p className="text-zinc-400 text-sm">{error}</p>
          <Button
            variant="outline"
            size="sm"
            onClick={loadNetwork}
            className="border-zinc-700"
          >
            <RefreshCw className="h-4 w-4 mr-2" />
            Retry
          </Button>
        </CardContent>
      </Card>
    );
  }

  if (graphData.nodes.length === 0) {
    return (
      <Card className={cardClassName}>
        <CardHeader>
          <CardTitle className="text-white flex items-center gap-2 text-lg">
            <Network className="h-5 w-5" />
            Entity Network
          </CardTitle>
        </CardHeader>
        <CardContent>
          <p className="text-zinc-400 text-sm text-center py-8">
            No connections found for this entity.
          </p>
        </CardContent>
      </Card>
    );
  }

  return (
    <Card className={cardClassName}>
      <CardHeader className="flex flex-row items-center justify-between">
        <CardTitle className="text-white flex items-center gap-2 text-lg">
          <Network className="h-5 w-5" />
          Entity Network
          <Badge variant="secondary" className="ml-2 bg-zinc-800 text-zinc-300">
            {graphData.links.filter(l => l.type === "co_mention").length} co-mentions
          </Badge>
          {graphData.links.filter(l => l.type === "relation").length > 0 && (
            <Badge variant="secondary" className="bg-green-900/50 text-green-300 border border-green-700/50">
              {graphData.links.filter(l => l.type === "relation").length} relations
            </Badge>
          )}
        </CardTitle>
        <div className="flex gap-2">
          <Button
            variant="ghost"
            size="sm"
            onClick={loadNetwork}
            disabled={loading}
            className="h-8 w-8 p-0"
          >
            <RefreshCw className={`h-4 w-4 ${loading ? "animate-spin" : ""}`} />
          </Button>
          <Button
            variant="ghost"
            size="sm"
            onClick={() => setIsExpanded(!isExpanded)}
            className="h-8 w-8 p-0"
            aria-label={isExpanded ? "Minimize entity network" : "Maximize entity network"}
          >
            {isExpanded ? (
              <Minimize2 className="h-4 w-4" />
            ) : (
              <Maximize2 className="h-4 w-4" />
            )}
          </Button>
        </div>
      </CardHeader>
      <CardContent>
        <div
          ref={setContainerElement}
          className="bg-zinc-950 rounded-lg overflow-hidden"
          style={{ height: isExpanded ? 600 : height }}
        >
          {containerWidth > 0 ? (
            <ForceGraph2D
              ref={graphRef}
              graphData={graphData}
              nodeLabel={(node: any) => `${node.name}`}
              nodeCanvasObject={nodeCanvasObject}
              nodePointerAreaPaint={(node: any, color, ctx) => {
                const radius = Math.sqrt(node.val) * 1.5;
                ctx.beginPath();
                ctx.arc(node.x, node.y, radius + 5, 0, 2 * Math.PI);
                ctx.fillStyle = color;
                ctx.fill();
              }}
              onNodeClick={handleNodeClick}
              onNodeHover={handleNodeHover}
              linkWidth={(link: any) => Math.min(5, 1 + link.value * 0.5)}
              linkColor={(link: any) => LINK_COLORS[link.type as keyof typeof LINK_COLORS] || LINK_COLORS.co_mention}
              linkLabel={(link: any) => link.relationType ? `${link.relationType} (${link.value}x)` : `co-mention (${link.value}x)`}
              linkDirectionalArrowLength={(link: any) => link.type === "relation" ? 6 : 0}
              linkDirectionalArrowRelPos={1}
              backgroundColor="#09090b"
              width={containerWidth}
              height={isExpanded ? 600 : height}
              enableZoomInteraction={true}
              enablePanInteraction={true}
            />
          ) : (
            <div className="flex items-center justify-center h-full">
              <Skeleton className="h-full w-full bg-zinc-800" />
            </div>
          )}
        </div>
      </CardContent>
    </Card>
  );
}
