"use client";

import { useState } from "react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { EditMemoryDialog } from "@/app/memories/components/EditMemoryDialog";
import { EntityNetworkGraph } from "@/components/EntityNetworkGraph";
import { StructuredCategory, MemoryScope, ArtifactType, SourceType } from "@/components/types";

// Mock data for testing - with proper typing
const MOCK_MEMORY = {
  id: "test-memory-123",
  text: "This is a test memory with structured metadata",
  created_at: "2024-01-15T10:30:00Z",
  state: "active",
  categories: ["test", "development"],
  app_name: "test-app",
  metadata: {
    category: "architecture" as StructuredCategory,
    scope: "project" as MemoryScope,
    artifact_type: "repo" as ArtifactType,
    artifact_ref: "openmemory/ui/app/test/page.tsx",
    entity: "grischa",
    source: "user" as SourceType,
    evidence: ["TEST-1"],
    tags: { demo: true },
  },
};

// Mock EntityNetwork data
const MOCK_NETWORK = {
  entity: "grischa",
  connections: [
    { entity: "julia", count: 15 },
    { entity: "project-alpha", count: 8 },
    { entity: "berlin", count: 12 },
    { entity: "el_juego", count: 6 },
    { entity: "cloudkit", count: 4 },
  ],
  total_connections: 5,
};

export default function TestPage() {
  const [editDialogOpen, setEditDialogOpen] = useState(false);
  const [debugInfo, setDebugInfo] = useState<string>("");

  const handleOpenEditDialog = () => {
    setDebugInfo(JSON.stringify(MOCK_MEMORY.metadata, null, 2));
    setEditDialogOpen(true);
  };

  return (
    <div className="container mx-auto py-8 px-4">
      <h1 className="text-2xl font-bold text-white mb-6">UI Component Test Page</h1>

      {/* Debug Info */}
      <Card className="bg-zinc-900 border-zinc-800 mb-6">
        <CardHeader>
          <CardTitle className="text-white">Debug Info</CardTitle>
        </CardHeader>
        <CardContent>
          <pre className="text-green-400 text-sm bg-zinc-950 p-4 rounded-lg overflow-auto">
            {debugInfo || "Click a button to see debug info"}
          </pre>
        </CardContent>
      </Card>

      {/* Test EditMemoryDialog */}
      <Card className="bg-zinc-900 border-zinc-800 mb-6">
        <CardHeader>
          <CardTitle className="text-white">Test EditMemoryDialog</CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          <p className="text-zinc-400 text-sm">
            Mock Memory Metadata: category={MOCK_MEMORY.metadata.category},
            scope={MOCK_MEMORY.metadata.scope},
            artifact_type={MOCK_MEMORY.metadata.artifact_type},
            artifact_ref={MOCK_MEMORY.metadata.artifact_ref},
            entity={MOCK_MEMORY.metadata.entity}
          </p>
          <Button onClick={handleOpenEditDialog}>
            Open Edit Dialog with Mock Data
          </Button>

          <EditMemoryDialog
            open={editDialogOpen}
            onOpenChange={setEditDialogOpen}
            memoryId={MOCK_MEMORY.id}
            initialContent={MOCK_MEMORY.text}
            initialMetadata={MOCK_MEMORY.metadata}
            onSuccess={() => {
              console.log("Edit success!");
              setDebugInfo("Edit dialog closed successfully");
            }}
          />
        </CardContent>
      </Card>

      {/* Test EntityNetworkGraph with Mock Data */}
      <Card className="bg-zinc-900 border-zinc-800 mb-6">
        <CardHeader>
          <CardTitle className="text-white">Test EntityNetworkGraph (Mock)</CardTitle>
        </CardHeader>
        <CardContent>
          <p className="text-zinc-400 text-sm mb-4">
            Mock Network: {MOCK_NETWORK.connections.length} connections for entity "{MOCK_NETWORK.entity}"
          </p>
          <MockEntityNetworkGraph
            entityName={MOCK_NETWORK.entity}
            mockData={MOCK_NETWORK}
          />
        </CardContent>
      </Card>

      {/* Test Real EntityNetworkGraph with API */}
      <Card className="bg-zinc-900 border-zinc-800 mb-6">
        <CardHeader>
          <CardTitle className="text-white">Test EntityNetworkGraph (Real API)</CardTitle>
        </CardHeader>
        <CardContent>
          <p className="text-zinc-400 text-sm mb-4">
            Using real API to fetch network for entity "grischa"
          </p>
          <EntityNetworkGraph
            entityName="grischa"
            onEntityClick={(name) => {
              setDebugInfo(`Clicked entity: ${name}`);
            }}
          />
        </CardContent>
      </Card>
    </div>
  );
}

// Inline Mock EntityNetworkGraph to test without API
function MockEntityNetworkGraph({
  entityName,
  mockData
}: {
  entityName: string;
  mockData: typeof MOCK_NETWORK;
}) {
  const [containerWidth, setContainerWidth] = useState(0);
  const containerRef = useState<HTMLDivElement | null>(null);

  // Simple visualization without react-force-graph
  return (
    <div className="bg-zinc-950 rounded-lg p-4" style={{ minHeight: 400 }}>
      <div className="flex flex-col items-center justify-center h-full">
        {/* Central node */}
        <div className="relative" style={{ width: 300, height: 300 }}>
          {/* Center entity */}
          <div
            className="absolute bg-purple-500 rounded-full flex items-center justify-center text-white text-sm font-bold"
            style={{
              width: 60,
              height: 60,
              left: '50%',
              top: '50%',
              transform: 'translate(-50%, -50%)'
            }}
          >
            {entityName}
          </div>

          {/* Connected entities in a circle */}
          {mockData.connections.map((conn, i) => {
            const angle = (i / mockData.connections.length) * 2 * Math.PI - Math.PI / 2;
            const radius = 120;
            const x = 150 + radius * Math.cos(angle);
            const y = 150 + radius * Math.sin(angle);
            const size = Math.min(50, 30 + conn.count * 2);

            return (
              <div key={conn.entity}>
                {/* Connection line */}
                <svg
                  className="absolute"
                  style={{ width: 300, height: 300, left: 0, top: 0, pointerEvents: 'none' }}
                >
                  <line
                    x1={150}
                    y1={150}
                    x2={x}
                    y2={y}
                    stroke="rgba(100,100,100,0.5)"
                    strokeWidth={Math.min(4, 1 + conn.count * 0.3)}
                  />
                </svg>
                {/* Entity node */}
                <div
                  className="absolute bg-blue-500 rounded-full flex items-center justify-center text-white text-xs cursor-pointer hover:bg-blue-400 transition-colors"
                  style={{
                    width: size,
                    height: size,
                    left: x - size/2,
                    top: y - size/2,
                  }}
                  title={`${conn.entity}: ${conn.count} connections`}
                >
                  {conn.entity.substring(0, 6)}
                </div>
              </div>
            );
          })}
        </div>

        {/* Legend */}
        <div className="mt-4 text-zinc-400 text-sm">
          <span className="inline-block w-3 h-3 rounded-full bg-purple-500 mr-2"></span>
          Center Entity
          <span className="inline-block w-3 h-3 rounded-full bg-blue-500 ml-4 mr-2"></span>
          Connected Entities ({mockData.connections.length})
        </div>
      </div>
    </div>
  );
}
