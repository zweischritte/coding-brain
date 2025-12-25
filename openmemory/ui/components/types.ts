export type Category = "personal" | "work" | "health" | "finance" | "travel" | "education" | "preferences" | "relationships"
export type Client = "chrome" | "chatgpt" | "cursor" | "windsurf" | "terminal" | "api"

// AXIS Protocol Types
export type Vault = "SOV" | "WLT" | "SIG" | "FRC" | "DIR" | "FGP" | "Q"
export type Layer = "somatic" | "emotional" | "narrative" | "cognitive" | "values" | "identity" | "relational" | "goals" | "resources" | "context" | "temporal" | "meta"
export type Vector = "say" | "want" | "do"
export type Circuit = 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8

export interface MemoryMetadata {
  vault?: Vault
  layer?: Layer
  circuit?: Circuit
  vector?: Vector
  re?: string  // entity reference
  tags?: Record<string, any>
  [key: string]: any  // Allow other metadata
}

export interface Memory {
  id: string
  memory: string
  metadata: MemoryMetadata
  client: Client
  categories: Category[]
  created_at: number
  updated_at?: number
  app_name: string
  state: "active" | "paused" | "archived" | "deleted"
}

// Entity Types
export interface Entity {
  name: string
  memory_count: number
  connections?: number
  mentionCount?: number
}

export interface EntityConnection {
  entity: string
  count: number
  memoryIds?: string[]
}

export interface EntityNetwork {
  entity: string
  connections: EntityConnection[]
  total_connections: number
}

export interface EntityRelation {
  target: string
  type: string
  direction: "outgoing" | "incoming"
  memory_id?: string
  count: number
}

// Graph Types
export interface GraphStats {
  enabled: boolean
  memoryCount?: number
  entityCount?: number
  coMentionEdges?: number
  similarityEdges?: number
  tagCooccurEdges?: number
  totalEdges?: number
  message?: string
}

export interface AggregationBucket {
  key: string
  count: number
}

export interface TagPair {
  tag1: string
  tag2: string
  count: number
  exampleMemoryIds?: string[]
}

// Similar Memory Types
export interface SimilarMemory {
  id: string
  content: string
  similarity_score: number
  rank: number
  vault?: Vault
  layer?: Layer
  circuit?: Circuit
  vector?: Vector
  created_at?: string
  updated_at?: string
}

// Subgraph Types
export interface GraphNode {
  id: string
  label: string
  memory_id?: string
  content?: string
  value?: string | number
  vault?: Vault
  layer?: Layer
  vector?: Vector
  circuit?: Circuit
  created_at?: string
  updated_at?: string
  shared_count?: number
}

export interface GraphEdge {
  source: string
  target: string
  type: string
  value?: string | number
}

export interface MemorySubgraph {
  seed_memory_id: string
  seed?: Record<string, any>
  nodes: GraphNode[]
  edges: GraphEdge[]
  relations?: any[]
  related?: any[]
}

// Update Request Type
export interface MemoryUpdateRequest {
  user_id: string
  memory_content?: string
  vault?: Vault
  layer?: Layer
  circuit?: Circuit
  vector?: Vector
  entity?: string
  tags?: Record<string, any>
}