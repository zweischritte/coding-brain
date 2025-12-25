"use client";

import { useState, useEffect } from "react";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Button } from "@/components/ui/button";
import { Label } from "@/components/ui/label";
import { Textarea } from "@/components/ui/textarea";
import { Input } from "@/components/ui/input";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { useToast } from "@/hooks/use-toast";
import { useMemoriesApi } from "@/hooks/useMemoriesApi";
import { Vault, Layer, Vector, Circuit } from "@/components/types";

interface EditMemoryDialogProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  memoryId: string;
  initialContent: string;
  initialMetadata?: {
    // API may return full vault names (e.g. "WEALTH_MATRIX") rather than short codes (e.g. "WLT")
    vault?: string;
    layer?: Layer;
    circuit?: Circuit;
    vector?: Vector;
    re?: string;
    tags?: Record<string, any>;
  };
  onSuccess?: () => void;
}

const VAULT_OPTIONS: { value: Vault; label: string }[] = [
  { value: "SOV", label: "SOV - Sovereignty Core" },
  { value: "WLT", label: "WLT - Wealth & Work" },
  { value: "SIG", label: "SIG - Signal Processing" },
  { value: "FRC", label: "FRC - Force Health" },
  { value: "DIR", label: "DIR - Direction System" },
  { value: "FGP", label: "FGP - Fingerprint Evolution" },
  { value: "Q", label: "Q - Questions" },
];

// Map full vault names (from API) to short codes (for UI)
const VAULT_NAME_TO_CODE: Record<string, Vault> = {
  // AXIS 3.4 (backend)
  "SOVEREIGNTY_CORE": "SOV",
  "WEALTH_MATRIX": "WLT",
  "SIGNAL_LIBRARY": "SIG",
  "FRACTURE_LOG": "FRC",
  "SOURCE_DIRECTIVES": "DIR",
  "FINGERPRINT": "FGP",
  "QUESTIONS_QUEUE": "Q",

  // Legacy / alternate names
  "WEALTH_WORK": "WLT",
  "SIGNAL_PROCESSING": "SIG",
  "FORCE_HEALTH": "FRC",
  "DIRECTION_SYSTEM": "DIR",
  "FINGERPRINT_EVOLUTION": "FGP",
  "QUESTIONS": "Q",

  // Also support short codes directly
  "SOV": "SOV",
  "WLT": "WLT",
  "SIG": "SIG",
  "FRC": "FRC",
  "DIR": "DIR",
  "FGP": "FGP",
  "Q": "Q",
};

function normalizeVault(vault: string | undefined): Vault | undefined {
  if (!vault) return undefined;
  const normalized = vault
    .trim()
    .toUpperCase()
    .replace(/[^A-Z0-9]+/g, "_")
    .replace(/^_+|_+$/g, "");
  return VAULT_NAME_TO_CODE[normalized];
}

const LAYER_OPTIONS: { value: Layer; label: string }[] = [
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

const VECTOR_OPTIONS: { value: Vector; label: string }[] = [
  { value: "say", label: "Say - What I express" },
  { value: "want", label: "Want - What I desire" },
  { value: "do", label: "Do - What I act on" },
];

const CIRCUIT_OPTIONS: { value: Circuit; label: string }[] = [
  { value: 1, label: "1 - Survival" },
  { value: 2, label: "2 - Emotional-Territorial" },
  { value: 3, label: "3 - Semantic-Symbolic" },
  { value: 4, label: "4 - Social-Sexual" },
  { value: 5, label: "5 - Neurosomatic" },
  { value: 6, label: "6 - Neuroelectric" },
  { value: 7, label: "7 - Neurogenetic" },
  { value: 8, label: "8 - Quantum" },
];

export function EditMemoryDialog({
  open,
  onOpenChange,
  memoryId,
  initialContent,
  initialMetadata,
  onSuccess,
}: EditMemoryDialogProps) {
  const { toast } = useToast();
  const { updateMemoryWithMetadata, isLoading } = useMemoriesApi();

  // Form state - initialize empty, useEffect will populate
  const [content, setContent] = useState("");
  const [vault, setVault] = useState<Vault | undefined>(undefined);
  const [layer, setLayer] = useState<Layer | undefined>(undefined);
  const [circuit, setCircuit] = useState<Circuit | undefined>(undefined);
  const [vector, setVector] = useState<Vector | undefined>(undefined);
  const [entity, setEntity] = useState<string>("");

  // Reset form when dialog opens or when initialMetadata changes
  useEffect(() => {
    if (open) {
      // Normalize vault from API format (e.g., "SOVEREIGNTY_CORE") to UI format (e.g., "SOV")
      const vaultCandidate =
        initialMetadata?.vault ||
        (initialMetadata as any)?.vault_full ||
        (initialMetadata as any)?.vaultFull;
      const normalizedVault = normalizeVault(vaultCandidate);
      setContent(initialContent || "");
      setVault(normalizedVault);
      setLayer(initialMetadata?.layer);
      setCircuit(initialMetadata?.circuit);
      setVector(initialMetadata?.vector);
      setEntity(initialMetadata?.re || "");
    }
  }, [open, initialContent, initialMetadata]);

  const handleSave = async () => {
    try {
      const updates: {
        content?: string;
        vault?: Vault;
        layer?: Layer;
        circuit?: Circuit;
        vector?: Vector;
        entity?: string;
      } = {};

      // Only include changed fields
      if (content !== initialContent) {
        updates.content = content;
      }
      // Compare with normalized vault value
      const normalizedInitialVault = normalizeVault(initialMetadata?.vault);
      if (vault !== normalizedInitialVault) {
        updates.vault = vault;
      }
      if (layer !== initialMetadata?.layer) {
        updates.layer = layer;
      }
      if (circuit !== initialMetadata?.circuit) {
        updates.circuit = circuit;
      }
      if (vector !== initialMetadata?.vector) {
        updates.vector = vector;
      }
      if (entity !== (initialMetadata?.re || "")) {
        updates.entity = entity;
      }

      // Only update if there are changes
      if (Object.keys(updates).length === 0) {
        toast({
          title: "No changes",
          description: "No changes were made to the memory.",
        });
        onOpenChange(false);
        return;
      }

      await updateMemoryWithMetadata(memoryId, updates);

      toast({
        title: "Memory updated",
        description: "The memory has been successfully updated.",
      });

      onOpenChange(false);
      onSuccess?.();
    } catch (error) {
      toast({
        title: "Error",
        description: "Failed to update memory. Please try again.",
        variant: "destructive",
      });
    }
  };

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="bg-zinc-900 border-zinc-800 max-w-2xl max-h-[90vh] overflow-y-auto">
        <DialogHeader>
          <DialogTitle className="text-white">Edit Memory</DialogTitle>
          <DialogDescription className="text-zinc-400">
            Update the memory content and AXIS metadata.
          </DialogDescription>
        </DialogHeader>

        <div className="space-y-4 py-4">
          {/* Memory Content */}
          <div className="space-y-2">
            <Label htmlFor="content" className="text-zinc-200">
              Content
            </Label>
            <Textarea
              id="content"
              value={content}
              onChange={(e) => setContent(e.target.value)}
              className="bg-zinc-800 border-zinc-700 text-white min-h-[100px]"
              placeholder="Memory content..."
            />
          </div>

          {/* AXIS Metadata Grid */}
          <div className="grid grid-cols-2 gap-4">
            {/* Vault */}
            <div className="space-y-2">
              <Label className="text-zinc-200">Vault</Label>
              <Select
                value={vault}
                onValueChange={(v) => setVault(v as Vault)}
              >
                <SelectTrigger className="bg-zinc-800 border-zinc-700 text-white">
                  <SelectValue placeholder="Select vault..." />
                </SelectTrigger>
                <SelectContent className="bg-zinc-800 border-zinc-700">
                  {VAULT_OPTIONS.map((opt) => (
                    <SelectItem
                      key={opt.value}
                      value={opt.value}
                      className="text-white"
                    >
                      {opt.label}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>

            {/* Layer */}
            <div className="space-y-2">
              <Label className="text-zinc-200">Layer</Label>
              <Select
                value={layer}
                onValueChange={(v) => setLayer(v as Layer)}
              >
                <SelectTrigger className="bg-zinc-800 border-zinc-700 text-white">
                  <SelectValue placeholder="Select layer..." />
                </SelectTrigger>
                <SelectContent className="bg-zinc-800 border-zinc-700">
                  {LAYER_OPTIONS.map((opt) => (
                    <SelectItem
                      key={opt.value}
                      value={opt.value}
                      className="text-white"
                    >
                      {opt.label}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>

            {/* Circuit */}
            <div className="space-y-2">
              <Label className="text-zinc-200">Circuit</Label>
              <Select
                value={circuit?.toString()}
                onValueChange={(v) => setCircuit(parseInt(v) as Circuit)}
              >
                <SelectTrigger className="bg-zinc-800 border-zinc-700 text-white">
                  <SelectValue placeholder="Select circuit..." />
                </SelectTrigger>
                <SelectContent className="bg-zinc-800 border-zinc-700">
                  {CIRCUIT_OPTIONS.map((opt) => (
                    <SelectItem
                      key={opt.value}
                      value={opt.value.toString()}
                      className="text-white"
                    >
                      {opt.label}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>

            {/* Vector */}
            <div className="space-y-2">
              <Label className="text-zinc-200">Vector</Label>
              <Select
                value={vector}
                onValueChange={(v) => setVector(v as Vector)}
              >
                <SelectTrigger className="bg-zinc-800 border-zinc-700 text-white">
                  <SelectValue placeholder="Select vector..." />
                </SelectTrigger>
                <SelectContent className="bg-zinc-800 border-zinc-700">
                  {VECTOR_OPTIONS.map((opt) => (
                    <SelectItem
                      key={opt.value}
                      value={opt.value}
                      className="text-white"
                    >
                      {opt.label}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>
          </div>

          {/* Entity Reference */}
          <div className="space-y-2">
            <Label htmlFor="entity" className="text-zinc-200">
              Entity Reference
            </Label>
            <Input
              id="entity"
              value={entity}
              onChange={(e) => setEntity(e.target.value)}
              className="bg-zinc-800 border-zinc-700 text-white"
              placeholder="e.g., person name, project, topic..."
            />
            <p className="text-xs text-zinc-500">
              The primary entity this memory is about
            </p>
          </div>
        </div>

        <DialogFooter>
          <Button
            variant="outline"
            onClick={() => onOpenChange(false)}
            className="border-zinc-700 text-zinc-300 hover:bg-zinc-800"
          >
            Cancel
          </Button>
          <Button
            onClick={handleSave}
            disabled={isLoading}
            className="bg-primary text-black hover:bg-primary/90"
          >
            {isLoading ? "Saving..." : "Save Changes"}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
