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
import {
  ArtifactType,
  MemoryScope,
  SourceType,
  StructuredCategory,
} from "@/components/types";

interface EditMemoryDialogProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  memoryId: string;
  initialContent: string;
  initialMetadata?: {
    category?: StructuredCategory;
    scope?: MemoryScope;
    artifact_type?: ArtifactType;
    artifact_ref?: string;
    entity?: string;
    source?: SourceType;
    evidence?: string[];
    tags?: Record<string, any>;
  };
  onSuccess?: () => void;
}

const CATEGORY_OPTIONS: { value: StructuredCategory; label: string }[] = [
  { value: "decision", label: "Decision" },
  { value: "convention", label: "Convention" },
  { value: "architecture", label: "Architecture" },
  { value: "dependency", label: "Dependency" },
  { value: "workflow", label: "Workflow" },
  { value: "testing", label: "Testing" },
  { value: "security", label: "Security" },
  { value: "performance", label: "Performance" },
  { value: "runbook", label: "Runbook" },
  { value: "glossary", label: "Glossary" },
];

const SCOPE_OPTIONS: { value: MemoryScope; label: string }[] = [
  { value: "session", label: "Session" },
  { value: "user", label: "User" },
  { value: "team", label: "Team" },
  { value: "project", label: "Project" },
  { value: "org", label: "Org" },
  { value: "enterprise", label: "Enterprise" },
];

const ARTIFACT_OPTIONS: { value: ArtifactType; label: string }[] = [
  { value: "repo", label: "Repo" },
  { value: "service", label: "Service" },
  { value: "module", label: "Module" },
  { value: "component", label: "Component" },
  { value: "api", label: "API" },
  { value: "db", label: "Database" },
  { value: "infra", label: "Infra" },
  { value: "file", label: "File" },
];

const SOURCE_OPTIONS: { value: SourceType; label: string }[] = [
  { value: "user", label: "User" },
  { value: "inference", label: "Inference" },
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

  const [content, setContent] = useState("");
  const [category, setCategory] = useState<StructuredCategory | undefined>(undefined);
  const [scope, setScope] = useState<MemoryScope | undefined>(undefined);
  const [artifactType, setArtifactType] = useState<ArtifactType | undefined>(undefined);
  const [artifactRef, setArtifactRef] = useState("");
  const [entity, setEntity] = useState("");
  const [source, setSource] = useState<SourceType | undefined>(undefined);
  const [evidence, setEvidence] = useState("");

  useEffect(() => {
    if (open) {
      setContent(initialContent || "");
      setCategory(initialMetadata?.category);
      setScope(initialMetadata?.scope);
      setArtifactType(initialMetadata?.artifact_type);
      setArtifactRef(initialMetadata?.artifact_ref || "");
      setEntity(initialMetadata?.entity || "");
      setSource(initialMetadata?.source);
      setEvidence((initialMetadata?.evidence || []).join(", "));
    }
  }, [open, initialContent, initialMetadata]);

  const handleSave = async () => {
    try {
      const updates: {
        content?: string;
        category?: StructuredCategory;
        scope?: MemoryScope;
        artifact_type?: ArtifactType;
        artifact_ref?: string;
        entity?: string;
        source?: SourceType;
        evidence?: string[];
      } = {};

      if (content !== initialContent) {
        updates.content = content;
      }
      if (category !== initialMetadata?.category) {
        updates.category = category;
      }
      if (scope !== initialMetadata?.scope) {
        updates.scope = scope;
      }
      if (artifactType !== initialMetadata?.artifact_type) {
        updates.artifact_type = artifactType;
      }
      if ((artifactRef || "") !== (initialMetadata?.artifact_ref || "")) {
        updates.artifact_ref = artifactRef || undefined;
      }
      if ((entity || "") !== (initialMetadata?.entity || "")) {
        updates.entity = entity || undefined;
      }
      if (source !== initialMetadata?.source) {
        updates.source = source;
      }

      const evidenceList = evidence
        .split(",")
        .map((item) => item.trim())
        .filter(Boolean);
      if (evidenceList.join(",") !== (initialMetadata?.evidence || []).join(",")) {
        updates.evidence = evidenceList.length ? evidenceList : undefined;
      }

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
      <DialogContent className="sm:max-w-[700px] bg-zinc-900 border-zinc-800">
        <DialogHeader>
          <DialogTitle className="text-white">Edit Memory</DialogTitle>
          <DialogDescription className="text-zinc-400">
            Update the content or structured metadata.
          </DialogDescription>
        </DialogHeader>
        <div className="space-y-4">
          <div className="space-y-2">
            <Label htmlFor="content" className="text-zinc-300">
              Content
            </Label>
            <Textarea
              id="content"
              value={content}
              onChange={(e) => setContent(e.target.value)}
              className="bg-zinc-800 border-zinc-700 text-white"
              rows={4}
            />
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div className="space-y-2">
              <Label className="text-zinc-300">Category</Label>
              <Select value={category} onValueChange={(value) => setCategory(value as StructuredCategory)}>
                <SelectTrigger className="bg-zinc-800 border-zinc-700 text-white">
                  <SelectValue placeholder="Select category" />
                </SelectTrigger>
                <SelectContent className="bg-zinc-800 border-zinc-700 text-white">
                  {CATEGORY_OPTIONS.map((option) => (
                    <SelectItem key={option.value} value={option.value}>
                      {option.label}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>

            <div className="space-y-2">
              <Label className="text-zinc-300">Scope</Label>
              <Select value={scope} onValueChange={(value) => setScope(value as MemoryScope)}>
                <SelectTrigger className="bg-zinc-800 border-zinc-700 text-white">
                  <SelectValue placeholder="Select scope" />
                </SelectTrigger>
                <SelectContent className="bg-zinc-800 border-zinc-700 text-white">
                  {SCOPE_OPTIONS.map((option) => (
                    <SelectItem key={option.value} value={option.value}>
                      {option.label}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>

            <div className="space-y-2">
              <Label className="text-zinc-300">Artifact Type</Label>
              <Select value={artifactType} onValueChange={(value) => setArtifactType(value as ArtifactType)}>
                <SelectTrigger className="bg-zinc-800 border-zinc-700 text-white">
                  <SelectValue placeholder="Select artifact type" />
                </SelectTrigger>
                <SelectContent className="bg-zinc-800 border-zinc-700 text-white">
                  {ARTIFACT_OPTIONS.map((option) => (
                    <SelectItem key={option.value} value={option.value}>
                      {option.label}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>

            <div className="space-y-2">
              <Label htmlFor="artifact-ref" className="text-zinc-300">
                Artifact Ref
              </Label>
              <Input
                id="artifact-ref"
                value={artifactRef}
                onChange={(e) => setArtifactRef(e.target.value)}
                className="bg-zinc-800 border-zinc-700 text-white"
                placeholder="repo/path/file or symbol"
              />
            </div>

            <div className="space-y-2">
              <Label htmlFor="entity" className="text-zinc-300">
                Entity
              </Label>
              <Input
                id="entity"
                value={entity}
                onChange={(e) => setEntity(e.target.value)}
                className="bg-zinc-800 border-zinc-700 text-white"
                placeholder="team, service, component"
              />
            </div>

            <div className="space-y-2">
              <Label className="text-zinc-300">Source</Label>
              <Select value={source} onValueChange={(value) => setSource(value as SourceType)}>
                <SelectTrigger className="bg-zinc-800 border-zinc-700 text-white">
                  <SelectValue placeholder="Select source" />
                </SelectTrigger>
                <SelectContent className="bg-zinc-800 border-zinc-700 text-white">
                  {SOURCE_OPTIONS.map((option) => (
                    <SelectItem key={option.value} value={option.value}>
                      {option.label}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>

            <div className="space-y-2 md:col-span-2">
              <Label htmlFor="evidence" className="text-zinc-300">
                Evidence (comma-separated)
              </Label>
              <Input
                id="evidence"
                value={evidence}
                onChange={(e) => setEvidence(e.target.value)}
                className="bg-zinc-800 border-zinc-700 text-white"
                placeholder="ADR-12, PR-330"
              />
            </div>
          </div>
        </div>
        <DialogFooter className="mt-4">
          <Button variant="ghost" onClick={() => onOpenChange(false)}>
            Cancel
          </Button>
          <Button onClick={handleSave} disabled={isLoading}>
            Save Changes
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
