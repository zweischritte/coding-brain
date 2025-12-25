"use client";

import { useEffect, useCallback } from "react";
import { MemoriesSection } from "@/app/memories/components/MemoriesSection";
import { MemoryFilters } from "@/app/memories/components/MemoryFilters";
import { useRouter, useSearchParams } from "next/navigation";
import "@/styles/animation.css";
import { EditMemoryDialog } from "@/app/memories/components/EditMemoryDialog";
import { useUI } from "@/hooks/useUI";
import { useMemoriesApi } from "@/hooks/useMemoriesApi";
import { useSelector } from "react-redux";
import { RootState } from "@/store/store";

export default function MemoriesPage() {
  const router = useRouter();
  const searchParams = useSearchParams();
  const { updateMemoryDialog, handleCloseUpdateMemoryDialog } = useUI();
  const { fetchMemories } = useMemoriesApi();

  // Get the selected memory's metadata from Redux store
  const memories = useSelector((state: RootState) => state.memories.memories);
  const selectedMemory = memories.find(m => m.id === updateMemoryDialog.memoryId);

  useEffect(() => {
    // Set default pagination values if not present in URL
    if (!searchParams.has("page") || !searchParams.has("size")) {
      const params = new URLSearchParams(searchParams.toString());
      if (!searchParams.has("page")) params.set("page", "1");
      if (!searchParams.has("size")) params.set("size", "10");
      router.push(`?${params.toString()}`);
    }
  }, []);

  const handleEditSuccess = useCallback(() => {
    fetchMemories();
  }, [fetchMemories]);

  return (
    <div className="">
      <EditMemoryDialog
        memoryId={updateMemoryDialog.memoryId || ""}
        initialContent={updateMemoryDialog.memoryContent || ""}
        initialMetadata={selectedMemory?.metadata}
        open={updateMemoryDialog.isOpen}
        onOpenChange={handleCloseUpdateMemoryDialog}
        onSuccess={handleEditSuccess}
      />
      <main className="flex-1 py-6">
        <div className="container">
          <div className="mt-1 pb-4 animate-fade-slide-down">
            <MemoryFilters />
          </div>
          <div className="animate-fade-slide-down delay-1">
            <MemoriesSection />
          </div>
        </div>
      </main>
    </div>
  );
}
