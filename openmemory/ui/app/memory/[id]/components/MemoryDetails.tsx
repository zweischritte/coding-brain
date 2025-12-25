"use client";
import { useMemoriesApi } from "@/hooks/useMemoriesApi";
import { MemoryActions } from "./MemoryActions";
import { ArrowLeft, Copy, Check, User } from "lucide-react";
import { Button } from "@/components/ui/button";
import { useRouter } from "next/navigation";
import { AccessLog } from "./AccessLog";
import Image from "next/image";
import Categories from "@/components/shared/categories";
import { useEffect, useState } from "react";
import { useSelector } from "react-redux";
import { RootState } from "@/store/store";
import { constants } from "@/components/shared/source-app";
import { RelatedMemories } from "./RelatedMemories";
import { SimilarMemories } from "@/app/memories/components/SimilarMemories";
import { Badge } from "@/components/ui/badge";
import Link from "next/link";

interface MemoryDetailsProps {
  memory_id: string;
}

export function MemoryDetails({ memory_id }: MemoryDetailsProps) {
  const router = useRouter();
  const { fetchMemoryById, hasUpdates } = useMemoriesApi();
  const memory = useSelector(
    (state: RootState) => state.memories.selectedMemory
  );
  const [copied, setCopied] = useState(false);

  const handleCopy = async () => {
    if (memory?.id) {
      await navigator.clipboard.writeText(memory.id);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    }
  };

  useEffect(() => {
    fetchMemoryById(memory_id);
  }, []);

  return (
    <div className="container mx-auto py-6 px-4">
      <Button
        variant="ghost"
        className="mb-4 text-zinc-400 hover:text-white"
        onClick={() => router.back()}
      >
        <ArrowLeft className="h-4 w-4 mr-2" />
        Back to Memories
      </Button>
      <div className="flex gap-4 w-full">
        <div className="rounded-lg w-2/3 border h-fit pb-2 border-zinc-800 bg-zinc-900 overflow-hidden">
          <div className="">
            <div className="flex px-6 py-3 justify-between items-center mb-6 bg-zinc-800 border-b border-zinc-800">
              <div className="flex items-center gap-2">
                <h1 className="font-semibold text-white">
                  Memory{" "}
                  <span className="ml-1 text-zinc-400 text-sm font-normal">
                    #{memory?.id?.slice(0, 6)}
                  </span>
                </h1>
                <Button
                  variant="ghost"
                  size="icon"
                  className="h-4 w-4 text-zinc-400 hover:text-white -ml-[5px] mt-1"
                  onClick={handleCopy}
                >
                  {copied ? (
                    <Check className="h-3 w-3" />
                  ) : (
                    <Copy className="h-3 w-3" />
                  )}
                </Button>
              </div>
              <MemoryActions
                memoryId={memory?.id || ""}
                memoryContent={memory?.text || ""}
                memoryState={memory?.state || ""}
              />
            </div>

            <div className="px-6 py-2">
              <div className="border-l-2 border-primary pl-4 mb-6">
                <p
                  className={`${
                    memory?.state === "archived" || memory?.state === "paused"
                      ? "text-zinc-400"
                      : "text-white"
                  }`}
                >
                  {memory?.text}
                </p>
              </div>

              {/* AXIS Metadata Section */}
              {(memory?.metadata?.vault || memory?.metadata?.layer || memory?.metadata?.circuit || memory?.metadata?.vector || memory?.metadata?.re) && (
                <div className="mt-4 pt-4 border-t border-zinc-800">
                  <h3 className="text-sm font-semibold text-zinc-400 mb-3">AXIS Metadata</h3>
                  <div className="flex flex-wrap gap-2">
                    {memory?.metadata?.vault && (
                      <Badge variant="secondary" className="bg-purple-900/30 text-purple-300 border border-purple-700">
                        Vault: {memory.metadata.vault}
                      </Badge>
                    )}
                    {memory?.metadata?.layer && (
                      <Badge variant="secondary" className="bg-blue-900/30 text-blue-300 border border-blue-700">
                        Layer: {memory.metadata.layer}
                      </Badge>
                    )}
                    {memory?.metadata?.circuit && (
                      <Badge variant="secondary" className="bg-green-900/30 text-green-300 border border-green-700">
                        Circuit: {memory.metadata.circuit}
                      </Badge>
                    )}
                    {memory?.metadata?.vector && (
                      <Badge variant="secondary" className="bg-amber-900/30 text-amber-300 border border-amber-700">
                        Vector: {memory.metadata.vector}
                      </Badge>
                    )}
                    {memory?.metadata?.re && (
                      <Link href={`/entity/${encodeURIComponent(memory.metadata.re)}`}>
                        <Badge variant="secondary" className="bg-primary/20 text-primary border border-primary/50 cursor-pointer hover:bg-primary/30">
                          <User className="h-3 w-3 mr-1" />
                          {memory.metadata.re}
                        </Badge>
                      </Link>
                    )}
                  </div>
                </div>
              )}

              <div className="mt-6 pt-4 border-t border-zinc-800">
                <div className="flex justify-between items-center">
                  <div className="">
                    <Categories
                      categories={memory?.categories || []}
                      isPaused={
                        memory?.state === "archived" ||
                        memory?.state === "paused"
                      }
                    />
                  </div>
                  <div className="flex items-center gap-2 min-w-[300px] justify-end">
                    <div className="flex items-center gap-2">
                      <div className="flex items-center gap-1 bg-zinc-700 px-3 py-1 rounded-lg">
                        <span className="text-sm text-zinc-400">
                          Created by:
                        </span>
                        <div className="w-4 h-4 rounded-full bg-zinc-700 flex items-center justify-center overflow-hidden">
                          <Image
                            src={
                              constants[
                                memory?.app_name as keyof typeof constants
                              ]?.iconImage || ""
                            }
                            alt="OpenMemory"
                            width={24}
                            height={24}
                          />
                        </div>
                        <p className="text-sm text-zinc-100 font-semibold">
                          {
                            constants[
                              memory?.app_name as keyof typeof constants
                            ]?.name
                          }
                        </p>
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
        <div className="w-1/3 flex flex-col gap-4">
          <SimilarMemories memoryId={memory?.id || ""} />
          <AccessLog memoryId={memory?.id || ""} />
          <RelatedMemories memoryId={memory?.id || ""} />
        </div>
      </div>
    </div>
  );
}
