import {
  Edit,
  MoreHorizontal,
  Trash2,
  Pause,
  Archive,
  Play,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { Checkbox } from "@/components/ui/checkbox";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { useToast } from "@/hooks/use-toast";
import { useMemoriesApi } from "@/hooks/useMemoriesApi";
import { useDispatch, useSelector } from "react-redux";
import { RootState } from "@/store/store";
import {
  selectMemory,
  deselectMemory,
  selectAllMemories,
  clearSelection,
} from "@/store/memoriesSlice";
import { Layers, Calendar } from "lucide-react";
import { useRouter } from "next/navigation";
import { useUI } from "@/hooks/useUI";
import { Badge } from "@/components/ui/badge";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import { formatDate } from "@/lib/helpers";

export function MemoryTable() {
  const { toast } = useToast();
  const router = useRouter();
  const dispatch = useDispatch();
  const selectedMemoryIds = useSelector(
    (state: RootState) => state.memories.selectedMemoryIds
  );
  const memories = useSelector((state: RootState) => state.memories.memories);

  const { deleteMemories, updateMemoryState, isLoading } = useMemoriesApi();

  const handleDeleteMemory = (id: string) => {
    deleteMemories([id]);
  };

  const handleSelectAll = (checked: boolean) => {
    if (checked) {
      dispatch(selectAllMemories());
    } else {
      dispatch(clearSelection());
    }
  };

  const handleSelectMemory = (id: string, checked: boolean) => {
    if (checked) {
      dispatch(selectMemory(id));
    } else {
      dispatch(deselectMemory(id));
    }
  };
  const { handleOpenUpdateMemoryDialog } = useUI();

  const handleEditMemory = (memory_id: string, memory_content: string) => {
    handleOpenUpdateMemoryDialog(memory_id, memory_content);
  };

  const handleUpdateMemoryState = async (id: string, newState: string) => {
    try {
      await updateMemoryState([id], newState);
    } catch (error) {
      toast({
        title: "Error",
        description: "Failed to update memory state",
        variant: "destructive",
      });
    }
  };

  const isAllSelected =
    memories.length > 0 && selectedMemoryIds.length === memories.length;
  const isPartiallySelected =
    selectedMemoryIds.length > 0 && selectedMemoryIds.length < memories.length;

  const handleMemoryClick = (id: string) => {
    router.push(`/memory/${id}`);
  };

  return (
    <div className="rounded-md border">
      <Table className="">
        <TableHeader>
          <TableRow className="bg-zinc-800 hover:bg-zinc-800">
            <TableHead className="w-[50px] pl-4">
              <Checkbox
                className="data-[state=checked]:border-primary border-zinc-500/50"
                checked={isAllSelected}
                data-state={
                  isPartiallySelected
                    ? "indeterminate"
                    : isAllSelected
                    ? "checked"
                    : "unchecked"
                }
                onCheckedChange={handleSelectAll}
              />
            </TableHead>
            <TableHead className="border-zinc-700">
              <div className="flex items-center min-w-[400px]">
                <Layers className="mr-1 h-4 w-4" />
                Memory
              </div>
            </TableHead>
            <TableHead className="w-[110px] border-zinc-700 text-center">
              Category
            </TableHead>
            <TableHead className="w-[100px] border-zinc-700 text-center">
              Scope
            </TableHead>
            <TableHead className="w-[120px] border-zinc-700 text-center">
              Entity
            </TableHead>
            <TableHead className="w-[80px] border-zinc-700 text-center">
              Artifact
            </TableHead>
            <TableHead className="w-[80px] border-zinc-700 text-center">
              Source
            </TableHead>
            <TableHead className="w-[140px] border-zinc-700">
              <div className="flex items-center w-full justify-center">
                <Calendar className="mr-1 h-4 w-4" />
                Created On
              </div>
            </TableHead>
            <TableHead className="text-right border-zinc-700 flex justify-center">
              <div className="flex items-center justify-end">
                <MoreHorizontal className="h-4 w-4 mr-2" />
              </div>
            </TableHead>
          </TableRow>
        </TableHeader>
        <TableBody>
          {memories.map((memory) => (
            <TableRow
              key={memory.id}
              className={`hover:bg-zinc-900/50 ${
                memory.state === "paused" || memory.state === "archived"
                  ? "text-zinc-400"
                  : ""
              } ${isLoading ? "animate-pulse opacity-50" : ""}`}
            >
              <TableCell className="pl-4">
                <Checkbox
                  className="data-[state=checked]:border-primary border-zinc-500/50"
                  checked={selectedMemoryIds.includes(memory.id)}
                  onCheckedChange={(checked) =>
                    handleSelectMemory(memory.id, checked as boolean)
                  }
                />
              </TableCell>
              <TableCell className="">
                {memory.state === "paused" || memory.state === "archived" ? (
                  <TooltipProvider>
                    <Tooltip delayDuration={0}>
                      <TooltipTrigger asChild>
                        <div
                          onClick={() => handleMemoryClick(memory.id)}
                          className={`font-medium ${
                            memory.state === "paused" ||
                            memory.state === "archived"
                              ? "text-zinc-400"
                              : "text-white"
                          } cursor-pointer`}
                        >
                          {memory.memory}
                        </div>
                      </TooltipTrigger>
                      <TooltipContent>
                        <p>
                          This memory is{" "}
                          <span className="font-bold">
                            {memory.state === "paused" ? "paused" : "archived"}
                          </span>{" "}
                          and <span className="font-bold">disabled</span>.
                        </p>
                      </TooltipContent>
                    </Tooltip>
                  </TooltipProvider>
                ) : (
                  <div
                    onClick={() => handleMemoryClick(memory.id)}
                    className={`font-medium text-white cursor-pointer`}
                  >
                    {memory.memory}
                  </div>
                )}
              </TableCell>
              <TableCell className="w-[110px] text-center">
                <Badge
                  variant="secondary"
                  className="bg-zinc-800 text-zinc-200 border border-zinc-700"
                >
                  {memory.metadata?.category ?? "—"}
                </Badge>
              </TableCell>
              <TableCell className="w-[100px] text-center">
                <Badge
                  variant="secondary"
                  className="bg-zinc-800 text-zinc-200 border border-zinc-700"
                >
                  {memory.metadata?.scope ?? "—"}
                </Badge>
              </TableCell>
              <TableCell className="w-[120px] text-center">
                {memory.metadata?.entity ? (
                  <Badge
                    variant="secondary"
                    className="bg-primary/20 text-primary border border-primary/50 cursor-pointer hover:bg-primary/30"
                    onClick={(e) => {
                      e.stopPropagation();
                      router.push(`/entity/${encodeURIComponent(memory.metadata?.entity || "")}`);
                    }}
                  >
                    {memory.metadata.entity}
                  </Badge>
                ) : (
                  <span className="text-zinc-500">—</span>
                )}
              </TableCell>
              <TableCell className="w-[80px] text-center">
                {memory.metadata?.artifact_type ? (
                  <Badge
                    variant="secondary"
                    className="bg-amber-900/30 text-amber-300 border border-amber-700"
                  >
                    {memory.metadata.artifact_type}
                  </Badge>
                ) : (
                  <span className="text-zinc-500">—</span>
                )}
              </TableCell>
              <TableCell className="w-[80px] text-center">
                {memory.metadata?.source ? (
                  <Badge
                    variant="secondary"
                    className="bg-green-900/30 text-green-300 border border-green-700"
                  >
                    {memory.metadata.source}
                  </Badge>
                ) : (
                  <span className="text-zinc-500">—</span>
                )}
              </TableCell>
              <TableCell className="w-[140px] text-center">
                {formatDate(memory.created_at)}
              </TableCell>
              <TableCell className="text-right flex justify-center">
                <DropdownMenu>
                  <DropdownMenuTrigger asChild>
                    <Button variant="ghost" size="icon" className="h-8 w-8">
                      <MoreHorizontal className="h-4 w-4" />
                    </Button>
                  </DropdownMenuTrigger>
                  <DropdownMenuContent
                    align="end"
                    className="bg-zinc-900 border-zinc-800"
                  >
                    <DropdownMenuItem
                      className="cursor-pointer"
                      onClick={() => {
                        const newState =
                          memory.state === "active" ? "paused" : "active";
                        handleUpdateMemoryState(memory.id, newState);
                      }}
                    >
                      {memory?.state === "active" ? (
                        <>
                          <Pause className="mr-2 h-4 w-4" />
                          Pause
                        </>
                      ) : (
                        <>
                          <Play className="mr-2 h-4 w-4" />
                          Resume
                        </>
                      )}
                    </DropdownMenuItem>
                    <DropdownMenuItem
                      className="cursor-pointer"
                      onClick={() => {
                        const newState =
                          memory.state === "active" ? "archived" : "active";
                        handleUpdateMemoryState(memory.id, newState);
                      }}
                    >
                      <Archive className="mr-2 h-4 w-4" />
                      {memory?.state !== "archived" ? (
                        <>Archive</>
                      ) : (
                        <>Unarchive</>
                      )}
                    </DropdownMenuItem>
                    <DropdownMenuItem
                      className="cursor-pointer"
                      onSelect={() => {
                        // Use setTimeout to allow dropdown to close first
                        setTimeout(() => handleEditMemory(memory.id, memory.memory), 0);
                      }}
                    >
                      <Edit className="mr-2 h-4 w-4" />
                      Edit
                    </DropdownMenuItem>
                    <DropdownMenuSeparator />
                    <DropdownMenuItem
                      className="cursor-pointer text-red-500 focus:text-red-500"
                      onClick={() => handleDeleteMemory(memory.id)}
                    >
                      <Trash2 className="mr-2 h-4 w-4" />
                      Delete
                    </DropdownMenuItem>
                  </DropdownMenuContent>
                </DropdownMenu>
              </TableCell>
            </TableRow>
          ))}
        </TableBody>
      </Table>
    </div>
  );
}
