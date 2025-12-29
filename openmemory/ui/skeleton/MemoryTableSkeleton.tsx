import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table"
import { MoreHorizontal, Layers, Calendar } from "lucide-react"

export function MemoryTableSkeleton() {
  // Create an array of 5 items for the loading state
  const loadingRows = Array(5).fill(null)

  return (
    <div className="rounded-md border">
      <Table>
        <TableHeader>
          <TableRow className="bg-zinc-800 hover:bg-zinc-800">
            <TableHead className="w-[50px] pl-4">
              <div className="h-4 w-4 rounded bg-zinc-700/50 animate-pulse" />
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
          {loadingRows.map((_, index) => (
            <TableRow key={index} className="animate-pulse">
              <TableCell className="pl-4">
                <div className="h-4 w-4 rounded bg-zinc-800" />
              </TableCell>
              <TableCell>
                <div className="h-4 w-3/4 bg-zinc-800 rounded" />
              </TableCell>
              <TableCell className="w-[110px]">
                <div className="h-5 w-12 mx-auto bg-zinc-800 rounded-full" />
              </TableCell>
              <TableCell className="w-[100px]">
                <div className="h-5 w-16 mx-auto bg-zinc-800 rounded-full" />
              </TableCell>
              <TableCell className="w-[120px]">
                <div className="h-5 w-16 mx-auto bg-zinc-800 rounded-full" />
              </TableCell>
              <TableCell className="w-[80px]">
                <div className="h-5 w-10 mx-auto bg-zinc-800 rounded-full" />
              </TableCell>
              <TableCell className="w-[80px]">
                <div className="h-5 w-8 mx-auto bg-zinc-800 rounded-full" />
              </TableCell>
              <TableCell className="w-[140px]">
                <div className="h-4 w-20 mx-auto bg-zinc-800 rounded" />
              </TableCell>
              <TableCell>
                <div className="h-8 w-8 bg-zinc-800 rounded mx-auto" />
              </TableCell>
            </TableRow>
          ))}
        </TableBody>
      </Table>
    </div>
  )
}
