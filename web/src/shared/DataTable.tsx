import { useState, useMemo, useCallback } from "react";
import {
  useReactTable,
  getCoreRowModel,
  getSortedRowModel,
  getFilteredRowModel,
  getPaginationRowModel,
  flexRender,
  type ColumnDef,
  type SortingState,
  type PaginationState,
  type VisibilityState,
} from "@tanstack/react-table";
import { ChevronUp, ChevronDown, ChevronsUpDown } from "lucide-react";

const PAGE_SIZES = [50, 100, 500, 999999] as const;
const PAGE_LABEL: Record<number, string> = { 50: "50", 100: "100", 500: "500", 999999: "All" };

export function DataTable<T extends object>({
  data,
  columns,
  defaultPageSize = 50,
  searchPlaceholder = "Search…",
  hideZeroCols = false,
  className = "",
}: {
  data: T[];
  columns: ColumnDef<T, unknown>[];
  defaultPageSize?: number;
  searchPlaceholder?: string;
  /** Auto-hide numeric columns where every row's value is 0 or absent. */
  hideZeroCols?: boolean;
  className?: string;
}) {
  const [sorting, setSorting] = useState<SortingState>([]);
  const [globalFilter, setGlobalFilter] = useState("");
  const [pagination, setPagination] = useState<PaginationState>({
    pageIndex: 0,
    pageSize: defaultPageSize,
  });

  // Compute column visibility: hide columns where all values are falsy / 0.
  const columnVisibility = useMemo<VisibilityState>(() => {
    if (!hideZeroCols || data.length === 0) return {};
    const vis: VisibilityState = {};
    for (const col of columns) {
      const key = (col as { accessorKey?: string }).accessorKey;
      if (!key) continue;
      const allZero = data.every((row) => {
        const v = (row as Record<string, unknown>)[key];
        return !v || v === 0;
      });
      if (allZero) vis[key] = false;
    }
    return vis;
  }, [data, columns, hideZeroCols]);

  const table = useReactTable({
    data,
    columns,
    state: { sorting, globalFilter, pagination, columnVisibility },
    onSortingChange: setSorting,
    onGlobalFilterChange: setGlobalFilter,
    onPaginationChange: setPagination,
    getCoreRowModel: getCoreRowModel(),
    getSortedRowModel: getSortedRowModel(),
    getFilteredRowModel: getFilteredRowModel(),
    getPaginationRowModel: getPaginationRowModel(),
  });

  const filteredCount = table.getFilteredRowModel().rows.length;
  const pageCount = table.getPageCount();

  const onFilterChange = useCallback((v: string) => {
    setGlobalFilter(v);
    setPagination((p) => ({ ...p, pageIndex: 0 }));
  }, []);

  const onPageSizeChange = useCallback((size: number) => {
    setPagination({ pageIndex: 0, pageSize: size });
  }, []);

  return (
    <div className={className}>
      {/* Toolbar */}
      <div className="mb-2 flex items-center gap-2">
        <input
          type="text"
          value={globalFilter}
          onChange={(e) => onFilterChange(e.target.value)}
          placeholder={searchPlaceholder}
          className="h-7 flex-1 rounded border border-border bg-background px-2 text-xs outline-none focus:ring-1 focus:ring-ring"
        />
        <select
          value={pagination.pageSize}
          onChange={(e) => onPageSizeChange(Number(e.target.value))}
          className="h-7 rounded border border-border bg-background px-2 text-xs"
        >
          {PAGE_SIZES.map((s) => (
            <option key={s} value={s}>
              {PAGE_LABEL[s]} / page
            </option>
          ))}
        </select>
      </div>

      {/* Table */}
      <div className="overflow-x-auto rounded-lg border border-border">
        <table className="w-full text-xs">
          <thead className="bg-muted/40">
            {table.getHeaderGroups().map((hg) => (
              <tr key={hg.id}>
                {hg.headers.map((header) => {
                  const canSort = header.column.getCanSort();
                  const sorted = header.column.getIsSorted();
                  return (
                    <th
                      key={header.id}
                      className={`whitespace-nowrap px-3 py-2 text-left text-xs font-semibold text-muted-foreground${canSort ? " cursor-pointer select-none hover:text-foreground" : ""}`}
                      onClick={canSort ? header.column.getToggleSortingHandler() : undefined}
                    >
                      <span className="inline-flex items-center gap-0.5">
                        {flexRender(header.column.columnDef.header, header.getContext())}
                        {canSort &&
                          (sorted === "asc" ? (
                            <ChevronUp className="h-3 w-3" />
                          ) : sorted === "desc" ? (
                            <ChevronDown className="h-3 w-3" />
                          ) : (
                            <ChevronsUpDown className="h-3 w-3 opacity-40" />
                          ))}
                      </span>
                    </th>
                  );
                })}
              </tr>
            ))}
          </thead>
          <tbody>
            {table.getRowModel().rows.length === 0 ? (
              <tr>
                <td
                  colSpan={table.getVisibleLeafColumns().length}
                  className="px-3 py-8 text-center text-muted-foreground"
                >
                  {globalFilter ? "No results match your search." : "No data."}
                </td>
              </tr>
            ) : (
              table.getRowModel().rows.map((row) => (
                <tr key={row.id} className="border-t border-border/50 hover:bg-muted/20">
                  {row.getVisibleCells().map((cell) => (
                    <td key={cell.id} className="whitespace-nowrap px-3 py-1.5 font-mono">
                      {flexRender(cell.column.columnDef.cell, cell.getContext())}
                    </td>
                  ))}
                </tr>
              ))
            )}
          </tbody>
        </table>
      </div>

      {/* Footer */}
      <div className="mt-1.5 flex items-center justify-between text-xs text-muted-foreground">
        <span>{filteredCount.toLocaleString()} rows</span>
        {pageCount > 1 && (
          <div className="flex items-center gap-1">
            <button
              disabled={!table.getCanPreviousPage()}
              onClick={() => table.previousPage()}
              className="rounded px-2 py-0.5 hover:bg-muted disabled:opacity-40"
            >
              ← Prev
            </button>
            <span className="px-1">
              {pagination.pageIndex + 1} / {pageCount}
            </span>
            <button
              disabled={!table.getCanNextPage()}
              onClick={() => table.nextPage()}
              className="rounded px-2 py-0.5 hover:bg-muted disabled:opacity-40"
            >
              Next →
            </button>
          </div>
        )}
      </div>
    </div>
  );
}
