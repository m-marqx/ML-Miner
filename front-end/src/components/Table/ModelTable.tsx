"use client"

import React from "react"
import { useEffect, useState } from "react"
import {
    ColumnDef,
    flexRender,
    getCoreRowModel,
    getPaginationRowModel,
    getSortedRowModel,
    PaginationState,
    SortingState,
    useReactTable,
} from "@tanstack/react-table"
import {
    ChevronDownIcon,
    ChevronLeftIcon,
    ChevronRightIcon,
    ChevronUpIcon,
} from "lucide-react"

import { usePagination } from "@/hooks/use-pagination"
import { cn } from "@/lib/utils"
import { Button } from "@/components/ui/button"
import {
    Pagination,
    PaginationContent,
    PaginationEllipsis,
    PaginationItem,
} from "@/components/ui/pagination"
import {
    Select,
    SelectContent,
    SelectItem,
    SelectTrigger,
    SelectValue,
} from "@/components/ui/select"

import { Badge } from "@/components/ui/badge"

import {
    Table,
    TableBody,
    TableCell,
    TableHead,
    TableHeader,
    TableRow,
} from "@/components/ui/table"

type TableData = {
    date: string
    model_33139: string
}

const columns: ColumnDef<TableData>[] = [
    {
        header: "Date",
        accessorKey: "date",
        size: 120,
        cell: ({ row }) => {
            const value = row.getValue("date") as string
            const date = new Date(value)
            const formattedDate = date.toLocaleDateString("en-US", {
                year: "numeric",
                month: "short",
                day: "2-digit",
            })
            const isDifferentTime = date.getHours() !== 20 || date.getMinutes() !== 59 || date.getSeconds() !== 59
            if (isDifferentTime) {
                return (
                    <div className="text-[#f0d25d] font-semibold">
                        {formattedDate} (Pending)
                    </div>
                )
            }
            return <div className="font-semibold">{formattedDate}</div>
        },
    },
    {
        header: "Position",
        accessorKey: "position",
        size: 60,
        cell: ({ row }) => {
            const value = row.getValue("position") as string
            return value[0] === "—" ? "" : value
        },
    },
    {
        header: "Side",
        accessorKey: "side",
        size: 60,
        cell: ({ row }) => {
            const value = row.getValue("side") as string
            let rowValue = <div>{value}</div>

            if (value === "Open Position") {
                rowValue = <Badge variant="default" className="bg-[#00e676] text-black/85 font-semibold">{value}</Badge>
            } else if (value === "Close Position") {
                rowValue = <Badge variant="default" className="bg-[#ef5350] text-black/85 font-semibold">{value}</Badge>
            }
            return rowValue
        },
    },
    {
        header: "Capital",
        accessorKey: "capital",
        size: 60,
        cell: ({ row }) => {
            const value = row.getValue("capital") as string
            return value
        },
    },
]

export default function TableData() {
    const pageSize = 20

    const [pagination, setPagination] = useState<PaginationState>({
        pageIndex: 0,
        pageSize: pageSize,
    })

    const [sorting, setSorting] = useState<SortingState>([
        {
            id: "name",
            desc: false,
        },
    ])

    const [data, setData] = useState<TableData[]>([])
    useEffect(() => {
        async function fetchTableData() {
            const res = await fetch("/api/tableData")
            const json = await res.json()
            setData(json.data)
        }
        fetchTableData()
    }, [])

    const table = useReactTable({
        data,
        columns,
        getCoreRowModel: getCoreRowModel(),
        getSortedRowModel: getSortedRowModel(),
        onSortingChange: setSorting,
        enableSortingRemoval: false,
        getPaginationRowModel: getPaginationRowModel(),
        onPaginationChange: setPagination,
        state: {
            sorting,
            pagination,
        },
    })

    const { pages, showLeftEllipsis, showRightEllipsis } = usePagination({
        currentPage: table.getState().pagination.pageIndex + 1,
        totalPages: table.getPageCount(),
        paginationItemsToDisplay: 5,
    })

    return (
        <div className="flex flex-col h-full">
            <div className="bg-background rounded-md border flex-1 min-h-0 flex flex-col">
                <div className="flex-1 min-h-0 overflow-y-auto rounded-md">
                    <Table>
                        <TableHeader className="text-center rounded-md">
                            {table.getHeaderGroups().map((headerGroup) => (
                                <TableRow key={headerGroup.id} className="bg-zinc-800 hover:bg-zinc-800 rounded-md">
                                    {headerGroup.headers.map((header) => {
                                        return (
                                            <TableHead
                                                key={header.id}
                                                style={{ width: `${header.getSize()}px` }}
                                                className="h-11"
                                            >
                                                {header.isPlaceholder ? null : header.column.getCanSort() ? (
                                                    <div
                                                        className={cn(
                                                            header.column.getCanSort() &&
                                                            "flex h-full cursor-pointer items-center justify-center gap-2 select-none text-white/85"
                                                        )}
                                                        onClick={header.column.getToggleSortingHandler()}
                                                        onKeyDown={(e) => {
                                                            // Enhanced keyboard handling for sorting
                                                            if (
                                                                header.column.getCanSort() &&
                                                                (e.key === "Enter" || e.key === " ")
                                                            ) {
                                                                e.preventDefault()
                                                                header.column.getToggleSortingHandler()?.(e)
                                                            }
                                                        }}
                                                        tabIndex={header.column.getCanSort() ? 0 : undefined}
                                                    >
                                                        {flexRender(
                                                            header.column.columnDef.header,
                                                            header.getContext()
                                                        )}
                                                        {{
                                                            asc: (
                                                                <ChevronUpIcon
                                                                    className="shrink-0 opacity-60"
                                                                    size={16}
                                                                    aria-hidden="true"
                                                                />
                                                            ),
                                                            desc: (
                                                                <ChevronDownIcon
                                                                    className="shrink-0 opacity-60"
                                                                    size={16}
                                                                    aria-hidden="true"
                                                                />
                                                            ),
                                                        }[header.column.getIsSorted() as string] ?? null}
                                                    </div>
                                                ) : (
                                                    flexRender(
                                                        header.column.columnDef.header,
                                                        header.getContext()
                                                    )
                                                )}
                                            </TableHead>
                                        )
                                    })}
                                </TableRow>
                            ))}
                        </TableHeader>
                        <TableBody>
                            {table.getRowModel().rows?.length ? (
                                table.getRowModel().rows.map((row) => (
                                    <TableRow
                                        key={row.id}
                                        data-state={row.getIsSelected() && "selected"}
                                        className="hover:bg-zinc-800"
                                    >
                                        {row.getVisibleCells().map((cell) => (
                                            <TableCell key={cell.id} className="text-center">
                                                {flexRender(
                                                    cell.column.columnDef.cell,
                                                    cell.getContext()
                                                )}
                                            </TableCell>
                                        ))}
                                    </TableRow>
                                ))
                            ) : (
                                <TableRow>
                                    <TableCell
                                        colSpan={columns.length}
                                        className="h-24 text-center"
                                    >
                                        No results.
                                    </TableCell>
                                </TableRow>
                            )}
                        </TableBody>
                    </Table>
                </div>
            </div>

            {/* Pagination */}
            <div className="flex items-center justify-between gap-3 max-sm:flex-col mt-4">
                {/* Page number information */}
                <span
                    className="text-white/65 flex-1 text-sm whitespace-nowrap"
                    aria-live="polite"
                >
                    Page {table.getState().pagination.pageIndex + 1} of {table.getPageCount()}
                </span>

                {/* Pagination buttons */}
                <div className="grow">
                    <Pagination>
                        <PaginationContent>
                            {/* Previous page button */}
                            <PaginationItem>
                                <Button
                                    size="icon"
                                    variant="outline"
                                    className="disabled:pointer-events-none disabled:opacity-50"
                                    onClick={() => table.previousPage()}
                                    disabled={!table.getCanPreviousPage()}
                                    aria-label="Go to previous page"
                                >
                                    <ChevronLeftIcon size={16} aria-hidden="true" />
                                </Button>
                            </PaginationItem>

                            {/* Left ellipsis (...) */}
                            {showLeftEllipsis && (
                                <PaginationItem>
                                    <PaginationEllipsis />
                                </PaginationItem>
                            )}

                            {/* Page number buttons */}
                            {pages.map((page) => {
                                const isActive =
                                    page === table.getState().pagination.pageIndex + 1
                                return (
                                    <PaginationItem key={page}>
                                        <Button
                                            size="icon"
                                            variant={`${isActive ? "outline" : "ghost"}`}
                                            onClick={() => table.setPageIndex(page - 1)}
                                            aria-current={isActive ? "page" : undefined}
                                        >
                                            {page}
                                        </Button>
                                    </PaginationItem>
                                )
                            })}

                            {/* Right ellipsis (...) */}
                            {showRightEllipsis && (
                                <PaginationItem>
                                    <PaginationEllipsis />
                                </PaginationItem>
                            )}

                            {/* Next page button */}
                            <PaginationItem>
                                <Button
                                    size="icon"
                                    variant="outline"
                                    className="disabled:pointer-events-none disabled:opacity-50"
                                    onClick={() => table.nextPage()}
                                    disabled={!table.getCanNextPage()}
                                    aria-label="Go to next page"
                                >
                                    <ChevronRightIcon size={16} aria-hidden="true" />
                                </Button>
                            </PaginationItem>
                        </PaginationContent>
                    </Pagination>
                </div>

                {/* Results per page */}
                <div className="flex flex-1 justify-end">
                    <Select
                        value={table.getState().pagination.pageSize.toString()}
                        onValueChange={(value) => {
                            table.setPageSize(Number(value))
                        }}
                        aria-label="Results per page"
                        >
                        <SelectTrigger
                            id="results-per-page"
                            className="w-fit whitespace-nowrap text-white/65"
                        >
                            <SelectValue placeholder="Select number of results" />
                        </SelectTrigger>
                        <SelectContent>
                            {[20, 25, 50].map((pageSize) => (
                                <SelectItem key={pageSize} value={pageSize.toString()}>
                                    {pageSize} / page
                                </SelectItem>
                            ))}
                        </SelectContent>
                    </Select>
                </div>
            </div>
        </div>
    )
}
