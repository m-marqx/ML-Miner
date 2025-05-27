"use client"

import React, { useEffect, useState, useRef, useCallback } from "react"
import {
    ColumnDef,
    flexRender,
    getCoreRowModel,
    getSortedRowModel,
    SortingState,
    useReactTable,
} from "@tanstack/react-table"
import { ChevronDownIcon, ChevronUpIcon } from "lucide-react"
import { cn } from "@/lib/utils"
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
            const isLastMinute = date.getHours() === 23 && date.getMinutes() === 59 && date.getSeconds() === 59
            if (!isLastMinute) {
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

export default function ModelTable() {
    const [data, setData] = useState<TableData[]>([])
    const [page, setPage] = useState(1)
    const [hasMore, setHasMore] = useState(true)
    const [loading, setLoading] = useState(false)
    const [sorting, setSorting] = useState<SortingState>([
        { id: "date", desc: true },
    ])
    const containerRef = useRef<HTMLDivElement>(null)

    // Fetch data
    const fetchTableData = useCallback(async () => {
        if (loading || !hasMore) return
        setLoading(true)
        const res = await fetch(`/api/tableData?page=${page}`)
        const json = await res.json()
        if (json.data.length === 0) setHasMore(false)
        setData(json.data)
        setLoading(false)
    }, [loading, hasMore, page])

    useEffect(() => {
        fetchTableData()
    }, [])

    // Infinite scroll handler
    useEffect(() => {
        const handleScroll = () => {
            const el = containerRef.current
            if (!el || loading || !hasMore) return
            if (el.scrollHeight - el.scrollTop - el.clientHeight < 100) {
                setPage(p => p + 1)
            }
        }
        const el = containerRef.current
        if (el) el.addEventListener("scroll", handleScroll)
        return () => {
            if (el) el.removeEventListener("scroll", handleScroll)
        }
    }, [])

    const table = useReactTable({
        data,
        columns,
        getCoreRowModel: getCoreRowModel(),
        getSortedRowModel: getSortedRowModel(),
        onSortingChange: setSorting,
        enableSortingRemoval: false,
        state: { sorting },
    })

    return (
        <div className="flex flex-col h-full">
            <div className="bg-zinc-900 rounded-md border flex-1 min-h-0 flex flex-col">
                <Table>
                    <TableHeader className="text-center rounded-md hover:bg-card-color">
                        {table.getHeaderGroups().map((headerGroup) => (
                            <TableRow key={headerGroup.id} className="hover:bg-card-color">
                                {headerGroup.headers.map((header) => {
                                    return (
                                        <TableHead
                                            key={header.id}
                                            style={{ width: `${header.getSize()}px` }}
                                            className="h-11 hover:bg-zinc-700 rounded-md"
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
                </Table>
                {/* Scrollable rows container */}
                <div
                    ref={containerRef}
                    className="flex-1 min-h-0 overflow-y-auto rounded-md scrollbar-subtle"
                >
                    <Table className="scrollbar-subtle">
                        <TableBody>
                            {table.getRowModel().rows?.length ? (
                                table.getRowModel().rows.map((row) => (
                                    <TableRow
                                        key={row.id}
                                        data-state={row.getIsSelected() && "selected"}
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
                                            {loading ? "Loading..." : "No results."}
                                    </TableCell>
                                </TableRow>
                            )}
                        </TableBody>
                    </Table>
                </div>
            </div>
        </div>
    )
}
