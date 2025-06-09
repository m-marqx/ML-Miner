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
import {
    Table,
    TableBody,
    TableCell,
    TableHead,
    TableHeader,
    TableRow,
} from "@/components/ui/table"

type PortfolioAsset = {
    token_address: string
    name: string
    symbol: string
    thumbnail: string
    usd_price: number
    usd_value: number
    usd_price_24hr_percent_change: number
    amount: number
}

const columns: ColumnDef<PortfolioAsset>[] = [
    {
        header: "Logo",
        accessorKey: "thumbnail",
        cell: ({ row }) => (
            <img
                src={row.getValue("thumbnail")}
                alt={row.getValue("name")}
                className="w-8 h-8 rounded mx-auto"
            />
        ),
    },
    {
        header: "Name",
        accessorKey: "name",
    },
    {
        header: "Symbol",
        accessorKey: "symbol",
    },
    {
        header: "Price",
        accessorKey: "usd_price",
        cell: ({ row }) => {
            const value = Number(row.getValue("usd_price"));
            return `$${value.toFixed(2)}`;
        },
    },
    {
        header: "Amount",
        accessorKey: "balance_formatted",
        cell: ({ row }) => {
            const balance_formatted = row.getValue("balance_formatted") as string | undefined;
            const symbol = row.getValue("symbol");
            if (balance_formatted && balance_formatted.length >= 12) {
                // If the formatted balance is too long, truncate it to 12 characters
                return `${balance_formatted.slice(0, 12)}... ${symbol}`;
            }
            return `${Number(balance_formatted ?? "0")} ${symbol}`;
        }
    },
    {
        header: "USD Value",
        accessorKey: "usd_value",

        cell: ({ row }) => {
            const value = Number(row.getValue("usd_value"));
            return `$${value.toFixed(2)}`;
        },
    },
    {
        header: "24h Change",
        accessorKey: "usd_price_24hr_percent_change",
        cell: ({ row }) => {
            const value = row.getValue("usd_price_24hr_percent_change")
            const num = Number(value)
            return (
                <span className={num < 0 ? "text-red-500" : "text-green-500"}>
                    {num.toFixed(2)}%
                </span>
            )
        },
    },
]

export default function PortfolioTable({ address }: { address: string }) {
    const [data, setData] = useState<PortfolioAsset[]>([])
    const [loading, setLoading] = useState(false)
    const [sorting, setSorting] = useState<SortingState>([
        { id: "usd_value", desc: true },
    ])
    const containerRef = useRef<HTMLDivElement>(null)

    // Infinite scroll handler
    useEffect(() => {
        const handleScroll = () => {
            const el = containerRef.current
            if (!el || loading) return
        }
        const el = containerRef.current
        if (el) el.addEventListener("scroll", handleScroll)
        return () => {
            if (el) el.removeEventListener("scroll", handleScroll)
        }
    }, [])

    const fetchPortfolio = useCallback(async () => {
        setLoading(true)
        const res = await fetch(`/api/portfolio?address=${address}`)
        const json = await res.json()
        setData(json.result || [])
        setLoading(false)
    }, [address])

    useEffect(() => {
        fetchPortfolio()
    }, [fetchPortfolio])

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
        <div className="flex flex-col h-88">
            <div className="bg-zinc-900 rounded-md flex-1 min-h-0 flex flex-col">
                <Table className="table-fixed w-full">
                    <TableHeader className="text-center rounded-md hover:bg-card-color">
                        {table.getHeaderGroups().map((headerGroup) => (
                            <TableRow key={headerGroup.id} className="hover:bg-card-color">
                                {headerGroup.headers.map((header) => {
                                    return (
                                        <TableHead
                                            key={header.id}
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
                    <Table className="table-fixed w-full scrollbar-subtle">
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
