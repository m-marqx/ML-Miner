"use client";

import React from "react";

import { useEffect, useState, useCallback } from "react";
import { SwapWidget } from "@/src/components/SwapWidget/swapWidget";
import { useAccount } from 'wagmi'
import Squares from "@/blocks/Backgrounds/Squares/Squares"
import { useAppKit } from "@reown/appkit/react";
import { Button } from "@/components/ui/button";
import AgGridClient from "@/src/components/Table/AgGridClient";

interface ModelRecommendation {
    date: string | null;
    model_33139: string | null;
}

export default function Home() {
    const [tableData, setTableData] = useState<ModelRecommendation[]>([]);
    const [tableLoading, setTableLoading] = useState(true);

    const fetchTableData = useCallback(async () => {
        try {
            setTableLoading(true);
            const response = await fetch("/api/tableData");

            if (!response.ok) {
                throw new Error("Failed to fetch table data");
            }

            const result = await response.json();
            setTableData(result.data || []);
        } catch (error) {
            console.error("Error fetching table data:", error);
        } finally {
            setTableLoading(false);
        }
    }, []);

    useEffect(() => {
        fetchTableData();
        const intervalId: NodeJS.Timeout | null = null;

        return () => {
            if (intervalId) {
                clearInterval(intervalId);
            }
        };
    }, [fetchTableData]);

    const { address: walletAddress } = useAccount();
    const { open } = useAppKit();

    useEffect(() => {
        if (!walletAddress) {
            open({ view: "Connect" });
        }
    }, [walletAddress, open]);

    if (!walletAddress) {
        return (
            <div className="relative w-full h-full">
                <Squares
                    speed={0}
                    squareSize={56}
                    borderColor='#71717a40'
                />
                <div className="flex justify-center relative z-10" >
                    <Button onClick={() => open({ view: "Connect" })}>Connect Wallet</Button>
                </div>
            </div>
        )
    }

    return (
        <div className="relative w-full h-full">
            <Squares
                speed={0}
                squareSize={56}
                borderColor='#71717a40'
            />
            <div className="grid grid-cols-[1fr_2fr] h-[calc(100%-2rem)] gap-[3.4svw] m-[3svh_2.7svw] justify-center relative z-10" >
                <div className="pb-6">
                    {tableLoading ? (
                        <div>Loading table data...</div>
                    ) : tableData.length > 0 ? (
                        <AgGridClient rowData={tableData} />
                    ) : (
                        <div>No table data available</div>
                    )}
                </div>
                <div className="w-140">
                    {walletAddress ? (
                        <SwapWidget />
                    ) : (
                        <div>Please connect your wallet</div>
                    )}
                </div>
            </div>
        </div>
    );
}