"use client";

import React, { useState, useEffect } from "react";

import PortfolioTable from "@/src/components/Table/PortfolioTable";
import TransactionTable from "@/src/components/Table/TransactionTable";

import { AppSidebar } from "@/src/components/Sidebar/sidebar";
import { SidebarInset, SidebarProvider } from "@/components/ui/sidebar";
import { useAccount } from "wagmi";
import DonutChart from "@/src/components/DonutChart/donut";

export default function PortfolioPage() {
    const [address, setAddress] = useState("");
    const { address: accountAddress } = useAccount();
    useEffect(() => {
        setAddress(accountAddress || "");
    }, [accountAddress]);

    return (
        <SidebarProvider open={true} defaultOpen={true}>
            <AppSidebar />
            <SidebarInset className="w-0 justify-center">
                <div className="grid grid-cols-2 gap-4 p-6 justify-center">
                    <div className="grid gap-4">
                        <div className="bg-card-color border rounded-md">
                            <span className="text-2xl grid text-center justify-center my-1">Token Balances</span>
                            <PortfolioTable address={address} />
                        </div>
                        <div className="bg-card-color border rounded-md">
                            <span className="text-2xl grid text-center justify-center my-1">
                                Allocation
                            </span>
                            <DonutChart address={address} />
                        </div>
                    </div>
                    <div className="bg-card-color border rounded-md">
                        <span className="text-2xl grid text-center justify-center my-1">Transactions</span>
                        <TransactionTable address={address} />
                    </div>
                </div>
            </SidebarInset>
        </SidebarProvider>
    );
}
