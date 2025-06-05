"use client";

import React from "react";

import { SwapWidget } from "@/src/components/SwapWidget/swapWidget";
import { useAccount } from 'wagmi'
import Squares from "@/blocks/Backgrounds/Squares/Squares"
import { AppSidebar } from "@/src/components/Sidebar/sidebar";
import { SidebarInset, SidebarProvider } from "@/components/ui/sidebar"

export default function Home() {
    const { address: walletAddress } = useAccount();

    if (!walletAddress) {
        return (
            <div className="relative w-full h-full">
                <Squares
                    speed={0}
                    squareSize={56}
                    borderColor='#71717a40'
                />
                <div className="grid grid-cols-[1fr_2fr] h-[calc(100%-2rem)] gap-[3.4svw] m-[3svh_2.7svw] justify-center relative z-10" >
                    <div></div>
                    <div className="w-140">
                        <SwapWidget />
                    </div>
                </div>
            </div>
        )
    }

    return (
        <SidebarProvider open={true} defaultOpen={true}>
            <AppSidebar />
        <SidebarInset className="w-0">
        <div className="relative w-full h-full flex flex-col justify-center items-center">
            <Squares
                speed={0}
                squareSize={56}
                borderColor='#71717a40'
            />
            <div className="w-fit h-fit items-center flex justify-center relative z-10" >
                <SwapWidget />
            </div>
                </div>
        </SidebarInset>
      </SidebarProvider >
    );
}