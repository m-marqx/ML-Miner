"use client";

import React from "react";

import { SwapWidget } from "@/src/components/SwapWidget/swapWidget";
import { useAccount } from 'wagmi'
import Squares from "@/blocks/Backgrounds/Squares/Squares"

export default function Home() {
    const { address: walletAddress } = useAccount();

    return (
        <div className="relative w-full h-full">
            <Squares
                speed={0}
                squareSize={56}
                borderColor='#71717a40'
            />
            <div className="flex justify-center m-20 relative z-10 h-full" >
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