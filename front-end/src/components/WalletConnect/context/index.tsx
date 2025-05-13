'use client'

import { wagmiAdapter, projectId } from "../config/appkit"
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { createAppKit } from '@reown/appkit/react'
import { mainnet, polygon, arbitrum, base, scroll } from '@reown/appkit/networks'
import { cookieToInitialState, WagmiProvider } from 'wagmi';

import React, { type ReactNode } from 'react'

// Set up queryClient
const queryClient = new QueryClient()

if (!projectId) {
    throw new Error('Project ID is not defined')
}

const metadata = {
    name: 'Quantitative System',
    description: 'A quantitative trading system for crypto assets',
    url: 'https://archiemarques.com/',
    icons: ['https://avatars.githubusercontent.com/u/124513922']
}

export const modal = createAppKit({
    adapters: [wagmiAdapter],
    networks: [mainnet, polygon, arbitrum, base, scroll],
    projectId,
    // siweConfig, 
    defaultNetwork: polygon,
    metadata: metadata,
    features: {
        email: false,
        analytics: true,
        socials: false,
    },
    themeVariables: {
        "--w3m-accent": '#d87a16',
        "--w3m-qr-color": '#000',
    },
})

export default function ContextProvider({
    children,
    cookies,
}: {
    children: ReactNode;
    cookies: string | null;
}) {
    const initialState = cookieToInitialState(wagmiAdapter.wagmiConfig, cookies)

    return (
        <WagmiProvider config={wagmiAdapter.wagmiConfig} initialState={initialState}>
            <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>
        </WagmiProvider>
    )
}