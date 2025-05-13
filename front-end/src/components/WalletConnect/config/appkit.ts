import {
    type SIWESession,
    type SIWEVerifyMessageArgs,
    type SIWECreateMessageArgs,
    createSIWEConfig,
    formatMessage,
} from '@reown/appkit-siwe'
import { WagmiAdapter } from '@reown/appkit-adapter-wagmi'
import { getCsrfToken, getSession, signIn, signOut } from 'next-auth/react';
import { cookieStorage, createStorage } from '@wagmi/core'

import { arbitrum, mainnet, sepolia, optimism, type AppKitNetwork, polygon } from '@reown/appkit/networks'
import { getAddress } from 'viem';

export const projectId = process.env.NEXT_PUBLIC_PROJECT_ID;

if (!projectId) throw new Error('Project ID is not defined');

export const metadata = {
    name: 'Quantitative System - SIWE',
    description: 'Quantitative System - SIWE',
    url: 'https://archiemarques.com/',
    icons: ["https://avatars.githubusercontent.com/u/179229932"],
};

// Create wagmiConfig
export const chains: [AppKitNetwork, ...AppKitNetwork[]] = [mainnet, optimism, arbitrum, sepolia, polygon];

// 4. Create Wagmi Adapter
export const wagmiAdapter = new WagmiAdapter({
    storage: createStorage({
        storage: cookieStorage
    }),
    ssr: true,
    projectId,
    networks: chains,
});

export const config = wagmiAdapter.wagmiConfig

// Normalize the address (checksum)
const normalizeAddress = (address: string): string => {
    try {
        const splitAddress = address.split(':');
        const extractedAddress = splitAddress[splitAddress.length - 1] || address;
        const checksumAddress = getAddress(extractedAddress);
        splitAddress[splitAddress.length - 1] = checksumAddress;
        const normalizedAddress = splitAddress.join(':');

        return normalizedAddress;
    } catch {
        return address;
    }
}