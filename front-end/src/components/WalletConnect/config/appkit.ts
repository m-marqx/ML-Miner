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

export const siweConfig = createSIWEConfig({
    getMessageParams: async () => ({
        domain: typeof window !== 'undefined' ? window.location.host : '',
        uri: typeof window !== 'undefined' ? window.location.origin : '',
        chains: chains.map((chain: AppKitNetwork) => parseInt(chain.id.toString())),
        statement: 'Log in to the Quantitative System',
    }),
    createMessage: ({ address, ...args }: SIWECreateMessageArgs) =>
        formatMessage(args, normalizeAddress(address)),
    getNonce: async () => {
        const nonce = await getCsrfToken();
        if (!nonce) {
            throw new Error('Failed to get nonce!');
        }

        return nonce;
    },
    getSession: async () => {
        const session = await getSession();
        if (!session) {
            return null;
        }

        // Validate address and chainId types
        if (typeof session.address !== "string" || typeof session.chainId !== "number") {
            return null;
        }

        return { address: session.address, chainId: session.chainId } satisfies SIWESession;
    },
    verifyMessage: async ({ message, signature }: SIWEVerifyMessageArgs) => {
        try {
            const success = await signIn('credentials', {
                message,
                redirect: false,
                signature,
                callbackUrl: '/protected',
            });

            return Boolean(success?.ok);
        } catch {
            return false;
        }
    },
    signOut: async () => {
        try {
            await signOut({
                redirect: false,
            });
            return true;
        } catch {
            return false;
        }
    },
});
