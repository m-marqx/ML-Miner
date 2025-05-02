import type { NextConfig } from "next";
import { hostname } from "os";

const nextConfig: NextConfig = {
  experimental: {
    authInterrupts: true,
  },
  images: {
    remotePatterns: [
      {
        protocol: 'https',
        hostname: 'raw.githubusercontent.com',
        pathname: '/trustwallet/assets/**',
      },
      {
        protocol: 'https',
        hostname: 'raw.githubusercontent.com',
        pathname: '/sameepsi/quickswap-interface/master/assets/**',
      },
      {
        protocol: 'https',
        hostname: 'raw.githubusercontent.com',
        pathname: '/wormhole-foundation/wormhole-token-list/main/assets/**',
      },
      {
        protocol: 'https',
        hostname: 's2.coinmarketcap.com',
        pathname: '/static/img/coins/**',
      },
      {
        protocol: 'https',
        hostname: 'raw.githubusercontent.com',
        pathname: '/uniswap/assets/master/blockchains/ethereum/assets/**',
      },
      {
        protocol: 'https',
        hostname: 'etherscan.io',
        pathname: '/token/images/**',
      },
      {
        protocol: 'https',
        hostname: 'tokens.1inch.exchange',
        pathname: '/**',
      },

      {
        protocol: 'https',
        hostname: 'assets.coingecko.com',
        pathname: '/coins/images/**',
      },
      {
        protocol: 'https',
        hostname: 'coin-images.coingecko.com',
        pathname: '/coins/images/**',
      },
      {
        protocol: 'https',
        hostname: 'arbitrum.foundation',
        pathname: '/logo.png',
      },
    ],
  },
};

export default nextConfig;
