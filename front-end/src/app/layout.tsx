import React from 'react';
import type { Metadata } from 'next'
import '@reown/appkit-wallet-button/react'
import { headers } from 'next/headers'
import ContextProvider from '../components/WalletConnect/context';
import "../styles/globals.css";
import { SpeedInsights } from "@vercel/speed-insights/next"
import { Analytics } from "@vercel/analytics/next"

export const metadata: Metadata = {
  title: 'Quantitative System',
  description: 'Powered by Archie Marques'
}

export default async function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  const headersData = await headers();
  const cookies = headersData.get('cookie');
  return (
    <html lang="en">
      <SpeedInsights />
      <Analytics />
      <body className="bg-background-color w-full h-full text-primary-text-color">
        <ContextProvider cookies={cookies}>
          {children}
        </ContextProvider>
      </body>
    </html>
  );
}