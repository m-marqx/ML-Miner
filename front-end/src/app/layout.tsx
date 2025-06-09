import React from 'react';
import type { Metadata } from 'next'
import '@reown/appkit-wallet-button/react'
import { headers } from 'next/headers'
import ContextProvider from '../components/WalletConnect/context';
import styles from './layout.module.css';
import "../styles/globals.css";

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
      <body className={styles.main}>
        <ContextProvider cookies={cookies}>
          {children}
        </ContextProvider>
      </body>
    </html>
  );
}