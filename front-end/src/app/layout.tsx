import type { Metadata } from 'next'
import '@reown/appkit-wallet-button/react'
import { headers } from 'next/headers'
import ContextProvider from '../components/WalletConnect/context';
import styles from './layout.module.css';
import "../styles/globals.css";
import NavBar from '../components/NavBar/navbar';

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
          <div className={styles.navBar}>
            <div className={styles.logoContainer}>
              <img
                src="https://avatars.githubusercontent.com/u/124513922?s=400&u=d374d7671e2f3d6f8d73a3cc0a8b5c6f702643ff"
                alt="Logo" className={styles.logo}
              />
            </div>
            <NavBar />
            <div className={styles.web3Button}>
              <appkit-button />
            </div>
          </div>
          {children}
        </ContextProvider>
    </body>
    </html>
  );
}