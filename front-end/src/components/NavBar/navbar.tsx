'use client'

import React from 'react';
import { usePathname } from 'next/navigation';
import Link from 'next/link';
import styles from './navbar.module.css';

export default function NavBar() {
    const pathname = usePathname();

    return (
        <div className={styles.navLinks}>
            <Link href="/swap" className={pathname === '/swap' ? styles.active : ''}>
                swap
            </Link>
            <Link href="/result" className={pathname === '/result' ? styles.active : ''}>
                result
            </Link>
        </div>
    );
}
