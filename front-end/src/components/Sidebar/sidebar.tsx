import type * as React from "react"
import { usePathname } from 'next/navigation';

import {
    Sidebar,
    SidebarContent,
    SidebarGroup,
    SidebarGroupContent,
    SidebarGroupLabel,
    SidebarMenu,
    SidebarMenuButton,
    SidebarMenuItem,
} from "@/components/ui/sidebar"
import { BarChart3, ArrowLeftRight, Wallet, TrendingUp, } from "lucide-react"

import { useAccount, useBalance } from 'wagmi'

// Navigation data based on the original interface
const navigationItems = [
    {
        title: "Swap",
        url: "/swap",
        icon: ArrowLeftRight,
    },
    {
        title: "AI Signals",
        url: "/ai-signals",
        icon: BarChart3,
    },
]

const walletItems = [
    {
        title: "Portfolio",
        url: "/portfolio",
        icon: TrendingUp,
    },
    {
        title: "Wallet",
        url: "/wallet",
        icon: Wallet,
    },
]

export function AppSidebar({ ...props }: React.ComponentProps<typeof Sidebar>) {
    const pathname = usePathname()

    const balance = () => {
        const { address } = useAccount();
        const { data, isError, isLoading } = useBalance({
            address: address,
            chainId: 137,
        });

        if (isLoading) return <span>Fetching balance…</span>;
        if (isError) return <span>Error fetching balance</span>;
        if (!data) return <span>No balance data</span>;
        return <span>{(Number(data.value) / 10 ** data.decimals).toFixed(4)} {data.symbol}</span>;
    }
    return (
        <Sidebar {...props} className="border-r border-[#27272a]/60">
            <SidebarContent className="bg-gradient-to-b from-[#18181B] to-[#0E0E11] text-white">
                <SidebarGroup>
                    <SidebarGroupLabel className="text-gray-400 text-xs uppercase tracking-wider font-medium ml-1">
                        Account
                    </SidebarGroupLabel>
                    <SidebarGroupContent>
                        <div className="px-4 py-3 mx-2 rounded-lg bg-[#27272a]/50 backdrop-blur-sm border border-[#27272a]/60 space-y-3">
                            <div className="flex items-center justify-center w-full">
                                <appkit-button balance="hide" />
                            </div>
                            <div className="h-px bg-[#27272a]/60"></div>
                            <div className="flex items-center justify-between">
                                <div className="flex items-center gap-2">
                                    <div className="w-2 h-2 bg-purple-500 rounded-full shadow-sm shadow-purple-500/50"></div>
                                    <span className="text-sm font-medium">POL</span>
                                </div>
                                <span className="text-sm font-semibold text-white">{balance()}</span>
                            </div>
                        </div>
                    </SidebarGroupContent>
                </SidebarGroup>
                <div className="h-px bg-gradient-to-r from-transparent via-[#27272a]/50 to-transparent my-3 mx-3"></div>
                <SidebarGroup>
                    <SidebarGroupLabel className="text-gray-400 text-xs uppercase tracking-wider font-medium ml-1">
                        Trade
                    </SidebarGroupLabel>
                    <SidebarGroupContent>
                        <SidebarMenu>
                            {navigationItems.map((item) => (
                                <SidebarMenuItem key={item.title}>
                                    <SidebarMenuButton
                                        asChild
                                        isActive={pathname === item.url}
                                        variant={"gradient"}
                                    >
                                        <a href={item.url} className="flex items-center gap-3">
                                            <item.icon className="h-4 w-4" />
                                            <span className="font-semibold">{item.title}</span>
                                            {pathname === item.url && <div className="ml-auto w-2 h-2 rounded-full bg-white"></div>}
                                        </a>
                                    </SidebarMenuButton>
                                </SidebarMenuItem>
                            ))}
                        </SidebarMenu>
                    </SidebarGroupContent>
                </SidebarGroup>
                <div className="h-px bg-gradient-to-r from-transparent via-[#27272a]/50 to-transparent my-3 mx-3"></div>
                <SidebarGroup>
                    <SidebarGroupLabel className="text-gray-400 text-xs uppercase tracking-wider font-medium ml-1">
                        Wallet
                    </SidebarGroupLabel>
                    <SidebarGroupContent>
                        <SidebarMenu>
                            {walletItems.map((item) => (
                                <SidebarMenuItem key={item.title}>
                                    <SidebarMenuButton
                                        asChild
                                        isActive={pathname === item.url}
                                        variant={"gradient"}
                                    >
                                        <a href={item.url} className="flex items-center gap-3">
                                            <item.icon className="h-4 w-4" />
                                            <span className="font-semibold">{item.title}</span>
                                            {pathname === item.url && <div className="ml-auto w-2 h-2 rounded-full bg-white"></div>}
                                        </a>
                                    </SidebarMenuButton>
                                </SidebarMenuItem>
                            ))}
                        </SidebarMenu>
                    </SidebarGroupContent>
                </SidebarGroup>
                <div className="h-px bg-gradient-to-r from-transparent via-[#27272a]/50 to-transparent my-3 mx-3"></div>
            </SidebarContent>
        </Sidebar>
    )
}
