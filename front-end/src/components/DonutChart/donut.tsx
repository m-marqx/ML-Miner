import * as echarts from "echarts";
import React, { useRef, useEffect, useCallback, useState } from "react";

type EChartsOption = echarts.EChartsOption;

interface DonutChartProps {
    address?: string;
    className?: string;
    theme?: string | object;
    title?: string;
}

type PortfolioAsset = {
    token_address: string;
    name: string;
    symbol: string;
    thumbnail: string;
    usd_price: number;
    usd_value: number;
    usd_price_24hr_percent_change: number;
    amount: number;
};

export default function DonutChart({
    address = "",
    className = "w-full h-88",
    theme = "dark",
    title = "Portfolio Balances",
}: DonutChartProps) {
    const chartRef = useRef<HTMLDivElement>(null);
    const [data, setData] = useState<PortfolioAsset[]>([]);
    const fetchPortfolio = useCallback(async () => {
        const res = await fetch(`/api/portfolio?address=${address}`);
        const json = await res.json();
        setData(json.result || []);
    }, [address]);

    useEffect(() => {
        fetchPortfolio();
    }, [fetchPortfolio]);

    // Define stable coins based on the provided data, all Stablcoins contains "USD" in their symbol,  the only exception is DAI which is a stable coin but does not contain "USD" in its symbol.
    const stableCoinsSymbols = data
        .filter((asset) => asset.symbol.includes("USD") || asset.symbol === "DAI")
        .map((asset) => asset.symbol);

    const stableCoinValue = data
        .filter((asset) => stableCoinsSymbols.includes(asset.symbol))
        .reduce((acc, asset) => acc + asset.usd_value, 0);

    const nonStableCoinValue = data
        .filter((asset) => !stableCoinsSymbols.includes(asset.symbol))
        .reduce((acc, asset) => acc + asset.usd_value, 0);

    const chartData = [
        { value: stableCoinValue, name: "Stable", itemStyle: { color: "#d4d4d8" } },
        { value: nonStableCoinValue, name: "Tokens", itemStyle: { color: "#d87a16" } },
    ];

    useEffect(() => {
        let myChart: echarts.ECharts | null = null;
        if (chartRef.current) {
            myChart = echarts.init(chartRef.current, theme);

            const option: EChartsOption = {
                backgroundColor: "transparent",
                tooltip: {
                    trigger: "item",
                    formatter: "{b}: {d}%", // Tooltip showing name, value, and percentage
                },
                legend: {
                    top: "2%",
                    left: "center",
                    orient: "horizontal",
                    textStyle: {
                        color: "#fff", // Legend text color
                        fontSize: "1rem",
                    },
                },
                series: [
                    {
                        name: title,
                        type: "pie",
                        radius: ["55%", "70%"], // Donut shape
                        avoidLabelOverlap: false,
                        padAngle: 4,
                        itemStyle: {
                            borderRadius: "5%",
                        },
                        label: {
                            show: false, // Typically, labels for all slices are not shown directly on a donut
                            formatter: "{b}\n({d}%)", // Show name and percentage
                            fontSize: 14,
                            position: "center",
                        },
                        emphasis: {
                            label: {
                                show: true, // Show label in center on hover
                                fontSize: 24,
                                fontWeight: "bold",
                                formatter: "{b}\n({d}%)", // Show name and percentage in the center on hover
                            },
                        },
                        // labelLine: { // Not needed if labels are in the center or not shown by default
                        //     show: false,
                        // },
                        data: chartData,
                    },
                ],
            };
            myChart.setOption(option);

            const handleResize = () => myChart?.resize();
            window.addEventListener("resize", handleResize);

            return () => {
                if (myChart) {
                    myChart.dispose();
                }
                window.removeEventListener("resize", handleResize);
            };
        }
    }, [data, theme, title]); // Re-run effect if data, theme, or title changes

    return <div ref={chartRef} className={className}></div>;
}
