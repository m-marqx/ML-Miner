import React, { useEffect, useRef } from 'react';

declare global {
    interface Window {
        createMyWidget?: (containerId: string, options: Record<string, unknown>) => void;
    }
}

const PRICE_CHART_ID = 'price-chart-widget-container';

interface PriceChartProps {
    pairAddress: string;
}

export const PriceChart: React.FC<PriceChartProps> = ({ pairAddress }) => {
    const containerRef = useRef(null);

    useEffect(() => {
        if (typeof window === 'undefined') return;

        const loadWidget = () => {
            if (typeof window.createMyWidget === 'function') {
                window.createMyWidget(PRICE_CHART_ID, {
                    autoSize: true,
                    chainId: '0x89',
                    pairAddress: pairAddress,
                    showHoldersChart: true,
                    defaultInterval: '1D',
                    timeZone: Intl.DateTimeFormat().resolvedOptions().timeZone ?? 'Etc/UTC',
                    theme: 'moralis',
                    locale: 'en',
                    backgroundColor: '#071321',
                    gridColor: '#0d2035',
                    textColor: '#68738D',
                    candleUpColor: '#4CE666',
                    candleDownColor: '#E64C4C',
                    hideLeftToolbar: false,
                    hideTopToolbar: false,
                    hideBottomToolbar: false
                });
            } else {
                console.error('createMyWidget function is not defined.');
            }
        };

        if (!document.getElementById('moralis-chart-widget')) {
            const script = document.createElement('script');
            script.id = 'moralis-chart-widget';
            script.src = 'https://moralis.com/static/embed/chart.js';
            script.type = 'text/javascript';
            script.async = true;
            script.onload = loadWidget;
            script.onerror = () => {
                console.error('Failed to load the chart widget script.');
            };
            document.body.appendChild(script);
        } else {
            loadWidget();
        }
    }, [pairAddress]);

    return (
        <div className='w-full h-full'>
            <div
                id={PRICE_CHART_ID}
                ref={containerRef}
                className='w-full h-full'
            />
        </div>
    );
};
