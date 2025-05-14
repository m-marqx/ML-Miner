"use client";

import React, { useEffect, useRef } from "react";
import * as echarts from "echarts";

export interface GraphProps {
  chartTitle: string;
  showLine: boolean;
  xValues: string[];
  yValues: number[];
  showDataValues: boolean;
  arrowOption: boolean;
  isAbsolute: boolean;
  watermark: string;
  watermarkLocation: string;
  watermarkFont: string;
  chartBgColor: string;
  className?: string;
  style?: React.CSSProperties;
  animation?: boolean;
}

export default function Graph({
  chartTitle,
  showLine,
  xValues,
  yValues,
  showDataValues,
  watermark,
  arrowOption,
  isAbsolute,
  watermarkLocation,
  watermarkFont,
  chartBgColor,
  style,
  className,
  animation,
}: GraphProps) {
  // Referências do gráfico
  const chartRef = useRef<HTMLDivElement>(null);
  const chartInstance = useRef<echarts.ECharts | null>(null);
  const annual_watermark = useRef([{}]);

  const textColor = "rgba(255,255,255,1)";
  const watermarkColor = "rgba(255,255,255,0.12)";


  useEffect(() => {
    if (chartRef.current) {
      let data = xValues.map((value, index) => ({
        year: String(value),
        value: Number(yValues[index]),
      }));

      const data_values = data.map((d) => d.value);
      const totalDataSum = data_values.reduce(
        (acc, value) => Number(acc) + Number(value),
        0
      );
      const data_percentage = data_values.map(
        (value) => (Number(value) / Number(totalDataSum)) * 100
      );

      data = data.map((d, index) => ({
        year: d.year,
        value: isAbsolute ? d.value : (data_percentage[index] ?? 0),
      }));

      const neutralColor = "#f0d25d"

      const barColors: string[] = data.map(d => d.value > 0 ? "#2196f3" : "#ef5350");
      barColors[barColors.length - 1] = neutralColor;


      annual_watermark.current = [
        {
          type: "group",
          rotation: (Math.PI * 3) / 2,
          bounding: "raw",
          right: "6%",
          top: watermarkLocation ? `${watermarkLocation}%` : "23%",
          z: 100,
          children: [
            {
              type: "text",
              left: "center",
              top: "center",
              z: 100,
              style: {
                fill: watermarkColor,
                text: watermark,
                font: watermarkFont,
              },
            },
          ],
        },
      ];
      const options: echarts.EChartsOption = {
        title: {
          text: chartTitle,
          left: "center",
          top: "6%",
          textStyle: {
            fontSize: 24,
            fontWeight: "bold",
            color: textColor,
          },
        },
        grid: {
          top: "20%",
        },
        graphic: annual_watermark.current,
        xAxis: {
          type: "category",
          data: data.map((d) => d.year.toString()),
          axisLabel: {
            fontSize: 18,
            fontWeight: "bold",
            color: textColor,
          },
        },
        yAxis: {
          type: "value",
          axisLabel: {
            fontSize: 18,
            fontWeight: "bold",
            color: textColor,
          },
        },
        series: [
          {
            type: "bar",
            data: data.map((d) => d.value > 0 ? d.value : '-'),
            stack: "total",
            itemStyle: {
              color: (params: echarts.DefaultLabelFormatterCallbackParams) =>
                barColors[params.dataIndex],
              borderRadius: [15, 15, 0, 0],
            },
            label: {
              show: true,
              position: "top",
              formatter: showDataValues ? (
                params: echarts.DefaultLabelFormatterCallbackParams
              ) =>
                `${(params.value as number).toFixed(2)}${isAbsolute ? "" : "%"
                }` : "",
              color: textColor,
              fontWeight: "bold",
              fontSize: 14,
            },
            barWidth: "50%",
          },
          {
            type: "bar",
            data: data.map((d) => d.value < 0 ? d.value : '-'),
            stack: "total",
            itemStyle: {
              color: (params: echarts.DefaultLabelFormatterCallbackParams) =>
                barColors[params.dataIndex],
              borderRadius: [0, 0, 15, 15],
            },
            label: {
              show: true,
              position: "bottom",
              formatter: showDataValues ? (
                params: echarts.DefaultLabelFormatterCallbackParams
              ) =>
                `${(params.value as number).toFixed(2)}${isAbsolute ? "" : "%"
                }` : "",
              color: textColor,
              fontWeight: "bold",
              fontSize: 14,
            },
            barWidth: "50%",
          },
        ],
        backgroundColor: chartBgColor,
      } as echarts.EChartsOption;

      const chart = echarts.init(chartRef.current, 'dark');
      chart.setOption({ ...options, animation: animation });
      chartInstance.current = chart;

      return () => chart.dispose();
    }
  }, [
    chartTitle,
    chartBgColor,
    watermark,
    watermarkLocation,
    isAbsolute,
    arrowOption,
    showLine,
    xValues,
    yValues,
    annual_watermark,
  ]);

  return <div ref={chartRef} style={style} className={className} />;
}
