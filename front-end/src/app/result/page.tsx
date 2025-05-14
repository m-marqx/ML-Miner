"use client";

import React from "react";
import { useEffect, useState, useCallback } from "react";
import styles from "./page.module.css";
import Graph, { type GraphProps } from "../../components/BarGraph/Graph";

export default function Home() {
  const [isLoading, setIsLoading] = useState(true);
  const [btcXValues, setBtcXValues] = useState<string[]>([]);
  const [btcYValues, setBtcYValues] = useState<number[]>([]);
  const [modelXValues, setModelXValues] = useState<string[]>([]);
  const [modelYValues, setModelYValues] = useState<number[]>([]);

  const fetchChartData = useCallback(async () => {
    try {
      setIsLoading(true);
      await fetch("/api/walletBalance")
        .then((response) => {
          if (!response.ok) {
            throw new Error(
              "Network response was not ok " + response.statusText,
            );
          }
          return response.json();
        })
        .then((data) => {
          const dataXValues = data.data.map((item) =>
            new Date(item.date).toLocaleDateString("pt-BR", {
              year: "numeric",
              month: "2-digit",
            }),
          );

          let btcDataXValues = dataXValues;

          let btcDataYValues = data.data.map((item) => item.btc);
          // Remove the last value if it is 0
          if (btcDataYValues.at(-1) === 0) {
            btcDataYValues = btcDataYValues.slice(0, -1);
            btcDataXValues = btcDataXValues.slice(0, -1);
          }
          setBtcXValues(btcDataXValues);
          setBtcYValues(btcDataYValues);

          let modelDataXValues = dataXValues;
          let modelDtaYValues = data.data.map((item) => item.modelo);
          // Remove the last value if it is 0
          if (modelDtaYValues.at(-1) === 0) {
            modelDtaYValues = modelDtaYValues.slice(0, -1);
            modelDataXValues = modelDataXValues.slice(0, -1);
          }
          setModelXValues(modelDataXValues);
          setModelYValues(modelDtaYValues);

          setIsLoading(false);
          return data;
        })
        .catch((error) => {
          console.error("Error fetching data:", error);
          setIsLoading(false);
          return null;
        });
    } catch (error) {
      console.error("Error while fetching chart data:", error);
      setIsLoading(false);
      return null;
    }
  }, []);

  useEffect(() => {
    fetchChartData();

    const intervalId: NodeJS.Timeout | null = null;

    return () => {
      if (intervalId) {
        clearInterval(intervalId);
      }
    };
  }, [fetchChartData]);

  if (isLoading) return <div>Carregando...</div>;
  if (
    !btcXValues.length ||
    !btcYValues.length ||
    !modelXValues.length ||
    !modelYValues.length
  )
    return <div>Erro ao carregar os dados.</div>;

  const BtcGraphConfig: GraphProps = {
    chartTitle: "Bitcoin monthly changes (%)",
    xValues: btcXValues,
    yValues: btcYValues,
    showLine: false,
    showDataValues: true,
    arrowOption: false,
    isAbsolute: true,
    watermark: "Archie Marques",
    watermarkLocation: "45%",
    watermarkFont: "2rem Inter",
    chartBgColor: "#18181B",
    className: styles.graph,
  };

  const modelGraphConfig: GraphProps = {
    chartTitle: "CatBoost Model monthly performance (%)",
    xValues: modelXValues,
    yValues: modelYValues,
    showLine: false,
    showDataValues: true,
    arrowOption: false,
    isAbsolute: true,
    watermark: "Archie Marques",
    watermarkLocation: "45%",
    watermarkFont: "2rem Inter",
    chartBgColor: "#18181B",
    className: styles.graph,
  };

  const monthly_summary = (
    <div className="bg-[#18181b] text-[var(--text-color)] rounded-[1svh] border-[0.08svh] border-solid border-[var(--main-color)] my-[3svh]">
      <div className="flex justify-between items-center border-b-[0.08svh] border-solid border-b-[var(--main-color)] h-[5.6%]">
        <h2 className="text-2xl leading-[calc(1.5rem+8px)] font-semibold text-center px-8 opacity-85 m-0">Monthly Summary</h2>
        <div className="flex justify-between items-center bg-[var(--pallete-color-950)] rounded-[1svw] border-[0.08svh] border-solid border-[var(--main-color)] mr-8">
          <p className="text-base leading-[calc(1rem+8px)] text-[var(--pallete-color-200)] font-semibold px-8 m-0">April - 2025</p>
        </div>
      </div>
      <div className="grid grid-rows-3 text-left h-[calc(100%-5.6%)]">
        <div className="text-base leading-[calc(1rem+8px)] mx-8 border-b-[0.08svh] border-solid border-b-[#444]">
          <h3 className="text-2xl leading-[calc(1.25rem+8px)] text-[var(--pallete-color-500)] font-semibold opacity-85 py-4 text-center mb-2.5">AI Model</h3>
          <p className="leading-[calc(1rem+8px)]">
            The AI Model demonstrated more moderate fluctuations compared to BTC. It
            showed impressive growth in November 2024 (+25.68%) followed by a
            correction in December (-3.52%) and a recovery in January (+8.8%).
          </p>
          <br />
          <p className="leading-[calc(1rem+8px)]">
            After a significant decline in February (-13.22%), March 2025 marked a
            notable recovery with +6.78% growth.
          </p>
        </div>
        <div className="text-base leading-[calc(1rem+8px)] mx-8 border-b-[0.08svh] border-solid border-b-[#444]">
          <h3 className="text-2xl leading-[calc(1.25rem+8px)] text-[var(--pallete-color-500)] font-semibold opacity-85 py-4 text-center mb-2.5">BTC</h3>
          <p className="leading-[calc(1rem+8px)]">
            BTC showed considerable volatility over the analyzed period. After a
            strong performance in late 2024 (+34.55% in November), it experienced a
            significant downturn in February 2025 (-28.2%).
          </p>
          <br />
          <p className="leading-[calc(1rem+8px)]">
            {`For March 2025, BTC recorded a slight decline of -0.32%, showing some
            stabilization after February's dramatic drop.`}
          </p>
        </div>
        <div className="text-base leading-[calc(1rem+8px)] mx-8 last:border-b-0">
          <h3 className="text-2xl leading-[calc(1.25rem+8px)] text-[var(--pallete-color-500)] font-semibold opacity-85 py-4 text-center mb-2.5">Summary</h3>
          <p className="leading-[calc(1rem+8px)]">
            {`While BTC delivered higher returns during bullish periods (November:
            +34.55% vs +25.68%), the AI Model demonstrated superior downside
            protection during market stress (February: -13.22% vs BTC's -28.2%).`}
          </p>
          <br />
          <p className="leading-[calc(1rem+8px)]">
            {`The AI suggests stronger resilience and recovery potential compared to
            BTC's.`}
          </p>
        </div>
      </div>
    </div>
  )

  return (
    <div className={styles.main}>
      <div className="grid grid-cols-1 content-between my-[3svh]">
        <div className="h-[calc(var(--grid-height)/2-6svh)]">
          <Graph {...modelGraphConfig} />
        </div>
        <div className="h-[calc(var(--grid-height)/2-6svh)]">
          <Graph {...BtcGraphConfig} />
        </div>
      </div>
      {monthly_summary}
    </div>
  );
}