"use client";

import React from "react";
import { useEffect, useState, useCallback } from "react";
import Graph, { type GraphProps } from "../../components/BarGraph/Graph";
import TableData from "@/src/components/Table/ModelTable";
import { AppSidebar } from "@/src/components/Sidebar/sidebar";
import { SidebarInset, SidebarProvider } from "@/components/ui/sidebar";

interface ModelRecommendation {
  date: string | null;
  model_33139: string | null;
}

export default function Home() {
  const [isLoading, setIsLoading] = useState(true);
  const [btcXValues, setBtcXValues] = useState<string[]>([]);
  const [btcYValues, setBtcYValues] = useState<number[]>([]);
  const [modelXValues, setModelXValues] = useState<string[]>([]);
  const [modelYValues, setModelYValues] = useState<number[]>([]);
  const [tableData, setTableData] = useState<ModelRecommendation[]>([]);
  const [tableLoading, setTableLoading] = useState(true);

  const fetchTableData = useCallback(async () => {
    try {
      setTableLoading(true);
      const response = await fetch("/api/tableData");

      if (!response.ok) {
        throw new Error("Failed to fetch table data");
      }

      const result = await response.json();
      setTableData(result.data || []);
    } catch (error) {
      console.error("Error fetching table data:", error);
    } finally {
      setTableLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchTableData();
    const intervalId: NodeJS.Timeout | null = null;

    return () => {
      if (intervalId) {
        clearInterval(intervalId);
      }
    };
  }, [fetchTableData]);

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
    watermarkLocation: "52%",
    watermarkFont: "2rem Inter",
    chartBgColor: "#18181B",
    className: "rounded-md bg-card-color h-full px-4 border",
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
    watermarkLocation: "52%",
    watermarkFont: "2rem Inter",
    chartBgColor: "#18181B",
    className: "rounded-md bg-card-color h-full px-4 border",
  };

  return (
    <div>
      <SidebarProvider open={true} defaultOpen={true}>
        <AppSidebar />
        <SidebarInset className="w-0">
          <div className="grid grid-cols-[2fr_1fr] h-220 gap-6 mx-6">
            <div className="grid grid-cols-1 my-6 gap-6">
              <div className="h-110">
                <Graph {...modelGraphConfig} />
              </div>
              <div className="h-110">
                <Graph {...BtcGraphConfig} />
              </div>
            </div>
            <div className="flex flex-col content-between my-6">
              {tableLoading ? (
                <div>Loading table data...</div>
              ) : tableData.length > 0 ? (
                <div className="h-226">
                  <TableData />
                </div>
              ) : (
                <div>No table data available</div>
              )}
            </div>
          </div>
        </SidebarInset>
      </SidebarProvider>
    </div>
  );
}
