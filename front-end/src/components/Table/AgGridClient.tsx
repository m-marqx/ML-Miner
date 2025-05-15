"use client";

import React, { useState, useEffect } from "react";
import type { ColDef } from "ag-grid-community";
import { AllCommunityModule, ModuleRegistry } from "ag-grid-community";
import { AgGridReact } from "ag-grid-react";

ModuleRegistry.registerModules([AllCommunityModule]);

interface ModelRecommendation {
    date: string | null;
    model_33139: string | null;
}

interface AgGridClientProps {
    rowData: ModelRecommendation[];
}


export default function AgGridClient({ rowData }: AgGridClientProps) {
    const [colDefs, setColDefs] = useState<ColDef[]>([]);

    useEffect(() => {
        if (rowData.length > 0) {
            setColDefs(
                Object.keys(rowData[0]).map((key) => ({
                    field: key,
                    headerName: key === "model_33139" ? "Catboost Model" : "Date (UTC-3)",
                    sort: key === "date" ? "desc" : undefined,
                    cellRenderer: ({ value }: { value: string }) => {
                        const mainText = value.split("<b>")[0];
                        if (mainText?.includes("<span style='color: red'>")) {
                            return (
                                <b>
                                    <span style={{ color: "#f0d25d" }}>
                                        {
                                            mainText
                                                .split("<span style='color: red'>")[1]
                                                ?.split("<")[0]
                                        }{" "}
                                        (Unconfirmed)
                                    </span>
                                </b>
                            );
                        }

                        let extraContent = <b></b>;
                        let positionInfo = "";

                        const isOpen = value.includes("Open Position");
                        const isClose = value.includes("Close Position");

                        if (isOpen || isClose) {
                            positionInfo = value
                                .split(`<b><span className="text-white">`)[1]
                                ?.split("</b>")[0];
                        }

                        if (value.includes("Open Position")) {
                            extraContent = (
                                <b>
                                    <span style={{ color: "#00e676", fontWeight: 600 }}>
                                        {"\u00A0"}Open Position
                                    </span>
                                    <span className="text-white">{positionInfo}</span>
                                </b>
                            );
                        } else if (value.includes("Close Position")) {
                            extraContent = (
                                <b>
                                    <span style={{ color: "#ef5350", fontWeight: 600 }}>
                                        {"\u00A0"}Close Position
                                    </span>
                                    <span className="text-white">{positionInfo}</span>
                                </b>
                            );
                        }

                        return (
                            <div className="flex">
                                <span className="text-white">{mainText}</span>
                                {extraContent}
                            </div>
                        );
                    },
                })),
            );
        }
    }, [rowData]);

    const defaultColDef: ColDef = {
        flex: 1,
        filter: true,
    };

    return (
        <AgGridReact
            rowData={rowData}
            columnDefs={colDefs}
            defaultColDef={defaultColDef}
        />
    );
}
