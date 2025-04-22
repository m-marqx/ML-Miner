import { NextResponse } from 'next/server';
import { db } from "../../../server/db";
import { modelRecommendations } from "../../../server/db/schema";

export async function GET() {
    try {
        const data = await db.select().from(modelRecommendations)
        return NextResponse.json({ data });
    } catch (error) {
        console.error("Error fetching table data:", error);
        return NextResponse.json({ error: "Failed to fetch data", details: String(error) }, { status: 500 });
    }
}