import { NextResponse } from "next/server"

export async function GET(request: Request) {
    const { searchParams } = new URL(request.url)
    const address = searchParams.get("address")

    if (!address) {
        return NextResponse.json({ error: "Missing address" }, { status: 400 })
    }

    const apiKey = process.env.MORALIS_API_KEY
    if (!apiKey) {
        return NextResponse.json({ error: "Missing Moralis API key" }, { status: 500 })
    }

    const url = `https://deep-index.moralis.io/api/v2.2/wallets/${address}/history?chain=polygon&order=DESC`

    try {
        const moralisRes = await fetch(url, {
            method: "GET",
            headers: {
                "Content-Type": "application/json",
                "X-API-Key": apiKey,
            },
        })

        if (!moralisRes.ok) {
            return NextResponse.json({ error: "Failed to fetch from Moralis" }, { status: moralisRes.status })
        }

        const data = await moralisRes.json()

        const swaps = Array.isArray(data.result)
            ? data.result.filter((item: any) => item.category === "token swap")
            : []

        return NextResponse.json({ result: swaps })
    } catch (err) {
        return NextResponse.json({ error: "Internal server error" }, { status: 500 })
    }
}