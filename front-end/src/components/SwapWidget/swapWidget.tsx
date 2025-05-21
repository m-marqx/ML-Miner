"use client";

import React from "react";
import { useEffect, useState, useCallback, useRef } from "react";
import { config } from "../WalletConnect/config/appkit";
import { sendTransaction } from "@wagmi/core";
import { ToggleGroup, ToggleGroupItem } from "@/components/ui/toggle-group";

import { useAppKit } from "@reown/appkit/react";
import { useAccount } from 'wagmi'

import axios from "axios";
import { getAccount, getBalance, estimateGas } from "@wagmi/core";
import Image from "next/image";
import { Button } from "@/components/ui/button";

import uniswapTokens from "./uniswap_tokens.json";

import {
    Dialog,
    DialogContent,
    DialogClose,
    DialogHeader,
    DialogTitle,
    DialogTrigger,
} from "@/components/ui/dialog";

import {
    Card,
    CardContent,
    CardFooter,
    CardHeader,
    CardTitle,
} from "@/components/ui/card";

import {
    Command,
    CommandEmpty,
    CommandGroup,
    CommandInput,
    CommandItem,
    CommandList,
} from "@/components/ui/command"


interface ParaSwapData {
    txParams: {
        to: `0x${string}`;
        data: `0x${string}`;
        value: string;
        gasPrice: string;
    };
    priceRoute: {
        destAmount: string;
        destDecimals: number;
        srcAmount: string;
        srcDecimals: number;
        srcUSD: number;
        destUSD: number;
        gasCostUSD: number;
    };
}

const tokenList = uniswapTokens.tokens

interface KyberSwapData {
    routerAddress: `0x${string}`;
    data: `0x${string}`;
    inputAmount: string;
    transactionValue: string;
    gasUsd: number;
}

interface TokenList {
    symbol: string;
    name: string;
    extensions: {
        bridgeInfo: Record<string, {
                tokenAddress: `0x${string}`;
            }>
    }
    address: `0x${string}`;
    decimals: number;
    logoURI: string;
    chainId: number;
}

async function getTokenBySymbol(
    symbol: string,
): Promise<TokenList> {
    const token = tokenList.find((item) => item.symbol === symbol);
    if (!token) return Promise.reject(`Token ${symbol} not found`);
    return token as unknown as TokenList;
}

const defaultTimerValue = 5;

export function SwapWidget() {
    const [paraSwapData, setParaSwapData] = useState<ParaSwapData | null>(null);
    const [btcPriceParaSwap, setBtcPriceParaSwap] = useState<number | null>(null);
    const [srcUSD, setSrcUSD] = useState<string>("");
    const [destUSD, setDestUSD] = useState<string>("");
    const inputTimeoutRef = useRef<NodeJS.Timeout | null>(null);

    const [kyberSwapData, setKyberSwapData] = useState<KyberSwapData | null>(null);
    const [btcPriceKyberSwap, setBtcPriceKyberSwap] = useState<number | null>(null);

    const [bestPriceText, setBestPriceText] = useState<string | null>(null);

    const [srcToken, setSrcToken] = useState<string>("WBTC");
    const [destToken, setDestToken] = useState<string>("USDT");
    const [srcTokenImage, setSrcTokenImage] = useState<string | null>("https://raw.githubusercontent.com/trustwallet/assets/master/blockchains/ethereum/assets/0x2260FAC5E5542a773Aa44fBCfeDf7C193bc2C599/logo.png");
    const [destTokenImage, setDestTokenImage] = useState<string | null>("https://raw.githubusercontent.com/trustwallet/assets/master/blockchains/ethereum/assets/0xdAC17F958D2ee523a2206206994597C13D831ec7/logo.png");
    const [srcBalanceValue, setSrcBalanceValue] = useState<string>("");
    const [destBalanceValue, setDestBalanceValue] = useState<string>("");
    const [srcAmount, setAmount] = useState<string | null>("1");
    const [destAmount, setDestAmount] = useState<string | undefined | null>(null);
    const [isValid, setIsValid] = useState<boolean>(false);
    const [secondsToNextFetch, setSecondsToNextFetch] = useState<number | null>(null);
    const [srcMaxDecimals, setSrcMaxDecimals] = useState<number>(4);
    const [destMaxDecimals, setDestMaxDecimals] = useState<number>(4);
    const [selectedPortion, setSelectedPortion] = useState<"33%" | "50%" | "max" | "">("33%");
    const srcInputRef = useRef<HTMLInputElement>(null);
    const destInputRef = useRef<HTMLInputElement>(null);

    const [lastInputChange, setLastInputChange] = useState<"src" | "dest">("src");
    const [textColor, setTextColor] = useState<string>("text-white/25");
    const [swapData, setSwapData] = useState<{
        to: `0x${string}`;
        data: `0x${string}`;
        value: bigint;
        maxFeePerGas: bigint;
        maxPriorityFeePerGas: bigint;
    } | null>(null);
    const [gasCost, setGasCost] = useState<bigint | undefined>(undefined);
    const [calculatedGasCost, setCalculatedGasCost] = useState<string | undefined>(undefined);
    const [srcRate, setSrcRate] = useState<string | undefined>(undefined);
    const [destRate, setDestRate] = useState<string | undefined>(undefined);
    const [isSrcRate, setIsSrcRate] = useState<boolean>(true);

    // whenever `srcAmount` changes we push it back into the DOM
    useEffect(() => {
        if (srcInputRef.current) {
            srcInputRef.current.value = srcAmount ?? "";
        }
    }, [srcAmount]);

    useEffect(() => {
        if (destInputRef.current) {
            destInputRef.current.value = destAmount ?? "";
        }
    }, [destAmount]);

    const account = getAccount(config);
    const walletAddress = account.address!;

    const { address: walletAccount } = useAccount();
    const { open } = useAppKit();

    const ConnectWalletButton = (
        <Button
            className="bg-main-color w-full rounded-3xl h-full"
            onClick={() => {
                open({ view: "Connect" });
            }}
        >
            <span className="text-large-size text-[#000000a6] font-semibold">Connect Wallet</span>
        </Button>
    )

    const abortControllerRef = useRef<AbortController | null>(null);

    const fetchSwapDataSell = useCallback(async () => {
        abortControllerRef.current?.abort();
        const controller = new AbortController();
        abortControllerRef.current = controller;
        const { signal } = controller;

        if (!srcToken || !destToken || !srcAmount) return;

        const symbolSrc = srcToken === "POL" ? "MATIC" : srcToken;
        const symbolDest = destToken === "POL" ? "MATIC" : destToken;

        const srcTokenData = await getTokenBySymbol(symbolSrc);
        srcTokenData.address = srcTokenData.extensions.bridgeInfo["137"].tokenAddress
        const destTokenData = await getTokenBySymbol(symbolDest);
        destTokenData.address = destTokenData.extensions.bridgeInfo["137"].tokenAddress

        if (!srcTokenData || !destTokenData) return;

        // Determine decimals and formatted amount
        const srcTokenDecimals = srcTokenData.decimals;
        const destTokenDecimals = destTokenData.decimals;
        setSrcTokenImage(srcTokenData.logoURI);
        setDestTokenImage(destTokenData.logoURI);
        setDestMaxDecimals(destTokenDecimals);
        setSrcMaxDecimals(srcTokenDecimals);

        const srcAmountNumber = Number(srcAmount);
        const formattedSrcAmount = `${(srcAmountNumber * 10 ** srcTokenDecimals).toFixed(0)}`;

        // Fetch balances immediately
        const srcTokenDataBalance = await getBalance(config, {
            address: walletAddress,
            chainId: 137,
            token: symbolSrc !== "MATIC" ? srcTokenData.address : undefined,
        });
        const srcTokenBalance = Number(srcTokenDataBalance.value) / 10 ** srcTokenDecimals;

        setSrcBalanceValue(
            `${srcTokenBalance.toFixed(srcTokenDecimals)} ${srcTokenData.symbol}`,
        );
        setIsValid(srcAmountNumber <= srcTokenBalance);

        const destTokenDataBalance = await getBalance(config, {
            address: walletAddress,
            chainId: 137,
            token: symbolSrc !== "MATIC" ? destTokenData.address : undefined,
        });
        const destBalance =
            Number(destTokenDataBalance.value) / 10 ** destTokenDecimals;
        setDestBalanceValue(
            `${destBalance.toFixed(destTokenDecimals)} ${destTokenData.symbol}`,
        );

        setSecondsToNextFetch(defaultTimerValue)

        try {
            // ParaSwap
            const paraswapUrl = new URL("https://api.paraswap.io/swap");
            paraswapUrl.searchParams.append("srcToken", srcTokenData.address);
            paraswapUrl.searchParams.append(
                "srcDecimals",
                srcTokenDecimals.toString(),
            );
            paraswapUrl.searchParams.append("destToken", destTokenData.address);
            paraswapUrl.searchParams.append(
                "destDecimals",
                destTokenDecimals.toString(),
            );
            paraswapUrl.searchParams.append("amount", formattedSrcAmount);
            paraswapUrl.searchParams.append("userAddress", walletAddress);
            paraswapUrl.searchParams.append("slippage", "50");
            paraswapUrl.searchParams.append("network", "137");
            paraswapUrl.searchParams.append("side", "SELL");

            const paraResp = await axios.get(paraswapUrl.toString(), { signal });
            const pData = paraResp.data as ParaSwapData;
            const paraswapFormattedAmount =
                Number(pData.priceRoute.destAmount) / 10 ** destTokenDecimals;
            setParaSwapData(pData);
            setSrcUSD(`${Number(pData.priceRoute.srcUSD).toFixed(2)} USD`);
            setDestUSD(`${Number(pData.priceRoute.destUSD).toFixed(2)} USD`);
            setGasCost(BigInt(pData.txParams.gasPrice))
            setBtcPriceParaSwap(paraswapFormattedAmount);

            // KyberSwap
            const routeUrl = new URL(
                "https://aggregator-api.kyberswap.com/polygon/api/v1/routes",
            );
            routeUrl.searchParams.append("tokenIn", srcTokenData.address);
            routeUrl.searchParams.append("tokenOut", destTokenData.address);
            routeUrl.searchParams.append("amountIn", formattedSrcAmount);
            routeUrl.searchParams.append("gasInclude", "false");

            const { data: { data: routeSummary } } = await axios.get(
                routeUrl.toString(),
                { signal }
            );

            const kyberPayload = {
                ...routeSummary,
                slippageTolerance: 50,
                sender: walletAddress,
                recipient: walletAddress,
                source: "ArchieMarqx",
                skipSimulateTx: false,
                enableGasEstimation: false,
                referral: "",
            };

            const { data: { data: kyberData } } = await axios.post(
                "https://aggregator-api.kyberswap.com/polygon/api/v1/route/build",
                kyberPayload,
                { signal },
            );

            const formattedOut =
                Number(kyberData.amountOut) / 10 ** destTokenDecimals;
            setKyberSwapData(kyberData);
            setBtcPriceKyberSwap(formattedOut);
        } catch (err) {
            if (err.name !== "CanceledError" || err.code !== "ERR_CANCELED") {
                console.error("Error fetching swap data:", err);
            }
        }
    }, [srcToken, destToken, srcAmount, walletAddress]);

    const fetchSwapDataBuy = useCallback(async () => {
        // cancel any in‑flight request
        abortControllerRef.current?.abort();
        const controller = new AbortController();
        abortControllerRef.current = controller;
        const { signal } = controller;

        if (!srcToken || !destToken || !destAmount || !srcAmount) return;

        const symbolSrc = srcToken === "POL" ? "MATIC" : srcToken;
        const symbolDest = destToken === "POL" ? "MATIC" : destToken;

        const srcTokenData = await getTokenBySymbol(symbolSrc);
        const destTokenData = await getTokenBySymbol(symbolDest);
        if (!srcTokenData || !destTokenData) return;

        // Determine decimals and formatted amount
        const srcTokenDecimals = srcTokenData.decimals;
        const destTokenDecimals = destTokenData.decimals;
        setSrcTokenImage(srcTokenData.logoURI);
        setDestTokenImage(destTokenData.logoURI);
        setDestMaxDecimals(destTokenDecimals);
        setSrcMaxDecimals(srcTokenDecimals);

        const srcAmountNumber = Number(srcAmount);
        const destAmountNumber = Number(destAmount);

        const formattedDestAmount = `${(destAmountNumber * 10 ** destTokenDecimals).toFixed(0)}`;

        // Fetch balances immediately
        const srcTokenDataBalance = await getBalance(config, {
            address: walletAddress,
            chainId: 137,
            token: symbolSrc !== "MATIC" ? srcTokenData.address : undefined,
        });
        const srcTokenBalance = Number(srcTokenDataBalance.value) / 10 ** srcTokenDecimals;

        setSrcBalanceValue(
            `${srcTokenBalance.toFixed(srcTokenDecimals)} ${srcTokenData.symbol}`,
        );
        setIsValid(srcAmountNumber <= srcTokenBalance);

        const destTokenDataBalance = await getBalance(config, {
            address: walletAddress,
            chainId: 137,
            token: symbolSrc !== "MATIC" ? destTokenData.address : undefined,
        });
        const destBalance = Number(destTokenDataBalance.value) / 10 ** destTokenDecimals
        setDestBalanceValue(`${destBalance.toFixed(srcTokenDecimals)} ${destTokenData.symbol}`)
        setSecondsToNextFetch(defaultTimerValue);

        try {
            // ParaSwap
            const paraswapUrl = new URL("https://api.paraswap.io/swap");
            paraswapUrl.searchParams.append("srcToken", destTokenData.address);
            paraswapUrl.searchParams.append("srcDecimals", destTokenDecimals.toString());

            paraswapUrl.searchParams.append("destToken", srcTokenData.address);
            paraswapUrl.searchParams.append("destDecimals", srcTokenDecimals.toString());

            paraswapUrl.searchParams.append("amount", formattedDestAmount);
            paraswapUrl.searchParams.append("userAddress", walletAddress);

            paraswapUrl.searchParams.append("slippage", "50");
            paraswapUrl.searchParams.append("network", "137");
            paraswapUrl.searchParams.append("side", "SELL");

            console.log("Paraswap URL:", paraswapUrl.toString());
            const paraResp = await axios.get(paraswapUrl.toString(), { signal });
            const pData = paraResp.data as ParaSwapData;
            const paraswapFormattedAmount = Number(pData.priceRoute.destAmount) / 10 ** srcTokenDecimals;

            setParaSwapData(pData);

            setSrcUSD(`${Number(pData.priceRoute.srcUSD).toFixed(2)} USD`);
            setDestUSD(`${Number(pData.priceRoute.destUSD).toFixed(2)} USD`);
            setBtcPriceParaSwap(paraswapFormattedAmount);

            // KyberSwap
            const routeUrl = new URL("https://aggregator-api.kyberswap.com/polygon/api/v1/routes");
            console.log("KyberSwap URL:", routeUrl);
            routeUrl.searchParams.append("tokenIn", destTokenData.address);
            routeUrl.searchParams.append("tokenOut", srcTokenData.address);
            routeUrl.searchParams.append("amountIn", formattedDestAmount);
            routeUrl.searchParams.append("gasInclude", "false");

            const { data: { data: routeSummary } } = await axios.get(
                routeUrl.toString(),
                { signal }
            );

            const kyberPayload = {
                ...routeSummary,
                slippageTolerance: 50,
                sender: walletAddress,
                recipient: walletAddress,
                source: "ArchieMarqx",
                skipSimulateTx: false,
                enableGasEstimation: false,
                referral: "",
            };

            const { data: { data: kyberData } } = await axios.post(
                "https://aggregator-api.kyberswap.com/polygon/api/v1/route/build",
                kyberPayload,
                { signal },
            );

            const formattedOut = Number(kyberData.amountOut) / 10 ** srcTokenDecimals;
            setKyberSwapData(kyberData);
            setBtcPriceKyberSwap(formattedOut);
        } catch (err) {
            if (err.name !== "CanceledError" || err.code !== "ERR_CANCELED") {
                console.error("Error fetching swap data:", err);
            }
        }
    }, [srcToken, destToken, destAmount, srcAmount, walletAddress]);

    const handleProportionSelection = (value: "33%" | "50%" | "max") => {
        if (srcBalanceValue === undefined) return;
        const balanceValue = Number(srcBalanceValue.split(" ")[0]);
        let amount = "0";

        switch (value) {
            case "max":
                amount = String(balanceValue);
                setSelectedPortion("max");
                setAmount(amount);
                setLastInputChange("src");
                break;
            case "50%":
                setSelectedPortion("50%");
                amount = ((balanceValue * 1) / 2).toFixed(srcMaxDecimals);
                setAmount(amount);
                setLastInputChange("src");
                break;
            case "33%":
                setSelectedPortion("33%");
                amount = ((balanceValue * 1) / 3).toFixed(srcMaxDecimals);
                setAmount(amount);
                setLastInputChange("src");
                break;
            default:
                setSelectedPortion("");
                setAmount(srcAmount);
                setLastInputChange("src");
                break;
        }
    };


    useEffect(() => {
        if (!srcAmount || !destAmount || !srcToken || !destToken || !srcMaxDecimals || !destMaxDecimals) return;
        const rate = (Number(srcAmount) / Number(destAmount)).toFixed(srcMaxDecimals)
        const inverseRate = (Number(destAmount) / Number(srcAmount)).toFixed(destMaxDecimals)
        setDestRate(`1 ${destToken} = ${rate} ${srcToken}`);
        setSrcRate(`1 ${srcToken} = ${inverseRate} ${destToken}`);
    }, [srcAmount, destAmount, srcToken, destToken, srcMaxDecimals, destMaxDecimals]);

    useEffect(() => {
        const balance = Number(srcBalanceValue.split(" ")[0])
        const amount = Number(srcAmount)

        if (amount && balance) {
            const one_third = Number((balance * 1 / 3).toFixed(srcMaxDecimals))
            const one_half = Number((balance * 1 / 2).toFixed(srcMaxDecimals))
            const max = Number((balance * 1).toFixed(srcMaxDecimals))

            const portions: Record<"33%" | "50%" | "max", number> = {
                "33%": one_third,
                "50%": one_half,
                max: max
            };

            const matchedPortion =
                (Object.keys(portions) as (keyof typeof portions)[])
                    .find((p) => portions[p] === amount) ?? "";

            if (matchedPortion !== selectedPortion) {
                setSelectedPortion(matchedPortion);
            }
        }
    }, [srcAmount, srcBalanceValue, selectedPortion, srcMaxDecimals]);

    useEffect(() => {
        const timer = setInterval(() => {
            if (secondsToNextFetch === null) return;
            if (!lastInputChange) return;
            setSecondsToNextFetch((secondsToNextFetch) => {
                if (secondsToNextFetch === null) return null;
                if (secondsToNextFetch <= 1) {
                    if (lastInputChange === "src") {
                        fetchSwapDataSell();
                    } else {
                        fetchSwapDataBuy();
                    }
                    return defaultTimerValue;
                }
                return secondsToNextFetch - 1;
            });
        }, 1000);
        return () => clearInterval(timer);
    }, [fetchSwapDataSell, fetchSwapDataBuy, lastInputChange]);

    useEffect(() => {
        if (!walletAddress || !srcToken || !destToken) return;

        if (lastInputChange === "src") {
            fetchSwapDataSell();
        } else {
            fetchSwapDataBuy();
        }
    }, [walletAddress, srcToken, destToken, lastInputChange, fetchSwapDataSell, fetchSwapDataBuy]);

    useEffect(() => {
        if (btcPriceKyberSwap || btcPriceParaSwap) {
            const kyberPrice = btcPriceKyberSwap ?? 0;
            const veloraPrice = btcPriceParaSwap ?? 0;
            const bestPriceValue = Math.max(kyberPrice, veloraPrice);

            setBestPriceText((bestPriceValue === kyberPrice) ? "KyberSwap" : "Velora")

            const upChange = "text-emerald-300"
            const downChange = "text-red-400"

            if (lastInputChange === "src") {
                const lastPrice = destAmount != null ? Number(destAmount) : null;
                setDestAmount(bestPriceValue.toString())
                if (lastPrice !== null) {
                    setTextColor(bestPriceValue > lastPrice ? upChange : downChange)
                }

            } else {
                const lastPrice = srcAmount != null ? Number(srcAmount) : null;
                setAmount(bestPriceValue.toString());
                if (lastPrice !== null) {
                    setTextColor(bestPriceValue > lastPrice ? upChange : downChange);
                }
            }
        }
    }, [btcPriceKyberSwap, btcPriceParaSwap, lastInputChange]);

    const paraswapSwap = async () => {
        if (paraSwapData) {
            sendTransaction(config, {
                to: paraSwapData.txParams.to,
                data: paraSwapData.txParams.data,
                maxFeePerGas: gasCost,
                maxPriorityFeePerGas: gasCost,
                value: BigInt(paraSwapData.txParams.value),
            });
        }
    };

    const kyberswapSwap = () => {
        if (kyberSwapData) {
            sendTransaction(config, {
                to: kyberSwapData.routerAddress,
                data: kyberSwapData.data,
                value: BigInt(kyberSwapData.transactionValue),
                maxFeePerGas: gasCost,
                maxPriorityFeePerGas: gasCost,
            });
        }
    };

    const handleSwap = () => {
        if (!swapData || !gasCost) return;
        sendTransaction(config, {
            to: swapData.to,
            data: swapData.data,
            value: BigInt(swapData.value),
            maxFeePerGas: gasCost,
            maxPriorityFeePerGas: gasCost,
        });
    }

    const gasCostCallback = useCallback(async () => {
        if (!swapData || !gasCost) return;

        const gasPriceValue = await estimateGas(config, {
            to: swapData.to,
            data: swapData.data,
            value: BigInt(swapData.value),
            maxFeePerGas: gasCost,
            maxPriorityFeePerGas: gasCost,
        });

        const calculatedGasCost = (Number(gasPriceValue * gasCost) / 10 ** 18).toFixed(4);
        console.log("Calculated Gas Cost:", calculatedGasCost);
        setCalculatedGasCost(calculatedGasCost);
    }, [swapData, gasCost]);

    useEffect(() => {
        if (swapData && gasCost) {
            gasCostCallback();
        }
    }, [swapData, gasCost, gasCostCallback]);

    useEffect(() => {
        if (textColor !== "text-white") {
            setTimeout(() => {
                setTextColor("text-white")
            }, 800)
        }
    }, [textColor])

    const handleSwapData = () => {
        if (!kyberSwapData || !paraSwapData) return;

        if (btcPriceKyberSwap && btcPriceParaSwap) {
            if (btcPriceKyberSwap > btcPriceParaSwap) {
                const swapData = {
                    to: kyberSwapData.routerAddress,
                    data: kyberSwapData.data,
                    value: BigInt(kyberSwapData.transactionValue),
                    maxFeePerGas: gasCost!,
                    maxPriorityFeePerGas: gasCost!,
                }
                setSwapData(swapData);
            } else {
                const swapData = {
                    to: paraSwapData.txParams.to,
                    data: paraSwapData.txParams.data,
                    value: BigInt(paraSwapData.txParams.value),
                    maxFeePerGas: gasCost!,
                    maxPriorityFeePerGas: gasCost!,
                }
                setSwapData(swapData);
            }
        }
    }

    function limitDecimalPlaces(e, count) {
        if (e.target.value.indexOf(".") == -1) {
            return;
        }
        if (e.target.value.length - e.target.value.indexOf(".") > count) {
            e.target.value = parseFloat(e.target.value).toFixed(count);
        }
    }

    const swapAlertDialog = () => {
        return (
            <Dialog>
                <DialogTrigger asChild>
                    <Button
                        className="bg-main-color w-full rounded-3xl h-full"
                        onClick={handleSwapData}
                        // disabled={!isValid || loading || !bestPriceText}
                        disabled={!bestPriceText}
                    >
                        <span className="text-large-size text-[#000000a6] font-semibold">Review with {bestPriceText}</span>
                    </Button>
                </DialogTrigger>
                <DialogContent className="w-[388px] h-120 p-4 bg-zinc-900/65 border border-zinc-800 border-solid backgroud-blur">
                    <DialogHeader>
                        <div className="gap-2 flex flex-col p-none">
                            <DialogTitle className="text-tertiary-text-color text-large-size font-semibold">{`You're swapping`}</DialogTitle>
                            <div className="grid gap-2">
                                <div className="flex justify-between items-center">
                                    <div className="flex flex-col gap-2">
                                        <div className="text-primary-text-color text-large-size font-semibold"> {srcAmount} {srcToken} </div>
                                        <div className="text-tertiary-text-color text-large-size font-semibold"> {srcUSD} </div>
                                    </div>
                                    <Image
                                        src={srcTokenImage ?? ""}
                                        alt="Token Logo"
                                        width={24}
                                        height={24}
                                        className="mr-[0.5rem] rounded-full w-13 h-13"
                                    />
                                </div>
                                <Image
                                    src={"/icons/bottom-arrow.svg"}
                                    alt="Ícone"
                                    height={14}
                                    width={14}
                                    className="h-6 w-6 opacity-45"
                                />
                                <div className="flex justify-between items-center">
                                    <div className="flex flex-col gap-2">
                                        <div className="text-primary-text-color text-large-size font-semibold">{destAmount} {destToken}</div>
                                        <div className="text-tertiary-text-color text-large-size font-semibold">{destUSD}</div>
                                    </div>
                                    <Image
                                        src={destTokenImage ?? ""}
                                        alt="Token Logo"
                                        width={24}
                                        height={24}
                                        className="mr-[0.5rem] rounded-full w-13 h-13"
                                    />
                                </div>
                            </div>
                        </div>
                    </DialogHeader>
                    <hr />
                    <div>
                        <div className="text-primary-text-color">
                            <span className="text-secondary-size text-secondary-text-color font-semibold">Network Fee POL:</span> {calculatedGasCost} POL
                        </div>
                        <div className="text-primary-text-color">
                            <span className="text-secondary-size text-secondary-text-color font-semibold">Network Fee USD:</span> {(() => {
                                const value = bestPriceText === "KyberSwap"
                                    ? Number(kyberSwapData?.gasUsd)
                                    : Number(paraSwapData?.priceRoute.gasCostUSD);
                                // toFixed(4) then trim any trailing zeros and optional decimal point
                                return parseFloat(value.toFixed(4)).toString();
                            })()} USD
                        </div>
                        <button className="text-primary-text-color" onClick={() => setIsSrcRate(!isSrcRate)}>
                            <span className="text-secondary-size text-secondary-text-color font-semibold">
                                Rate: {" "}
                            </span>
                            <span>{rateText()}</span>
                        </button>
                    </div>
                    <hr />
                    <div className="flex flex-col justify-end">
                        <Button className="w-full h-13" onClick={handleSwap}><span className="text-large-size text-[#000000a6] font-semibold">Swap</span></Button>
                    </div>
                </DialogContent>
            </Dialog>
        );
    }

    const srcTokenButton = () => {
        return (
            <div>
                <Dialog>
                    <DialogTrigger asChild>
                        <Button
                            variant="outline"
                            className="bg-card-color border border-solid border-zinc-500 w-full"
                        >
                            <Image
                                src={srcTokenImage ?? ""}
                                alt="Token Logo"
                                width={24}
                                height={24}
                                className="mr-[0.5rem] rounded-full w-6 h-6"
                            />
                            {srcToken}
                            <Image
                                src={"/icons/open-swap-modal.svg"}
                                alt="Ícone"
                                width={24}
                                height={24}
                                className="opacity-45 w-6 h-6"
                            />
                        </Button>
                    </DialogTrigger>
                    <DialogContent className="w-[26.875rem]">
                        <DialogHeader>
                            <DialogTitle>Select a token</DialogTitle>
                        </DialogHeader>
                        <Command className="w-full bg-none">
                            <CommandInput placeholder="Search token..." />
                            <CommandList>
                                <CommandGroup heading="Tokens">
                                    {tokenList?.map((token) => {
                                        if (token.extensions?.bridgeInfo['137']?.tokenAddress == undefined) return null;
                                        return (
                                            <DialogClose key={token.symbol}>
                                                <CommandItem
                                                    onSelect={() => {
                                                        setSrcToken(token.symbol);
                                                        setSrcTokenImage(token.logoURI);
                                                        setSrcMaxDecimals(token.decimals);
                                                        setSrcBalanceValue("");
                                                        setAmount("");
                                                    }}
                                                    className="flex flex-row"
                                                >
                                                    <Image
                                                        src={token.logoURI}
                                                        alt="Token Logo"
                                                        width={40}
                                                        height={40}
                                                        className="mr-[0.5rem] rounded-full w-10 h-10"
                                                    />
                                                    <div className="flex flex-col">
                                                        <span className="text-large-size text-primary-text-color text-left">{token.symbol}</span>
                                                        <span className="text-tertiary-size text-tertiary-text-color text-left">{token.extensions.bridgeInfo['137'].tokenAddress}</span>
                                                    </div>
                                                </CommandItem>
                                            </DialogClose>
                                        );
                                    })}
                                </CommandGroup>
                                <CommandEmpty className="text-white/65 text-center">No results found.</CommandEmpty>
                            </CommandList>
                        </Command>
                    </DialogContent>
                </Dialog>
            </div>
        );
    };

    const rateText = () => {
        if (!srcRate || !destRate) return undefined;
        return isSrcRate ? srcRate : destRate
    }

    const destTokenButton = () => {
        return (
            <div>
                <Dialog>
                    <DialogTrigger asChild>
                        <Button
                            variant="outline"
                            className="bg-card-color border border-solid border-zinc-500 w-full"
                        >
                            <Image
                                src={destTokenImage ?? ""}
                                alt="Token Logo"
                                width={24}
                                height={24}
                                className="mr-[0.5rem] rounded-full w-6 h-6"
                            />
                            {destToken}
                            <Image
                                src={"/icons/open-swap-modal.svg"}
                                alt="Ícone"
                                width={24}
                                height={24}
                                className="opacity-45 w-6 h-6"
                            />
                        </Button>
                    </DialogTrigger>
                    <DialogContent className="sm:max-w-[425px] m-0 p-4">
                        <DialogHeader>
                            <DialogTitle>Select a token</DialogTitle>
                        </DialogHeader>
                        <Command>
                            <CommandInput placeholder="Type a command or search..." />
                            <CommandList>
                                <CommandGroup heading="Tokens">
                                    {tokenList?.map((token) => {
                                        return (
                                            <DialogClose key={token.symbol}>
                                                <CommandItem
                                                    onSelect={() => {
                                                        console.log("Selected token:", token);
                                                        setDestToken(token.symbol);
                                                        setDestTokenImage(token.logoURI);
                                                        setDestMaxDecimals(token.decimals);
                                                        setDestBalanceValue("");
                                                        setDestAmount(undefined);
                                                    }}
                                                    className="w-full"
                                                >
                                                    <Image
                                                        src={token.logoURI}
                                                        alt="Token Logo"
                                                        width={40}
                                                        height={40}
                                                        className="mr-[0.5rem] rounded-full w-10 h-10"
                                                    />
                                                    <div className="flex flex-col">
                                                        <span className="text-large-size text-primary-text-color text-left">{token.symbol}</span>
                                                        <span className="text-tertiary-size text-tertiary-text-color text-left">{token.address}</span>
                                                    </div>
                                                </CommandItem>
                                            </DialogClose>
                                        );
                                    })}
                                </CommandGroup>
                                <CommandEmpty className="text-white/65 text-center">No results found.</CommandEmpty>
                            </CommandList>
                        </Command>
                    </DialogContent>
                </Dialog>
            </div>
        );
    };

    const staticTextInputClass = `min-w-0 outline-none text-huge-size font-semibold h-11 no-spinner ${isValid ? "text-white/65" : "text-[#FF593C]"}`
    const dynamicTextInputClass = `min-w-0 outline-none text-huge-size font-semibold h-11 no-spinner transition-colors duration-200 ${textColor}`

    const swapTokensButton = (
        <button
            className="group absolute top-1/2 left-1/2 transform -translate-x-1/2 -translate-y-1/2 bg-zinc-800 hover:bg-neutral-700 rounded-lg border-3 border-solid border-zinc-900 w-10 h-10 flex items-center justify-center cursor-pointer"
            onClick={() => {
                setSrcToken(destToken);
                setDestToken(srcToken);
                if (lastInputChange === "src") {
                    setDestAmount("");
                    setAmount(destAmount ?? "");
                } else {
                    setAmount(destAmount ?? "");
                    setDestAmount("");
                }

            }}
        >
            <Image
                src="/icons/token-inverse.svg"
                alt="invert tokens"
                width={24}
                height={24}
                className="h-6 w-6 group-hover:transform group-hover:transition-transform group-hover:duration-300 group-hover:rotate-180"
            />
        </button>
    );

    const srcTokenContainer = (
        <div
            className="bg-zinc-800 px-2 h-27 rounded-lg flex justify-between py-1 focus-within:bg-neutral-700/65 hover:bg-neutral-700/65"
            itemProp="srcToken"
        >
            <div className="flex flex-col w-[13.75rem] justify-around">
                <span className="text-primary-size font-semibold text-tertiary-text-color">
                    Sell
                </span>
                <div className="grid grid-cols-1 gap-[0.5rem] items-center">
                    <input
                        className={lastInputChange === "src" ? staticTextInputClass : dynamicTextInputClass}
                        placeholder="Amount"
                        ref={srcInputRef}
                        onChange={(e) => {
                            const match = /^[+]?([0-9]+(?:[.][0-9]*)?|\.[0-9]+)$/.exec(e.target.value);
                            if (!match) {
                                e.target.value = e.target.value.slice(0, -1);
                            }
                            limitDecimalPlaces(e, srcMaxDecimals);
                            setLastInputChange("src");
                            if (inputTimeoutRef.current)
                                clearTimeout(inputTimeoutRef.current);
                            inputTimeoutRef.current = setTimeout(() => {
                                if (e.target.value !== "") {
                                    setAmount(e.target.value);
                                }
                            }, 1000);
                        }}
                    />
                </div>
                <span className="text-tertiary-size font-semibold text-tertiary-text-color">
                    {srcUSD}
                </span>
            </div>
            <div className="flex flex-col w-[131px] justify-around">
                <ToggleGroup
                    type="single"
                    onValueChange={handleProportionSelection}
                    value={selectedPortion}
                    className="flex justify-between"
                >
                    <ToggleGroupItem
                        value="33%"
                        className="bg-transparent border border-solid border-main-color h-5.5 w-9 flex items-center justify-center rounded-lg"
                    >
                        <span className="text-secondary-size font-semibold">33%</span>
                    </ToggleGroupItem>
                    <ToggleGroupItem
                        value="50%"
                        className="bg-transparent border border-solid border-main-color h-5.5 w-9 flex items-center justify-center rounded-lg"
                    >
                        <span className="text-secondary-size font-semibold">50%</span>
                    </ToggleGroupItem>
                    <ToggleGroupItem
                        value="max"
                        className="bg-transparent border border-solid border-main-color h-5.5 w-9 flex items-center justify-center rounded-lg"
                    >
                        <span className="text-secondary-size font-semibold">Max</span>
                    </ToggleGroupItem>
                </ToggleGroup>
                {srcTokenImage ? srcTokenButton() : null}
                <span className={`text-tertiary-size font-semibold flex justify-center w-full ${isValid ? "text-white/65" : "text-[#FF593C]"}`}>
                    {srcMaxDecimals <= 8 ? srcBalanceValue : Number(srcBalanceValue.split(" ")[0]).toFixed(8) + " " + srcToken}
                </span>
            </div>
        </div>
    );

    const destTokenContainer = (
        <div
            className="bg-zinc-800 px-2 h-27 rounded-lg flex justify-between py-1 focus-within:bg-neutral-700/65 hover:bg-neutral-700/65"
            itemProp="destToken"
        >
            <div className="flex flex-col w-[13.75rem] justify-around">
                <span className="text-primary-size font-semibold text-tertiary-text-color">
                    Receive
                </span>
                <div className="grid grid-cols-1 gap-[0.5rem] items-center">
                    <input
                        className={lastInputChange === "dest" ? staticTextInputClass : dynamicTextInputClass}
                        placeholder="Amount"
                        ref={destInputRef}
                        onChange={(e) => {
                            const match = /^[+]?([0-9]+(?:[.][0-9]*)?|\.[0-9]+)$/.exec(e.target.value);
                            if (!match) {
                                e.target.value = e.target.value.slice(0, -1);
                            }
                            limitDecimalPlaces(e, destMaxDecimals);
                            setLastInputChange("dest");
                            if (inputTimeoutRef.current)
                                clearTimeout(inputTimeoutRef.current);
                            inputTimeoutRef.current = setTimeout(() => {
                                if (e.target.value !== "") {
                                    setDestAmount(e.target.value);
                                }
                            }, 250);
                        }}
                    />
                </div>
                <span className="text-tertiary-size font-semibold text-tertiary-text-color">
                    {destUSD}
                </span>
            </div>
            <div className="flex flex-col w-[131px] justify-around">
                <div className="flex justify-between h-5.5"></div>
                {destTokenImage ? destTokenButton() : null}
                <span className="text-tertiary-size font-semibold text-tertiary-text-color flex justify-center w-full">
                    {destMaxDecimals <= 8 ? destBalanceValue : Number(destBalanceValue.split(" ")[0]).toFixed(8) + " " + destToken}
                </span>
            </div>
        </div>
    )

    const swapButtons = (
        <div className="grid grid-cols-1 grid-rows-2 w-full gap-4">
            {swapAlertDialog()}
            <div className="grid grid-cols-2 gap-2">
                <Button
                    variant="outline"
                    className="rounded-none rounded-l-2xl bg-transparent hover:bg-[#f75c2a27] border-l border-solid border-[#f75c2a] w-full"
                    onClick={paraswapSwap}
                    // disabled={!isValid || loading}
                    disabled={!paraSwapData}
                >
                    Swap <span className="text-[#f75c2a] font-semibold">Velora</span>
                </Button>
                <Button
                    variant="outline"
                    className="rounded-none rounded-r-2xl bg-transparent hover:bg-[#31cb9e27] border-solid border-[#31cb9e] w-full"
                    onClick={kyberswapSwap}
                    // disabled={!isValid || loading}
                    disabled={!kyberSwapData}
                >
                    Swap with <span className="text-[#31cb9e] font-semibold">Kyber</span>
                </Button>
            </div>
        </div>
    );

    return (
        <Card className="bg-card-color text-text-color rounded-[1rem]">
            <CardHeader>
                <CardTitle className="flex justify-between items-end">
                    <span className="text-big-size font-semibold">Swap</span>
                    <Image
                        src={"/icons/gear.svg"}
                        alt="Ícone"
                        width={32}
                        height={32}
                        className="opacity-45 w-8 h-8 cursor-pointer"
                        onClick={() => {
                            console.log("Settings clicked");
                        }}
                    />
                </CardTitle>
            </CardHeader>
            <CardContent>
                <div className="grid grid-rows-2 gap-2 relative">
                    {srcTokenContainer}
                    {swapTokensButton}
                    {destTokenContainer}
                </div>
            </CardContent>
            <CardFooter>
                {!walletAccount  ? ConnectWalletButton : swapButtons}
            </CardFooter>
        </Card>
    );
}
