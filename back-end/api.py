import os
import json
from ast import literal_eval

import ccxt
import pandas as pd
import numpy as np
import pytz
from datetime import datetime
import klib
from crypto_explorer import CcxtAPI, AccountAPI, MoralisAPI

from flask import Flask, request, jsonify
from flask_cors import CORS

from machine_learning.model_builder import model_creation
from machine_learning.ml_utils import DataHandler, get_recommendation
from machine_learning.ml_pipeline import ModelPipeline

app = Flask(__name__)

CORS(app, resources={r"/*": {
    "origins": ["http://localhost:3000"],
    "methods": ["GET", "POST", "OPTIONS"],
    "allow_headers": ["Content-Type", "Authorization"]
}})


@app.after_request
def add_cors_headers(response):
    response.headers.add("Access-Control-Allow-Origin", "*")
    response.headers.add(
        "Access-Control-Allow-Headers", "Content-Type,Authorization"
    )
    response.headers.add("Access-Control-Allow-Methods", "GET,POST,OPTIONS")
    return response

@app.route("/update_price_data", methods=["GET"])
def update_BTC_price_data():
    database_url = os.getenv("DATABASE_URL")
    try:
        old_data = pd.read_sql("btc", con=database_url, index_col="date")
        last_time = pd.to_datetime(old_data.index[-3]).timestamp() * 1000

        ccxt_api = CcxtAPI(
            "BTC/USDT", "1d", ccxt.binance(), since=int(last_time), verbose="Text"
        )

        new_data = ccxt_api.get_all_klines().to_OHLCV().data_frame

        btc = new_data.combine_first(old_data).drop_duplicates()
        btc.to_sql("btc", con=database_url, if_exists="replace")

        return jsonify({"status": "success", "message": "Data updated successfully"})

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})

def calculate_targets(price_data):
    target_length = 7

    model_df = DataHandler(price_data).calculate_targets(target_length)
    model_df = klib.convert_datatypes(model_df)
    model_df["Target_bin"] = model_df["Target_bin"].replace({0: -1})
    return model_df

def create_model(configs_dataset, x, model_df):
    hyperparams = configs_dataset.loc[x, "hyperparameters"][0]
    hyperparams["iterations"] = 1000

    feat_params = literal_eval(configs_dataset.loc[x, "feat_parameters"])[0]
    test_index = int(configs_dataset.loc[x, "test_index"])
    train_in_mid = configs_dataset.loc[x, "train_in_middle"]
    side = int(configs_dataset.loc[x, "side"])
    max_trades = int(configs_dataset.loc[x, "max_trades"])
    off_days = int(configs_dataset.loc[x, "off_days"])

    mta, _, _, _ = model_creation(
        feat_params,
        hyperparams,
        test_index,
        model_df,
        dev=False,
        train_in_middle=train_in_mid,
        cutoff_point=5,
        side=side,
        max_trades=max_trades,
        off_days=off_days,
    )

    return {x: mta}

@app.route("/get_recommendations", methods=["GET"])
def get_recommendations():
    database_url = os.getenv("DATABASE_URL")
    price_data = pd.read_sql("btc", con=database_url, index_col="date")
    model_df = calculate_targets(price_data)

    results = pd.read_sql(
        "backtest_results_model", con=database_url, index_col="date"
    )

    hyperparams = literal_eval(json.loads(os.getenv("33139_hyperparams")))
    features = literal_eval(json.loads(os.getenv("33139_features")))
    model_configs = literal_eval(json.loads(os.getenv("33139_configs")))

    configs = pd.DataFrame(
        {
            "feat_parameters": [str(features)],
            "hyperparameters": [hyperparams],
            **model_configs,
        },
        index=pd.Index([33139], name="model_index"),
    )

    test_model = create_model(configs, 33139, model_df)[33139][
        "Liquid_Result"
    ].loc[:"28-07-2024"]
    expected_results = results["Liquid_Result"].loc[:"28-07-2024"]

    pd.testing.assert_series_equal(test_model, expected_results)

    result_models_dfs = create_model(configs, 33139, model_df)[33139]
    result_models_dfs["BTC"] = model_df["close"].pct_change() + 1

    recommendation_ml = result_models_dfs[
        ["y_pred_probs", "Predict", "Position"]
    ]

    new_recommendations = get_recommendation(
        recommendation_ml["Position"].loc["15-09-2024":],
        add_span_tag=True,
    ).rename(f"model_33139")

    recommendations = new_recommendations.copy()

    recommendations.index = (
        recommendations.index.tz_localize("UTC")
        + pd.Timedelta(hours=23, minutes=59, seconds=59)
    ).strftime("%Y-%m-%d %H:%M:%S")

    last_index = pd.Timestamp(datetime.now(pytz.timezone("UTC"))).strftime(
        "%Y-%m-%d %H:%M:%S"
    )

    recommendations.index = (
        pd.DatetimeIndex(recommendations.index[:-1].tolist() + [last_index])
        .tz_localize("UTC")
        .tz_convert("America/Sao_Paulo")
    )

    itables_recommendations = recommendations.copy()

    last_index_hour = itables_recommendations.index[-1].hour
    last_index_minute = itables_recommendations.index[-1].minute
    last_index_second = itables_recommendations.index[-1].second

    itables_recommendations.index = itables_recommendations.index.strftime(
        "%Y-%m-%d %H:%M:%S"
    )

    if (
        last_index_hour != 20
        and last_index_minute != 59
        and last_index_second != 59
    ):
        span_tag = "<span style='color: red'>"
        close_span_tag = "</span>"
        last_index = (
            span_tag + itables_recommendations.index[-1] + close_span_tag
        )
        itables_recommendations.index = itables_recommendations.index[
            :-1
        ].tolist() + [last_index]

    return jsonify(itables_recommendations.iloc[::-1].to_dict())


@app.route("/account/buys", methods=["POST"])
def get_buys():
    api_key = request.json["api_key"]
    wallet = request.json["wallet"]

    df = AccountAPI(api_key, False).get_buys(wallet, "WBTC").iloc[-3:]
    df["sell_when"] = ["67%", "33%", "0%"]
    df["buyed_when"] = ["33%", "67%", "100%"]
    df = df.rename(
        columns={
            "from": "cost",
            "to": "qty",
            "from_coin_name": "cost_token",
            "to_coin_name": "buyed_token",
        }
    )

    df["price"] = (df["cost"] / df["qty"]).round(2)

    columns = [
        "cost",
        "cost_token",
        "qty",
        "buyed_token",
        "price",
        "buyed_when",
        "sell_when",
    ]
    response = jsonify(df[columns].to_dict())
    response.headers.add("Content-Type", "application/json")

    return response


def get_account_balance(wallet: str, api_key: str, update: bool = False):
    database_url = os.getenv("DATABASE_URL")

    wallet_df = pd.read_sql(
        "wallet_balances",
        con=database_url,
        index_col="height",
    )

    if update:
        wallet_df = wallet_df.sort_index(ascending=False)

        moralis_api = MoralisAPI(
            verbose=True,
            api_key=api_key,
        )

        wallet_blocks = moralis_api.get_wallet_blocks(
            wallet_address=wallet,
            from_block=int(wallet_df.index[1]),
        )

        wallet_blocks = (
            pd.Series(wallet_blocks)
            .sort_values(ascending=True)
            .tolist()
        )

        wallet_df_adj = wallet_df.copy().iloc[2:]

        for block in wallet_blocks:
            moralis_api.logger.info(
                f"Getting token balances for block {block}."
            )

            temp_df = moralis_api.get_wallet_token_balances(wallet, block).T

            token_price = moralis_api.fetch_token_price(
                block,
                "0x1BFD67037B42Cf73acF2047067bd4F2C47D9BfD6",
            )

            temp_df["usdPrice"] = token_price["usdPrice"]
            temp_df["blockTimestamp"] = pd.Timestamp(
                int(token_price["blockTimestamp"]),
                unit="ms",
            )
            temp_df.index.name = "height"

            wallet_df_adj = pd.concat([temp_df, wallet_df_adj], axis=0)

            wallet_df_adj.to_sql(
                "wallet_balances",
                con=database_url,
                if_exists="replace",
                index=True,
                index_label="height",
            )

    return wallet_df


def wallet_data(
    wallet_df,
    asset_column: str,
    usdt_column: str | list | pd.Index | None  = None,
):
    if isinstance(usdt_column, (list, pd.Index, str)):
        raise ValueError(
            "usdt_column must be a list, pd.Index, or str."
        )

    if usdt_column is None:
        usdt_column = ["USDT", "USDC", "DAI", "TUSD", "BUSD"]

    usd_df = pd.DataFrame()
    for column in wallet_df.columns:
        if column in usdt_column:
            usd_df[column] = wallet_df[column].fillna(0)

    usd_df = usd_df.sum(axis=1)

    wallet_df["usd"] = usd_df

    wallet_df["asset_usd"] = wallet_df["usdPrice"] * wallet_df[asset_column]
    wallet_df["total_usd"] = wallet_df["usd"] + wallet_df["asset_usd"]

    wallet_df["asset_ratio"] = (
        wallet_df["asset_usd"] / wallet_df["total_usd"]
    ).round(2) * 100

    wallet_df[["usdPrice", "asset_usd", "total_usd"]] = wallet_df[
        ["usdPrice", "asset_usd", "total_usd"]
    ].round(2)

    return wallet_df.sort_index()


def process_balance(wallet_df: pd.DataFrame, indexes_to_drop: pd.Index): 
    result = (
        wallet_df.reset_index()["total_usd"]
        .pct_change()
        .drop(index=indexes_to_drop)
        + 1
    ).cumprod() - 1

    btc_result = (
        wallet_df.reset_index()["usdPrice"]
        .pct_change()
        .drop(index=indexes_to_drop)
        + 1
    ).cumprod() - 1

    results = pd.concat(
        [result.rename("Modelo"), btc_result.rename("BTC")], axis=1
    )

    results = results.fillna(0)
    results["highest_BTC"] = results["BTC"].cummax()
    results["highest_Modelo"] = results["Modelo"].cummax()
    results["data"] = wallet_df.reset_index()["blockTimestamp"]
    results = results.set_index("data")
    results["asset_ratio"] = wallet_df.reset_index().set_index(
        "blockTimestamp"
    )["asset_ratio"]
    results[results.columns[:-1]] = (
        results[results.columns[:-1]]
    )
    return results


@app.route("/account/history", methods=["POST"])
def get_account_balance_history():
    api_key = request.json["api_key"]
    wallet = request.json["wallet"]
    update = request.json.get("update", "false").lower() == "true"

    wallet_balance = get_account_balance(wallet, api_key, update)
    wallet_df = wallet_data(wallet_balance, "WBTC")
    wallet_df = process_balance(wallet_df, 33)
    wallet_df.index = wallet_df.index.strftime("%Y-%m-%d")

    if wallet_df.empty:
        return (
            jsonify({"error": "No data available for the given wallet."}),
            404,
        )

    response = jsonify(wallet_df.to_dict())
    response.headers.add("Content-Type", "application/json")

    return response

@app.route("/account/changes/month", methods=["POST"])
def get_account_balance_monthly():
    api_key = request.json["api_key"]
    wallet = request.json["wallet"]
    update = request.json.get("update", "false").lower() == "true"

    wallet_balance = get_account_balance(wallet, api_key, update)
    wallet_df = wallet_data(wallet_balance, "WBTC")
    wallet_df = process_balance(wallet_df, 33)
    wallet_df = (wallet_df.diff()).resample("M").sum() * 100
    wallet_df = wallet_df.iloc[:, :2].round(2)

    wallet_df.index = wallet_df.index.strftime("%Y-%m-%d")

    if wallet_df.empty:
        return (
            jsonify({"error": "No data available for the given wallet."}),
            404,
        )

    response = jsonify(wallet_df.to_dict())

    if update:
        database_url = os.getenv("DATABASE_URL")
        wallet_df.to_sql(
            "wallet_balance_monthly",
            con=database_url,
            if_exists="replace",
            index=True,
            index_label="date",
        )

    response.headers.add("Content-Type", "application/json")

    return response

@app.route("/btc/update", methods=["POST"])
def update_btc_data():
    database_url = os.getenv("DATABASE_URL")
    database_df = pd.read_sql("btc", con=database_url, index_col="date").iloc[:-2]
    database_df.to_sql(
        "btc",
        con=database_url,
        if_exists="replace",
        index=True,
        index_label="date",
    )

    old_database = pd.read_sql("btc", con=database_url, index_col="date")
    last_time = pd.to_datetime(old_database.index[-2]).timestamp() * 1000

    ccxt_api = CcxtAPI(
        "BTC/USDT",
        "1d",
        ccxt.binance(),
        since=int(last_time),
        verbose="Text",
    )

    new_data = ccxt_api.get_all_klines().to_OHLCV().data_frame
    btc = new_data.combine_first(old_database).drop_duplicates()

    btc.to_sql(
        "btc",
        con=database_url,
        if_exists="replace",
        index=True,
        index_label="date",
    )

    return jsonify(
        {"status": "success", "message": "Data updated successfully"}
    )


@app.route("/model/recommendations", methods=["GET"])
def get_model_recommendations():
    database_url = os.getenv("DATABASE_URL")

    model_pipeline = ModelPipeline("btc", database_url=database_url)
    recommendations = model_pipeline.get_model_recommendations()[::-1]
    clean_recomendations = (
        model_pipeline.get_model_recommendations(False)[::-1]
        .rename("Clean")
        .to_frame()
    )

    clean_recomendations_splitted = clean_recomendations["Clean"].str.split(" ", expand=True)
    clean_recomendation = clean_recomendations_splitted[0].rename("position").to_frame()
    clean_recomendation["side"] = (
        clean_recomendations_splitted[1].fillna("").astype(str)
        + " "
        + clean_recomendations_splitted[2].fillna("").astype(str).fillna("")
    )
    clean_recomendation["capital"] = clean_recomendations_splitted[4].fillna("")
    recommendations = pd.concat([recommendations, clean_recomendation], axis=1)
    recommendations = recommendations.fillna("")

    recommendations.to_sql(
        "model_recommendations",
        con=database_url,
        if_exists="replace",
        index=True,
        index_label="date",
    )

    return jsonify(
        {"status": "success", "message": "Data updated successfully"}
    )

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=2000, debug=True)
