import os
import json
from ast import literal_eval

import ccxt
import pandas as pd
import pytz
from datetime import datetime
import klib
from crypto_explorer import CcxtAPI, AccountAPI, MoralisAPI
from flask import Flask, request, jsonify
from flask_cors import CORS
from machine_learning.model_builder import model_creation
from machine_learning.ml_utils import DataHandler, get_recommendation


app = Flask(__name__)

@app.route("/update_price_data", methods=["GET"])
def update_BTC_price_data():
    try:
        old_data = pd.read_parquet("data/api_data/cryptodata/dynamic_btc.parquet")
        last_time = pd.to_datetime(old_data.index[-3]).timestamp() * 1000

        ccxt_api = CcxtAPI(
            "BTC/USDT", "1d", ccxt.binance(), since=int(last_time), verbose="Text"
        )

        new_data = ccxt_api.get_all_klines().to_OHLCV().data_frame

        btc = new_data.combine_first(old_data).drop_duplicates()
        btc.to_parquet("data/api_data/cryptodata/dynamic_btc.parquet")
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
    price_data = pd.read_parquet(
        "data/api_data/cryptodata/dynamic_btc.parquet"
    )
    model_df = calculate_targets(price_data)

    results = pd.read_parquet(
        r"data\models\05-08-2024\btc_best_results_1_max_trade_05082024.parquet"
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
    expected_results = results["33139"]["Liquid_Result"].loc[:"28-07-2024"]

    pd.testing.assert_series_equal(test_model, expected_results)

    result_models_dfs = create_model(configs, 33139, model_df)[33139]
    result_models_dfs["BTC"] = model_df["close"].pct_change() + 1

    recommendation_ml = result_models_dfs[
        ["y_pred_probs", "Predict", "Position"]
    ]

    new_recommendations = get_recommendation(
        recommendation_ml["Position"].loc["15-09-2024":],
        add_span_tag=True,
    ).rename(f"model_{33139}")

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
    api_key = request.form["api_key"]
    wallet = request.form["wallet"]

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
    response.headers.add("Content-Type", "multipart/form-data; boundary=---011000010111000001101001")

    return response

def get_account_balance(wallet, api_key):
    moralis_API = MoralisAPI(verbose=False, api_key=api_key)

    return moralis_API.get_wallet_token_balances_history(
        wallet,
        "0x1BFD67037B42Cf73acF2047067bd4F2C47D9BfD6",
    )

def wallet_data(wallet_df, asset_column: str, usdt_column: str):
    wallet_df[asset_column] = wallet_df[asset_column].fillna(0)
    wallet_df[usdt_column] = wallet_df[usdt_column].fillna(0)

    wallet_df["asset_usd"] = wallet_df["usdPrice"] * wallet_df[asset_column]
    wallet_df["total_usd"] = wallet_df[usdt_column] + wallet_df["asset_usd"]

    wallet_df["asset_ratio"] = (
        wallet_df["asset_usd"] / wallet_df["total_usd"]
    ).round(2) * 100

    wallet_df[["usdPrice", "asset_usd", "total_usd"]] = wallet_df[
        ["usdPrice", "asset_usd", "total_usd"]
    ].round(2)

    return wallet_df.sort_index()

@app.route("/account/history", methods=["POST"])
def get_account_balance_history():
    api_key = request.form["api_key"]
    wallet = request.form["wallet"]

    wallet_balance = get_account_balance(wallet, api_key)
    wallet_df = wallet_data(wallet_balance, "WBTC", "USDT").drop(
        columns=["LGNS", "WBTC", "usdPrice"]
    )

    return jsonify(wallet_df.to_dict())

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=2000, debug=True)
