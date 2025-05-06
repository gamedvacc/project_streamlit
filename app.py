import os
import streamlit as st
from PIL import Image
import openai
import base64
import requests
import pandas as pd
import ta
import random
import time


openai.api_key = st.secrets["openai"]["api_key"]

st.set_page_config(page_title="📊 Chart Analyzer", layout="centered")
st.title("📈 Chart Analyzer with Screenshot or Coin Data")

# --- Input Fields ---
coin_name = st.text_input("Coin Symbol (e.g., BTCUSDT)")
timeframe = st.selectbox("Timeframe", ["1m", "3m", "5m", "15m", "30m", "1h", "4h", "1d", "1w", "1M"], index=3)
uploaded_file = st.file_uploader("Chart Screenshot (optional)", type=["png", "jpg", "jpeg"])

if st.button("Analyze 📊"):
    if uploaded_file:
        st.subheader(f"Analyzing Screenshot for {coin_name or ''} on {timeframe}")
        uploaded_file.seek(0)
        img_bytes = uploaded_file.read()
        img_b64 = base64.b64encode(img_bytes).decode("utf-8")
        data_uri = f"data:image/png;base64,{img_b64}"

        prompt = (
            f"You are an expert crypto analyst. Analyze the screenshot chart of {coin_name or 'the coin'} on {timeframe}. "
            "Explain MA, EMA, BOLL, SAR, AVL, VOL, MACD, RSI, KDJ, OBV, WR, StochRSI indicators. "
            "End the explanation with '## Conclusion:' in Roman Urdu."
        )

        # Note: Currently OpenAI GPT-4 Vision API is required for image input. If you have access, use openai.ChatCompletion with image tool.
        response = openai.ChatCompletion.create(
            model="gpt-4-vision-preview",
            messages=[
                {"role": "system", "content": "You are an expert crypto analyst."},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": data_uri}},
                    ],
                },
            ],
            max_tokens=3000
        )

        analysis = response.choices[0].message.content
        st.subheader("Detailed Analysis & Reasoning:")
        st.text_area("", analysis, height=400)
        st.subheader("Main Conclusion:")
        st.write(analysis.split("## Conclusion:")[-1].strip() if "## Conclusion:" in analysis else "Conclusion missing.")

    elif coin_name and timeframe:
        st.subheader(f"Fetching Real-Time Data for {coin_name} on {timeframe}")
        try:
            url = f"https://fapi.binance.com/fapi/v1/klines?symbol={coin_name}&interval={timeframe}&limit=100&random={random.randint(1,100000)}"
            headers = {
                "User-Agent": "Mozilla/5.0",
                "Cache-Control": "no-cache",
                "Pragma": "no-cache"
            }

            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            data = response.json()

            price_url = f"https://fapi.binance.com/fapi/v1/ticker/price?symbol={coin_name}"
            price_resp = requests.get(price_url, timeout=5)
            current_price = float(price_resp.json().get("price", 0.0))
            if data:
                data[-1][4] = str(current_price)

            df = pd.DataFrame(data, columns=[
                "open_time", "open", "high", "low", "close", "volume", "close_time",
                "qav", "num_trades", "taker_base_vol", "taker_quote_vol", "ignore"
            ])
            df = df.astype({"open": float, "high": float, "low": float, "close": float, "volume": float})

            # Calculate indicators
            df["MA"] = df["close"].rolling(window=20).mean()
            df["EMA"] = ta.trend.EMAIndicator(df["close"], window=20).ema_indicator()
            bb = ta.volatility.BollingerBands(df["close"], window=20, window_dev=2)
            df["BOLL_upper"] = bb.bollinger_hband()
            df["BOLL_mid"] = bb.bollinger_mavg()
            df["BOLL_lower"] = bb.bollinger_lband()

            try:
                df["SAR"] = ta.trend.PSARIndicator(df["high"], df["low"], df["close"]).psar()
            except Exception as e:
                st.warning(f"SAR calculation failed: {e}")
                df["SAR"] = 0

            df["AVL"] = df["volume"].rolling(window=20).mean()
            df["VOL"] = df["volume"]
            macd = ta.trend.MACD(df["close"])
            df["MACD"] = macd.macd()
            df["RSI"] = ta.momentum.RSIIndicator(df["close"]).rsi()
            stoch = ta.momentum.StochasticOscillator(df["high"], df["low"], df["close"])
            df["K"] = stoch.stoch()
            df["D"] = stoch.stoch_signal()
            df["J"] = 3 * df["K"] - 2 * df["D"]
            df["OBV"] = ta.volume.OnBalanceVolumeIndicator(df["close"], df["volume"]).on_balance_volume()
            df["WR"] = ta.momentum.WilliamsRIndicator(df["high"], df["low"], df["close"]).williams_r()
            df["StochRSI"] = ta.momentum.StochRSIIndicator(df["close"]).stochrsi()

            latest = df.iloc[-1]
            prompt = f"""
You are an expert crypto analyst. Here's the latest indicator data for {coin_name} on {timeframe}:

Latest Price: {current_price}
MA: {latest['MA']}, EMA: {latest['EMA']}, BOLL(Upper/Mid/Lower): {latest['BOLL_upper']}/{latest['BOLL_mid']}/{latest['BOLL_lower']}
SAR: {latest['SAR']}, AVL: {latest['AVL']}, VOL: {latest['VOL']}, MACD: {latest['MACD']}
RSI: {latest['RSI']}, K: {latest['K']}, D: {latest['D']}, J: {latest['J']}
OBV: {latest['OBV']}, WR: {latest['WR']}, StochRSI: {latest['StochRSI']}

## Please provide:
- Explanation of each indicator
- Conclusion in Roman Urdu
- Support & Resistance levels
- Suggest Long/Short and leverage if user has $100
- Entry, Take Profit, Stop Loss
"""

            gpt_response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are an expert crypto analyst."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2,
                max_tokens=3000
            )

            analysis = gpt_response.choices[0].message.content
            st.subheader("Detailed Analysis & Reasoning:")
            st.text_area("", analysis, height=400)
            st.subheader("Main Conclusion:")
            st.write(analysis.split("## Conclusion:")[-1].strip() if "## Conclusion:" in analysis else "Conclusion missing.")

        except Exception as e:
            st.error(f"Error fetching or processing data: {str(e)}")
    else:
        st.error("Please provide either a chart screenshot or both Coin Symbol and Timeframe.")
