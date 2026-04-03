import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei']  # 微軟正黑體
plt.rcParams['axes.unicode_minus'] = False
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei']  # 微軟正黑體
plt.rcParams['axes.unicode_minus'] = False
import warnings
from typing import Dict

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf

warnings.filterwarnings("ignore", category=FutureWarning)

# =========================
# 基本設定
# =========================
ETF_MAP: Dict[str, str] = {
    "0050": "0050.TW",
    "006208": "006208.TW",
    "0052": "0052.TW",
    "00878": "00878.TW",
    "0056": "0056.TW",
    "00919": "00919.TW",
    "00881": "00881.TW",
    "00757": "00757.TW",
    "00891": "00891.TW",
    "00662": "00662.TW"
}

ETF_CATEGORY: Dict[str, str] = {
    "0050": "大盤型",
    "006208": "大盤型",
    "0052": "科技分散型",
    "00878": "高股息型",
    "0056": "高股息型",
    "00919": "高股息型",
    "00881": "科技型",
    "00757": "全球科技型",
    "00891": "半導體型",
    "00662": "美股成長型"
}

ETF_SUITABILITY: Dict[str, str] = {
    "0050": "適合",
    "006208": "適合",
    "0052": "適合",
    "00878": "適合",
    "0056": "普通",
    "00919": "適合",
    "00881": "普通",
    "00757": "普通",
    "00891": "進階",
    "00662": "進階"
}

START_DATE = "2021-01-01"
END_DATE = "2024-12-31"
MIN_DISPOSABLE = 3000
INVEST_RATIO = 0.30

# =========================
# Streamlit 基本設定
# =========================
st.set_page_config(page_title="大學生多標的 ETF 個人投資決策系統", layout="wide")

plt.rcParams["font.sans-serif"] = ["Microsoft JhengHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


# =========================
# 核心邏輯
# =========================
def evaluate(income: float, expense: float, fund: str):
    disposable = income - expense
    if fund == "沒有":
        return False, disposable, "沒有預備金"
    if disposable < MIN_DISPOSABLE:
        return False, disposable, f"可支配金額低於 {MIN_DISPOSABLE}"
    return True, disposable, "可投資"


def classify_risk_from_choice(choice: str):
    if choice == "1":
        return 10, "保守型"
    elif choice == "2":
        return 20, "穩健型"
    else:
        return 30, "積極型"


def get_allowed_categories_by_risk(risk_type: str):
    if risk_type == "保守型":
        return ["大盤型", "高股息型"]
    elif risk_type == "穩健型":
        return ["大盤型", "高股息型", "科技分散型", "科技型", "全球科技型"]
    else:
        return ["大盤型", "高股息型", "科技分散型", "科技型", "全球科技型", "半導體型", "美股成長型"]


def filter_etfs_by_risk(etf_category: Dict[str, str], risk_type: str):
    allowed_categories = get_allowed_categories_by_risk(risk_type)
    return {etf: category for etf, category in etf_category.items() if category in allowed_categories}


def generate_portfolio_advice(risk_type: str):
    if risk_type == "保守型":
        return {
            "主力": ["006208", "0050"],
            "輔助": ["00878", "00919"],
            "說明": "以穩定成長與低波動為主，適合剛開始投資的大學生。"
        }
    elif risk_type == "穩健型":
        return {
            "主力": ["0050", "006208"],
            "輔助": ["00878", "00919"],
            "成長": ["0052", "00881", "00757"],
            "說明": "兼顧成長與穩定，適合有一定風險承受能力的投資者。"
        }
    else:
        return {
            "主力": ["00881", "00757"],
            "進階": ["00891", "00662"],
            "分散": ["0050"],
            "說明": "以成長為導向，但波動較高，適合願意承擔較高風險的投資者。"
        }


# =========================
# 資料處理
# =========================
def flatten_single_cell(value):
    if isinstance(value, pd.Series):
        if value.empty:
            raise ValueError("Series 為空，無法轉成數值。")
        value = value.iloc[0]
    return float(value)


def extract_price_series(df: pd.DataFrame) -> pd.Series:
    if df is None or df.empty:
        raise ValueError("下載結果為空。")

    if not isinstance(df.columns, pd.MultiIndex):
        for col in ["Adj Close", "Close"]:
            if col in df.columns:
                s = df[col]
                if isinstance(s, pd.DataFrame):
                    s = s.iloc[:, 0]
                s = pd.to_numeric(s, errors="coerce").dropna()
                if not s.empty:
                    return s

    if isinstance(df.columns, pd.MultiIndex):
        level0 = list(df.columns.get_level_values(0).unique())
        for target in ["Adj Close", "Close"]:
            if target in level0:
                sub = df.xs(target, axis=1, level=0)
                if isinstance(sub, pd.DataFrame):
                    s = sub.iloc[:, 0]
                else:
                    s = sub
                s = pd.to_numeric(s, errors="coerce").dropna()
                if not s.empty:
                    return s

    s = df.iloc[:, 0]
    if isinstance(s, pd.DataFrame):
        s = s.iloc[:, 0]
    s = pd.to_numeric(s, errors="coerce").dropna()

    if s.empty:
        raise ValueError("找不到可用價格欄位。")

    return s


@st.cache_data(show_spinner=False)
def download_data():
    data = {}

    for name, ticker in ETF_MAP.items():
        try:
            df = yf.download(
                ticker,
                start=START_DATE,
                end=END_DATE,
                progress=False,
                auto_adjust=False,
                threads=False
            )

            if df.empty:
                continue

            series = extract_price_series(df)
            series.index = pd.to_datetime(series.index)
            series = series.sort_index()

            if not series.empty:
                data[name] = series

        except Exception:
            continue

    return data


def simulate(price: pd.Series, monthly_invest: float):
    if price is None or price.empty:
        empty_df = pd.DataFrame(columns=["date", "value", "invested", "price", "shares"])
        return empty_df, 0.0, 0.0, 0.0, 0.0, 0.0

    s = pd.to_numeric(price, errors="coerce").dropna()
    if s.empty:
        empty_df = pd.DataFrame(columns=["date", "value", "invested", "price", "shares"])
        return empty_df, 0.0, 0.0, 0.0, 0.0, 0.0

    df = pd.DataFrame({"price": s})
    df.index = pd.to_datetime(df.index)

    monthly_df = df.resample("MS").first().dropna()

    shares = 0.0
    total_invest = 0.0
    history_records = []

    for date, row in monthly_df.iterrows():
        p = flatten_single_cell(row["price"])
        if p <= 0:
            continue

        buy_shares = monthly_invest / p
        shares += buy_shares
        total_invest += monthly_invest
        value = shares * p

        history_records.append({
            "date": pd.to_datetime(date),
            "value": float(value),
            "invested": float(total_invest),
            "price": float(p),
            "shares": float(shares)
        })

    history_df = pd.DataFrame(history_records)

    if history_df.empty:
        empty_df = pd.DataFrame(columns=["date", "value", "invested", "price", "shares"])
        return empty_df, 0.0, 0.0, 0.0, 0.0, 0.0

    final_value = float(history_df["value"].iloc[-1])
    total_invest = float(history_df["invested"].iloc[-1])
    profit = final_value - total_invest
    roi = (profit / total_invest * 100) if total_invest > 0 else 0.0

    monthly_returns = history_df["value"].pct_change().dropna()
    volatility = float(monthly_returns.std() * 100) if not monthly_returns.empty else 0.0

    return history_df, final_value, total_invest, profit, roi, volatility


# =========================
# 推薦分數（新加）
# =========================
def get_suitability_bonus(label: str) -> float:
    if label == "適合":
        return 8.0
    elif label == "普通":
        return 3.0
    else:
        return 0.0


def calculate_recommendation_score(result: dict) -> float:
    """
    推薦分數：
    - 報酬率高加分
    - 波動度高扣分
    - 適合大學生加分
    """
    roi = result["roi"]
    volatility = result["volatility"]
    suitability_bonus = get_suitability_bonus(result["suitability"])

    score = roi - (volatility * 0.6) + suitability_bonus
    return score


# =========================
# 畫面
# =========================
st.title("大學生多標的 ETF 個人投資決策系統")
st.write("輸入你的資料後，系統會用真實 ETF 歷史資料模擬定期定額結果。")

income = st.number_input("請輸入每月收入", min_value=0.0, step=100.0)
expense = st.number_input("請輸入每月支出", min_value=0.0, step=100.0)
fund = st.selectbox("是否有預備金", ["有", "沒有"])

risk_choice = st.selectbox(
    "請選擇你的投資風險接受程度",
    ["1：幾乎不能虧（保守）", "2：小幅波動可接受（穩健）", "3：可以接受較大波動（積極）"]
)

if st.button("開始分析"):
    choice_value = risk_choice[0]
    risk_value, risk_type = classify_risk_from_choice(choice_value)

    ok, disposable, reason = evaluate(income, expense, fund)

    st.subheader("基本分析")
    st.write(f"每月可支配金額：{disposable:.0f} 元")

    if not ok:
        st.error(f"目前不建議投資：{reason}")
    else:
        monthly_invest = disposable * INVEST_RATIO
        filtered_etfs = filter_etfs_by_risk(ETF_CATEGORY, risk_type)
        advice = generate_portfolio_advice(risk_type)

        st.success("你目前具備基本投資條件。")
        st.write(f"建議每月投資金額：{monthly_invest:.0f} 元")
        st.write(f"風險類型：{risk_type}")

        st.subheader("依風險篩選後可考慮的 ETF")
        for etf, category in filtered_etfs.items():
            st.write(f"- {etf}（{category}）")

        st.subheader("投資組合建議")
        for key, value in advice.items():
            if key != "說明":
                st.write(f"{key}配置：{'、'.join(value)}")
        st.info(advice["說明"])

        st.subheader("下載並模擬 ETF 資料")
        with st.spinner("正在下載 ETF 歷史資料並模擬中..."):
            data = download_data()

        if not data:
            st.error("目前無法下載 ETF 資料，請稍後再試。")
        else:
            histories = {}
            results = {}

            for etf, price in data.items():
                if etf not in filtered_etfs:
                    continue

                try:
                    history_df, final, total, profit, roi, volatility = simulate(price, monthly_invest)

                    if history_df.empty:
                        continue

                    results[etf] = {
                        "final_value": final,
                        "total_invest": total,
                        "profit": profit,
                        "roi": roi,
                        "volatility": volatility,
                        "category": ETF_CATEGORY.get(etf, "未分類"),
                        "suitability": ETF_SUITABILITY.get(etf, "未知")
                    }
                    results[etf]["recommend_score"] = calculate_recommendation_score(results[etf])

                    histories[etf] = history_df

                except Exception:
                    continue

            if not results:
                st.error("沒有可用的模擬結果。")
            else:
                # 整體最終資產最高
                best_overall = max(results, key=lambda x: results[x]["final_value"])

                # 風險偏好下最推薦
                best_recommended = max(results, key=lambda x: results[x]["recommend_score"])

                col1, col2 = st.columns(2)

                with col1:
                    st.subheader("整體績效最高")
                    st.write(f"ETF：{best_overall}")
                    st.write(f"ETF 類型：{results[best_overall]['category']}")
                    st.write(f"最終資產：{results[best_overall]['final_value']:.0f} 元")

                with col2:
                    st.subheader("依你的風險偏好較推薦")
                    st.write(f"ETF：{best_recommended}")
                    st.write(f"ETF 類型：{results[best_recommended]['category']}")
                    st.write(f"推薦分數：{results[best_recommended]['recommend_score']:.2f}")

                st.subheader("ETF 模擬結果表")
                table_data = []
                for etf, result in sorted(results.items(), key=lambda x: x[1]["final_value"], reverse=True):
                    table_data.append({
                        "ETF": etf,
                        "類型": result["category"],
                        "適合度": result["suitability"],
                        "投入本金": round(result["total_invest"]),
                        "最終資產": round(result["final_value"]),
                        "總報酬": round(result["profit"]),
                        "報酬率(%)": round(result["roi"], 2),
                        "波動度(%)": round(result["volatility"], 2),
                        "推薦分數": round(result["recommend_score"], 2),
                    })

                result_df = pd.DataFrame(table_data)
                st.dataframe(result_df, use_container_width=True)

                st.subheader("ETF 成長圖")
                fig1, ax1 = plt.subplots(figsize=(12, 6))
                for etf, df in histories.items():
                    if etf == best_overall:
                        ax1.plot(df["date"], df["value"], linewidth=3, label=f"{etf} ⭐績效最高")
                    elif etf == best_recommended:
                        ax1.plot(df["date"], df["value"], linewidth=2.5, label=f"{etf} ✅較推薦")
                    else:
                        ax1.plot(df["date"], df["value"], linestyle="--", linewidth=1.6, label=etf)

                ax1.set_title("多標的 ETF 定期定額投資成長（真實歷史資料）")
                ax1.set_xlabel("時間")
                ax1.set_ylabel("資產（元）")
                ax1.legend()
                ax1.grid(True)
                st.pyplot(fig1)

                st.subheader("不同 ETF 最終資產比較")
                sorted_items = sorted(results.items(), key=lambda x: x[1]["final_value"], reverse=True)
                names = [k for k, _ in sorted_items]
                values = [v["final_value"] for _, v in sorted_items]

                fig2, ax2 = plt.subplots(figsize=(10, 5))
                bars = ax2.bar(names, values)

                for bar, name, value in zip(bars, names, values):
                    ax2.text(
                        bar.get_x() + bar.get_width() / 2,
                        value,
                        f"{int(value)}",
                        ha="center",
                        va="bottom",
                        fontsize=9
                    )

                ax2.set_title("不同 ETF 最終資產比較")
                ax2.set_xlabel("ETF")
                ax2.set_ylabel("最終資產（元）")
                ax2.grid(axis="y")
                st.pyplot(fig2)
