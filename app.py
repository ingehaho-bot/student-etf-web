import warnings
from typing import Dict
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import yfinance as yf

warnings.filterwarnings("ignore", category=FutureWarning)

# =========================
# Streamlit page config
# =========================
st.set_page_config(
    page_title="Student ETF Investment Decision System",
    layout="wide"
)

plt.rcParams["axes.unicode_minus"] = False

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
    "00662": "00662.TW",
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
    "00662": "美股成長型",
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
    "00662": "進階",
}

MIN_DISPOSABLE = 3000
INVEST_RATIO = 0.30

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
def download_data(start_date: str, end_date: str):
    data = {}

    for name, ticker in ETF_MAP.items():
        try:
            df = yf.download(
                ticker,
                start=start_date,
                end=end_date,
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
        empty_df = pd.DataFrame(columns=["date", "value", "invested", "price", "shares", "buy_shares"])
        return empty_df, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

    s = pd.to_numeric(price, errors="coerce").dropna()
    if s.empty:
        empty_df = pd.DataFrame(columns=["date", "value", "invested", "price", "shares", "buy_shares"])
        return empty_df, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

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
            "shares": float(shares),
            "buy_shares": float(buy_shares)
        })

    history_df = pd.DataFrame(history_records)

    if history_df.empty:
        empty_df = pd.DataFrame(columns=["date", "value", "invested", "price", "shares", "buy_shares"])
        return empty_df, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

    final_value = float(history_df["value"].iloc[-1])
    total_invest = float(history_df["invested"].iloc[-1])
    profit = final_value - total_invest
    roi = (profit / total_invest * 100) if total_invest > 0 else 0.0
    monthly_returns = history_df["value"].pct_change().dropna()
    volatility = float(monthly_returns.std() * 100) if not monthly_returns.empty else 0.0
    final_shares = float(history_df["shares"].iloc[-1])

    return history_df, final_value, total_invest, profit, roi, volatility, final_shares


# =========================
# 推薦分數
# =========================
def get_suitability_bonus(label: str) -> float:
    if label == "適合":
        return 8.0
    elif label == "普通":
        return 3.0
    else:
        return 0.0


def calculate_recommendation_score(result: dict) -> float:
    roi = result["roi"]
    volatility = result["volatility"]
    suitability_bonus = get_suitability_bonus(result["suitability"])
    score = roi - (volatility * 0.6) + suitability_bonus
    return score


# =========================
# 日期工具
# =========================
def get_simulation_period(years: int):
    end_date = pd.Timestamp.today().normalize()
    start_date = end_date - pd.DateOffset(years=years)
    return start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d")


# =========================
# UI 工具
# =========================
def show_value_with_explanation(label: str, value_text: str, explanation: str):
    st.markdown(f"**{label}：{value_text}**")
    st.caption(explanation)


def show_metric_card_with_explanation(title: str, value_text: str, explanation: str):
    st.metric(title, value_text)
    st.caption(explanation)


# =========================
# 畫面
# =========================
st.title("大學生多標的 ETF 個人投資決策系統")
st.write("輸入你的資料後，系統會用真實 ETF 歷史資料模擬定期定額結果。")

col1, col2 = st.columns(2)

with col1:
    income = st.number_input("請輸入每月收入", min_value=0.0, step=100.0)
    expense = st.number_input("請輸入每月支出", min_value=0.0, step=100.0)
    invest_years = st.selectbox("預計投資年數", [1, 3, 5, 10], index=1)

with col2:
    fund = st.selectbox("是否有預備金", ["有", "沒有"])
    risk_choice = st.selectbox(
        "請選擇你的投資風險接受程度",
        ["1：幾乎不能虧（保守）", "2：小幅波動可接受（穩健）", "3：可以接受較大波動（積極）"]
    )

if st.button("開始分析", use_container_width=True):
    choice_value = risk_choice[0]
    risk_value, risk_type = classify_risk_from_choice(choice_value)

    ok, disposable, reason = evaluate(income, expense, fund)

    st.subheader("基本分析")
    show_value_with_explanation(
        "每月可支配金額",
        f"{disposable:.0f} 元",
        f"計算方式：每月收入 {income:.0f} 元 − 每月支出 {expense:.0f} 元 = {disposable:.0f} 元。"
    )
    show_value_with_explanation(
        "預計投資年數",
        f"{invest_years} 年",
        f"這個值由使用者自行設定，系統會依照這個年數決定模擬區間。"
    )

    if not ok:
        st.error(f"目前不建議投資：{reason}")
        if fund == "沒有":
            st.caption("原因說明：系統設定為若沒有預備金，則優先保留現金，不先進行投資。")
        elif disposable < MIN_DISPOSABLE:
            st.caption(f"原因說明：系統設定最低可支配金額門檻為 {MIN_DISPOSABLE} 元，低於此數值則不建議投資。")
    else:
        monthly_invest = disposable * INVEST_RATIO
        filtered_etfs = filter_etfs_by_risk(ETF_CATEGORY, risk_type)
        advice = generate_portfolio_advice(risk_type)
        start_date, end_date = get_simulation_period(invest_years)

        st.success("你目前具備基本投資條件。")
        show_value_with_explanation(
            "建議每月投資金額",
            f"{monthly_invest:.0f} 元",
            f"計算方式：可支配金額 {disposable:.0f} 元 × 投資比例 {int(INVEST_RATIO * 100)}% = {monthly_invest:.0f} 元。"
        )
        show_value_with_explanation(
            "風險類型",
            risk_type,
            f"這是根據你選擇的風險偏好選項所對應出的分類結果。"
        )
        show_value_with_explanation(
            "模擬期間",
            f"{start_date} ～ {end_date}",
            f"計算方式：以今天往前回推 {invest_years} 年，作為歷史模擬區間。"
        )

        st.subheader("依風險篩選後可考慮的 ETF")
        st.write("、".join([f"{etf}（{category}）" for etf, category in filtered_etfs.items()]))
        st.caption("篩選方式：系統會先依據你的風險類型，保留相對應類型的 ETF，例如保守型主要保留大盤型與高股息型。")

        st.subheader("投資組合建議")
        for key, value in advice.items():
            if key != "說明":
                st.write(f"{key}配置：{'、'.join(value)}")
                st.caption(f"{key}配置說明：這是依照 {risk_type} 的風險屬性，優先分配的 ETF 類型。")
        st.info(advice["說明"])

        st.subheader("下載並模擬 ETF 資料")
        with st.spinner("正在下載 ETF 歷史資料並模擬中..."):
            data = download_data(start_date, end_date)

        if not data:
            st.error("目前無法下載 ETF 資料，請稍後再試。")
        else:
            histories = {}
            results = {}

            for etf, price in data.items():
                if etf not in filtered_etfs:
                    continue

                try:
                    history_df, final, total, profit, roi, volatility, final_shares = simulate(price, monthly_invest)

                    if history_df.empty:
                        continue

                    results[etf] = {
                        "final_value": final,
                        "total_invest": total,
                        "profit": profit,
                        "roi": roi,
                        "volatility": volatility,
                        "category": ETF_CATEGORY.get(etf, "未分類"),
                        "suitability": ETF_SUITABILITY.get(etf, "未知"),
                        "final_shares": final_shares,
                        "last_price": float(history_df["price"].iloc[-1]),
                    }
                    results[etf]["recommend_score"] = calculate_recommendation_score(results[etf])
                    histories[etf] = history_df

                except Exception:
                    continue

            if not results:
                st.error("沒有可用的模擬結果。")
            else:
                best_overall = max(results, key=lambda x: results[x]["final_value"])
                best_recommended = max(results, key=lambda x: results[x]["recommend_score"])

                c1, c2 = st.columns(2)

                with c1:
                    st.subheader("整體績效最高")
                    show_value_with_explanation(
                        "ETF",
                        best_overall,
                        "這是所有已模擬 ETF 中，最終資產最高的一檔。"
                    )
                    show_value_with_explanation(
                        "ETF 類型",
                        results[best_overall]["category"],
                        "這是該 ETF 在系統中的商品分類。"
                    )
                    show_value_with_explanation(
                        "最終資產",
                        f"{results[best_overall]['final_value']:.0f} 元",
                        f"計算方式：最後持有股數 {results[best_overall]['final_shares']:.4f} 股 × 最後價格 {results[best_overall]['last_price']:.2f} 元 = {results[best_overall]['final_value']:.0f} 元。"
                    )

                with c2:
                    st.subheader("依你的風險偏好較推薦")
                    show_value_with_explanation(
                        "ETF",
                        best_recommended,
                        "這是綜合考慮報酬、波動度與大學生適合度後，分數最高的一檔。"
                    )
                    show_value_with_explanation(
                        "ETF 類型",
                        results[best_recommended]["category"],
                        "這是該 ETF 的分類結果。"
                    )
                    show_value_with_explanation(
                        "推薦分數",
                        f"{results[best_recommended]['recommend_score']:.2f}",
                        f"計算方式：報酬率 {results[best_recommended]['roi']:.2f} − 波動度 {results[best_recommended]['volatility']:.2f} × 0.6 + 適合度加分 = {results[best_recommended]['recommend_score']:.2f}。"
                    )

                st.subheader("最終資產計算說明")
                explain_etf = best_overall
                st.write(
                    f"以 {explain_etf} 為例，系統是用「每月固定投入 {monthly_invest:.0f} 元」"
                    f"在模擬期間內每月第一個交易日買入 ETF。"
                )
                st.write(
                    f"最後持有股數約為 {results[explain_etf]['final_shares']:.4f} 股，"
                    f"最後一期價格約為 {results[explain_etf]['last_price']:.2f} 元。"
                )
                st.write(
                    f"因此最終資產 ≈ 持有股數 × 最後價格 = "
                    f"{results[explain_etf]['final_shares']:.4f} × {results[explain_etf]['last_price']:.2f} "
                    f"= {results[explain_etf]['final_value']:.0f} 元。"
                )

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
                st.caption("表格中的每個數值，皆由歷史價格模擬後計算得出，並非手動輸入。")

                st.subheader("各 ETF 數值解釋")
                selected_etf = st.selectbox("選擇一檔 ETF 查看各數值說明", list(results.keys()))
                r = results[selected_etf]

                m1, m2, m3 = st.columns(3)
                with m1:
                    show_metric_card_with_explanation(
                        "投入本金",
                        f"{r['total_invest']:.0f} 元",
                        f"計算方式：每月投資金額 {monthly_invest:.0f} 元 × 實際投入月份數。"
                    )
                with m2:
                    show_metric_card_with_explanation(
                        "最終資產",
                        f"{r['final_value']:.0f} 元",
                        f"計算方式：最後持有股數 {r['final_shares']:.4f} 股 × 最後價格 {r['last_price']:.2f} 元。"
                    )
                with m3:
                    show_metric_card_with_explanation(
                        "總報酬",
                        f"{r['profit']:.0f} 元",
                        f"計算方式：最終資產 {r['final_value']:.0f} 元 − 投入本金 {r['total_invest']:.0f} 元。"
                    )

                m4, m5, m6 = st.columns(3)
                with m4:
                    show_metric_card_with_explanation(
                        "報酬率",
                        f"{r['roi']:.2f}%",
                        f"計算方式：總報酬 {r['profit']:.0f} 元 ÷ 投入本金 {r['total_invest']:.0f} 元 × 100%。"
                    )
                with m5:
                    show_metric_card_with_explanation(
                        "波動度",
                        f"{r['volatility']:.2f}%",
                        "計算方式：根據每月資產變化率的標準差估算而來，數值越高代表波動越大。"
                    )
                with m6:
                    show_metric_card_with_explanation(
                        "推薦分數",
                        f"{r['recommend_score']:.2f}",
                        f"計算方式：報酬率 {r['roi']:.2f} − 波動度 {r['volatility']:.2f} × 0.6 + 適合度加分。"
                    )

                st.subheader("ETF 成長圖")
                fig1, ax1 = plt.subplots(figsize=(12, 6))
                for etf, df in histories.items():
                    if etf == best_overall:
                        ax1.plot(df["date"], df["value"], linewidth=3, label=f"{etf} - Best Performance")
                    elif etf == best_recommended:
                        ax1.plot(df["date"], df["value"], linewidth=2.5, label=f"{etf} - Recommended")
                    else:
                        ax1.plot(df["date"], df["value"], linestyle="--", linewidth=1.6, label=etf)

                ax1.set_title("ETF DCA Growth Comparison")
                ax1.set_xlabel("Date")
                ax1.set_ylabel("Portfolio Value")
                ax1.legend()
                ax1.grid(True)
                st.pyplot(fig1)
                st.caption("這張圖顯示不同 ETF 在定期定額投資下，資產如何隨時間累積成長。")

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

                ax2.set_title("Final Portfolio Value by ETF")
                ax2.set_xlabel("ETF")
                ax2.set_ylabel("Final Value")
                ax2.grid(axis="y")
                st.pyplot(fig2)
                st.caption("這張圖比較各 ETF 在相同投資條件下，最終資產的高低差異。")
