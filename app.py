import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as plotly_go
from plotly.subplots import make_subplots
import datetime
from config import Config
from train import run_training_from_df
from agent.dqn_agent import DQNAgent

# 設定頁面配置 (必須是第一個 Streamlit 指令)
st.set_page_config(page_title="PD Array & DRL Prediction Platform", layout="wide")

# 初始化設定
cfg = Config()

def load_data(ticker, start, end):
    try:
        df = yf.download(ticker, start=start, end=end)
        if df.empty:
            return None
        
        # 處理 yf 回傳的 MultiIndex
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.droplevel(1)
            
        df = df.reset_index()
        # 統一欄位名稱
        rename_map = {"Date": "date", "Datetime": "date", "Open": "open", "High": "high", "Low": "low", "Close": "close", "Volume": "volume"}
        df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns}, inplace=True)
        
        # 特徵工程 (PD Array)
        window = cfg.rolling_window
        df["range_high"] = df["high"].rolling(window).max()
        df["range_low"] = df["low"].rolling(window).min()
        
        denom = (df["range_high"] - df["range_low"]).replace(0, np.nan)
        df["pd_pos"] = (df["close"] - df["range_low"]) / denom
        df["pd_pos"] = df["pd_pos"].clip(0.0, 1.0)
        df["is_premium"] = (df["pd_pos"] > 0.5).astype(int)
        df["is_discount"] = (df["pd_pos"] < 0.5).astype(int)
        df = df.bfill().ffill()
        df = df.fillna(0)
        return df
    except Exception as e:
        st.error(f"資料獲取失敗: {e}")
        return None

def main():
    # 頂部列
    col_input1, col_input2, col_input3, col_btn = st.columns([1.5, 1.5, 1.5, 1])
    with col_input1:
        ticker = st.text_input("選擇資料來源 (股票代號)", value="AAPL")
    with col_input2:
        start_date = st.date_input("開始時間", value=datetime.date.today() - datetime.timedelta(days=365))
    with col_input3:
        end_date = st.date_input("結束時間", value=datetime.date.today())
    with col_btn:
        st.write("") # 對齊用
        start_btn = st.button("▶ 開始 PD Array", use_container_width=False)

    st.divider()

    # 主要版面分割
    col_left, col_right = st.columns([2.5, 1])

    with col_left:
        # 圖表區
        chart_container = st.container(border=True)
        chart_container.write("📄 K 線圖與 PD Array")
        chart_placeholder = chart_container.empty()
        
        # 訓練狀態區
        log_container = st.container(border=True)
        log_container.write("💻 PD RL 訓練模型動態")
        log_placeholder = log_container.empty()
        
    with col_right:
        # 報告區
        report_container = st.container(border=True)
        # 設定固定高度
        report_container.write("📄 分析建議報告")
        report_placeholder = report_container.empty()
        
    # 初始化畫面
    if not start_btn:
        chart_placeholder.info("等待抓取資料繪製圖表...")
        log_placeholder.info("等待訓練開始...")
        report_placeholder.info("等待模型訓練完成...")
        return

    # 按下按鈕後的邏輯
    with st.spinner("獲取股票資料中..."):
        df = load_data(ticker, start_date, end_date)
        
    if df is None:
        st.error("無法獲取股票資料，請檢查代號或時間。")
        return
        
    # 繪圖
    with chart_placeholder.container():
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=[0.7, 0.3])
        
        # 把這一段轉成 category 型態避免因為假日無開盤導致圖表變形、斷層、或是放大消失
        # 把 date 轉為字串格式
        df['date_str'] = df['date'].dt.strftime('%Y-%m-%d')
        
        # K線圖 (x使用字串時間，讓X軸變為連續類別)
        fig.add_trace(plotly_go.Candlestick(x=df['date_str'], open=df['open'], high=df['high'], low=df['low'], close=df['close'], name='K線'), row=1, col=1)
        
        # PD Array 指標圖
        fig.add_trace(plotly_go.Scatter(x=df['date_str'], y=df['pd_pos'], name='PD Position', line=dict(color='blue')), row=2, col=1)
        # 添加 0.5 的基準線
        fig.add_shape(type="line", x0=df['date_str'].iloc[0], y0=0.5, x1=df['date_str'].iloc[-1], y1=0.5, line=dict(color="red", width=1, dash="dash"), row=2, col=1)
        
        # 隱藏下方 rangeslider，關閉原本強制顯示假日的行為 (type="category")
        fig.update_layout(height=550, margin=dict(l=0, r=0, t=0, b=0), xaxis_rangeslider_visible=False, xaxis_type="category")
        fig.update_xaxes(type="category", nticks=10) # 防止 X 軸標籤擠在一起
        
        st.plotly_chart(fig)

    # 模型訓練
    log_placeholder.info("🚀 開始進行 DRL 模型訓練...")
    
    try:
        ret = run_training_from_df(df, cfg)
        log_text = f"✅ 訓練完成！\n\n"
        log_text += f"- **總獎勵 (Total Reward)**: {ret.get('total_reward', 0):.2f}\n"
        log_text += f"- **最終權益 (Final Equity)**: {ret.get('final_equity', 0):.2f}\n"
        log_text += f"- **勝率 (Win Rate)**: {ret.get('win_rate', 0)*100:.2f}%\n"
        log_text += f"- **夏普比率 (Sharpe Ratio)**: {ret.get('sharpe_ratio', 0):.2f}\n"
        log_placeholder.success(log_text)
    except Exception as e:
        log_placeholder.error(f"❌ 訓練中發生錯誤: {e}")
        return

    # 模型推論與報告
    report_placeholder.info("🧠 進行最新一筆資料推論...")
    agent = DQNAgent(state_dim=cfg.state_dim, action_dim=cfg.action_dim)
    if cfg.model_path.exists():
        agent.load(str(cfg.model_path))
        
    latest_action = 0
    dummy_state = np.zeros(cfg.state_dim)
    if cfg.state_dim >= 3:
        dummy_state[0] = df["pd_pos"].iloc[-1]
        dummy_state[1] = df["is_premium"].iloc[-1]
        dummy_state[2] = df["is_discount"].iloc[-1]

    try:
        latest_action = agent.select_action(dummy_state, greedy=True)
    except Exception as e:
        latest_action = 0

    action_map = {0: "⚪ 持有 (Hold)", 1: "🟢 買入 (Buy)", 2: "🔴 賣出 (Sell)"}
    
    report_md = f"""
    ### 📝 **{ticker}** 分析報告
    
    * **資料日期**: {df['date'].iloc[-1].strftime('%Y-%m-%d')}
    * **最新收盤價**: {df['close'].iloc[-1]:.2f}
    * **PD Array 狀態**: {'Premium (溢價區)' if df['is_premium'].iloc[-1] else 'Discount (折價區) / Equilibrium'}
    * **模型建議動作**: **{action_map.get(latest_action, '未知')}**
    
    ---
    **模型成效概要**:
    近期訓練達到勝率 {ret.get('win_rate', 0)*100:.1f}%，夏普值 {ret.get('sharpe_ratio', 0):.2f}。
    建議將結果搭配大盤環境綜合評估後操作。
    """
    report_placeholder.markdown(report_md)

if __name__ == '__main__':
    main()
