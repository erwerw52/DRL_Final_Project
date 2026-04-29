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

from utils.data_utils import prepare_data_from_df

# 設定頁面配置 (必須是第一個 Streamlit 指令)
st.set_page_config(page_title="SMC & DRL Prediction Platform", layout="wide")

# 初始化設定
cfg = Config()

def load_data_raw(ticker):
    try:
        # 強制抓取兩年小時級別資料
        df = yf.download(ticker, period="730d", interval="1h")
        if df.empty:
            return None
        
        # 處理 yf 回傳的 MultiIndex
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.droplevel(1)
            
        df = df.reset_index()
        # 統一欄位名稱
        rename_map = {"Date": "date", "Datetime": "date", "Open": "open", "High": "high", "Low": "low", "Close": "close", "Volume": "volume"}
        df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns}, inplace=True)
        
        # 轉換為台灣時間 (Asia/Taipei)
        if pd.api.types.is_datetime64_any_dtype(df['date']):
            if df['date'].dt.tz is not None:
                # 若帶有時區，直接轉換成台北時間，並移除時區標籤以利後續運算
                df['date'] = df['date'].dt.tz_convert('Asia/Taipei').dt.tz_localize(None)
            else:
                # 若沒有時區標籤，預設當作 UTC 處理後再轉
                df['date'] = df['date'].dt.tz_localize('UTC').dt.tz_convert('Asia/Taipei').dt.tz_localize(None)
                
        return df
    except Exception as e:
        st.error(f"資料獲取失敗: {e}")
        return None

def process_data(raw_df, interval, rolling_window):
    df = raw_df.copy()
    df.set_index("date", inplace=True)
    
    if interval == "1h":
        pass
    elif interval == "2h":
        df = df.resample('2h').agg({'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'}).dropna()
    elif interval == "4h":
        df = df.resample('4h').agg({'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'}).dropna()
    elif interval == "1d":
        df = df.resample('D').agg({'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'}).dropna()
    elif interval == "1wk":
        # 週線以週一為起始
        df = df.resample('W-MON').agg({'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'}).dropna()
    elif interval == "1mo":
        df = df.resample('ME').agg({'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'}).dropna()
        
    df.reset_index(inplace=True)
    
    # 檢查長度是否足夠計算 SMC
    if len(df) < rolling_window:
        st.warning(f"當前資料數量 ({len(df)}) 小於設定的滾動窗口 ({rolling_window})，這可能導致計算 SMC 特徵失敗。")
        
    # 使用 data_utils 準備包含 SMC 數據
    df = prepare_data_from_df(df, rolling_window=rolling_window)
    return df

def main():
    # 注入 CSS 來自訂按鈕樣式為天空藍
    st.markdown("""
        <style>
        .stButton > button {
            background-color: #87CEEB !important; /* 天空藍 */
            color: #1E1E1E !important; /* 深色文字增加對比度與易讀性 */
            border-color: #87CEEB !important;
            font-weight: bold !important;
        }
        .stButton > button:hover {
            background-color: #00BFFF !important; /* 滑鼠懸停變成更亮/深的天空藍 */
            color: #ffffff !important;
            border-color: #00BFFF !important;
        }
        </style>
    """, unsafe_allow_html=True)

    # 頂部列
    col_input1, col_input2, col_btn = st.columns([2, 1.5, 1], vertical_alignment="bottom")
    with col_input1:
        ticker = st.text_input("選擇資料來源 (股票代號 EX. AAPL or 2330.TW)", value="")
    with col_input2:
        interval_option = st.selectbox("時間級別", ["1h", "2h", "4h", "1d", "1wk", "1mo"], index=4)
    with col_btn:
        start_btn = st.button("▶ 獲取資料並分析", width=190)

    st.divider()

    # 主要版面分割
    col_left, col_right = st.columns([2.5, 1])

    with col_left:
        # 圖表區
        chart_container = st.container(border=True)
        chart_container.write("📄 SMC (Smart Money Concepts) 盤面解析")
        chart_placeholder = chart_container.empty()
        
    with col_right:
        # 報告區
        report_container = st.container(border=True)
        # 設定固定高度
        report_container.write("📄 分析建議報告")
        report_placeholder = report_container.empty()

    # 訓練狀態區移到下方全寬
    log_container = st.container(border=True)
    log_container.write("💻 DRL 模型訓練動態 Log")
    log_placeholder = log_container.empty()
    train_btn_placeholder = log_container.empty()
        
    # 初始化畫面
    if start_btn:
        if not ticker:
            st.warning("請輸入股票代號")
            return
        
        # 按下分析時，清除舊的訓練紀錄，確保右側報告區重置
        st.session_state.pop("model_ret", None)
        
        with st.spinner("取過去 2 年的小時級別資料中..."):
            raw_df = load_data_raw(ticker)
            if raw_df is not None:
                st.session_state["raw_df"] = raw_df
                st.session_state["ticker"] = ticker
            else:
                st.error("無法獲取股票資料，請檢查代號。")
                return

    if "raw_df" not in st.session_state:
        chart_placeholder.info("等待抓取資料繪製圖表...")
        log_placeholder.info("等待訓練開始...")
        report_placeholder.info("等待模型訓練完成...")
        return
        
    raw_df = st.session_state["raw_df"]
    ticker = st.session_state.get("ticker", "UNKNOWN")
    
    with chart_placeholder.container():
        with st.spinner(f"正在組合 {interval_option} 級別並計算 SMC 特徵..."):
            try:
                df = process_data(raw_df, interval_option, cfg.rolling_window)
            except Exception as e:
                st.error(f"資料轉換或 SMC 計算失敗: {e}")
                return
            
    # 繪圖
    with chart_placeholder.container():
        # 改為單一主圖
        fig = plotly_go.Figure()
        
        if interval_option in ['1h', '2h', '4h']:
            df['date_str'] = df['date'].dt.strftime('%Y-%m-%d %H:%M')
        else:
            df['date_str'] = df['date'].dt.strftime('%Y-%m-%d')
        
        # K線圖
        fig.add_trace(plotly_go.Candlestick(x=df['date_str'], open=df['open'], high=df['high'], low=df['low'], close=df['close'], name='K線'))
        
        # 標示 Old Highs (阻力) / Old Lows (支撐)
        fig.add_trace(plotly_go.Scatter(x=df['date_str'], y=df['old_high'], mode='lines', name='Old High (BSL)', line=dict(color='red', width=1, dash='dash')))
        fig.add_trace(plotly_go.Scatter(x=df['date_str'], y=df['old_low'], mode='lines', name='Old Low (SSL)', line=dict(color='green', width=1, dash='dash')))
            
        # 標示 Order Block - 改用 filled Scatter 取代 shape 以支援圖例開關
        if "ob" in df.columns:
            ob_pos_x, ob_pos_y, ob_neg_x, ob_neg_y = [], [], [], []
            for i, row in df[df['ob'] != 0].iterrows():
                x0 = row['date_str']
                x1 = df['date_str'].iloc[-1] if i == len(df)-1 else df['date_str'].iloc[min(i+10, len(df)-1)]
                y0 = row.get('ob_bottom', row['low'])
                y1 = row.get('ob_top', row['high'])
                
                if row['ob'] < 0:
                    ob_neg_x.extend([x0, x0, x1, x1, None])
                    ob_neg_y.extend([y0, y1, y1, y0, None])
                else:
                    ob_pos_x.extend([x0, x0, x1, x1, None])
                    ob_pos_y.extend([y0, y1, y1, y0, None])
                    
            if ob_pos_x:
                fig.add_trace(plotly_go.Scatter(x=ob_pos_x, y=ob_pos_y, fill='toself', fillcolor='rgba(0, 255, 0, 0.2)', mode='lines', line=dict(width=0), name='+ OB (看漲)'))
            if ob_neg_x:
                fig.add_trace(plotly_go.Scatter(x=ob_neg_x, y=ob_neg_y, fill='toself', fillcolor='rgba(255, 0, 0, 0.2)', mode='lines', line=dict(width=0), name='- OB (看跌)'))

        # 標示 Fair Value Gap - 改用 filled Scatter 取代 shape 以支援圖例開關
        if "fvg" in df.columns:
            fvg_pos_x, fvg_pos_y, fvg_neg_x, fvg_neg_y = [], [], [], []
            for i, row in df[df['fvg'] != 0].iterrows():
                x0 = row['date_str']
                x1 = df['date_str'].iloc[-1] if i == len(df)-1 else df['date_str'].iloc[min(i+3, len(df)-1)]
                y0 = row.get('fvg_bottom', row['low'])
                y1 = row.get('fvg_top', row['high'])
                
                if row['fvg'] < 0:
                    fvg_neg_x.extend([x0, x0, x1, x1, None])
                    fvg_neg_y.extend([y0, y1, y1, y0, None])
                else:
                    fvg_pos_x.extend([x0, x0, x1, x1, None])
                    fvg_pos_y.extend([y0, y1, y1, y0, None])
                    
            if fvg_pos_x:
                fig.add_trace(plotly_go.Scatter(x=fvg_pos_x, y=fvg_pos_y, fill='toself', fillcolor='rgba(0, 191, 255, 0.2)', mode='lines', line=dict(width=0), name='+ FVG (向上缺口)'))
            if fvg_neg_x:
                fig.add_trace(plotly_go.Scatter(x=fvg_neg_x, y=fvg_neg_y, fill='toself', fillcolor='rgba(255, 165, 0, 0.2)', mode='lines', line=dict(width=0), name='- FVG (向下缺口)'))

        # 標示 Liquidity Pools
        if "liq_swept" in df.columns:
            liq_df = df[df['liq_swept'] != 0]
            if not liq_df.empty:
                fig.add_trace(plotly_go.Scatter(x=liq_df['date_str'], y=liq_df['high'] * 1.01, mode='markers', name='Liquidity Swept', marker=dict(symbol='x', color='purple', size=8)))
            else:
                # 即使沒發生也加個空標示提醒使用者該區間內沒有觸發
                fig.add_trace(plotly_go.Scatter(x=[df['date_str'].iloc[0]], y=[np.nan], mode='markers', name='無流動性獵取', marker=dict(symbol='x', color='purple', size=8)))
        
        fig.update_layout(height=550, margin=dict(l=0, r=0, t=30, b=0), xaxis_rangeslider_visible=False, xaxis_type="category", title="SMC Price Action")
        fig.update_xaxes(type="category", nticks=10)
        
        st.plotly_chart(fig, width='stretch')

    train_btn = train_btn_placeholder.button(f"🚀 訓練 DRL 模型", width=180)

    if train_btn:
        # 模型訓練
        # 為了避免被 overwrite，我們在原本的 container 中建立獨立的區塊
        log_status = log_container.empty()
        log_area = log_container.empty()
        
        log_status.info("🚀 開始進行 DRL 模型訓練...")
        
        # 建立具備固定高度且可滑動的動態 Log 視窗
        log_messages = []

        def update_log(msg):
            log_messages.append(msg)
            # 透過 HTML 與 CSS column-reverse 技巧，讓捲軸永遠保持在最底部
            display_text = "\n".join(log_messages)
            html_code = f"""
            <div style="background-color: #1e1e1e; color: #00ff00; padding: 10px; border-radius: 5px; font-family: Consolas, monospace; font-size: 14px; height: 280px; display: flex; flex-direction: column-reverse; overflow-y: auto; border: 1px solid #444;">
                <div style="white-space: pre-wrap;">{display_text}</div>
            </div>
            """
            log_area.markdown(html_code, unsafe_allow_html=True)
        
        try:
            ret = run_training_from_df(df, cfg, progress_callback=update_log)
            # 更新 Session State
            st.session_state["model_ret"] = ret
            
            # 訓練完成後，在 log 區額外加上完成資訊
            final_msg = f"✅ 訓練完成！勝率: {ret.get('win_rate', 0)*100:.1f}% | 夏普值: {ret.get('sharpe_ratio', 0):.2f}"
            update_log(final_msg)
            
            # 將狀態從 info 改為 success，不覆蓋下方的 code 區塊
            log_status.success("✅ 訓練已儲存至模型中！")
            
        except Exception as e:
            log_status.error(f"❌ 訓練中發生錯誤: {e}")
            return
            
    # 模型推論與報告
    ret = st.session_state.get("model_ret", {})
    if not ret and not train_btn:
        log_placeholder.info("等待新的模型訓練...")
        report_placeholder.info("⏳ 等待模型訓練完成...")
        return
        
    report_placeholder.info("🧠 進行最新一筆資料推論...")
    agent = DQNAgent(state_dim=cfg.state_dim, action_dim=cfg.action_dim)
    if cfg.model_path.exists():
        agent.load(str(cfg.model_path))
        
    latest_action = 0
    dummy_state = np.zeros(cfg.state_dim)
    if cfg.state_dim >= 12:
        dummy_state[0] = df["pd_pos"].iloc[-1]
        dummy_state[1] = df["is_premium"].iloc[-1]
        dummy_state[2] = df["is_discount"].iloc[-1]
        dummy_state[7] = df["fvg"].iloc[-1]
        dummy_state[8] = df["ob"].iloc[-1]
        dummy_state[9] = df["liq_swept"].iloc[-1]
        dummy_state[10] = df["dist_old_high"].iloc[-1]
        dummy_state[11] = df["dist_old_low"].iloc[-1]

    try:
        latest_action = agent.select_action(dummy_state, greedy=True)
    except Exception as e:
        latest_action = 0

    action_map = {0: "⚪ 持有 (Hold)", 1: "🟢 買入 (Buy)", 2: "🔴 賣出 (Sell)"}
    
    # 判斷趨勢或狀態
    smc_status = []
    if df["is_premium"].iloc[-1]: smc_status.append("Premium")
    if df["is_discount"].iloc[-1]: smc_status.append("Discount")
    if df["liq_swept"].iloc[-1] != 0: smc_status.append("Liquidity Swept")
    
    report_md = f"""
    ### 📝 **{ticker}** SMC 報告
    
    * **資料日期**: {df['date'].iloc[-1].strftime('%Y-%m-%d')}
    * **最新收盤價**: {df['close'].iloc[-1]:,.2f}
    * **目前 SMC 狀態**: {', '.join(smc_status) if smc_status else 'Equilibrium'}
    * **模型建議動作**: **{action_map.get(latest_action, '未知')}**
    
    ---
    **模型成效概要**:
    勝率 {ret.get('win_rate', 0)*100:.1f}%, 夏普值 {ret.get('sharpe_ratio', 0):.2f}, 最終獲利 {ret.get('total_reward', 0):,.2f}
    """
    report_placeholder.markdown(report_md)

if __name__ == '__main__':
    main()
