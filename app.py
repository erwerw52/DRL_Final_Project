import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as plotly_go
import datetime
from config import Config, ACTION_NAMES
from train import run_training_pipeline
from recommend import recommend_strategy
from utils.data_utils import FEATURE_COLUMNS, prepare_data_for_chart

# 設定頁面配置 (必須是第一個 Streamlit 指令)
st.set_page_config(page_title="SMC × DRL Trading Platform", layout="wide")

# 初始化設定
cfg = Config()

def load_data_raw(ticker, start_date, end_date):
    try:
        df = yf.download(ticker, start=str(start_date), end=str(end_date), interval="1h")
        if df.empty:
            return None
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.droplevel(1)
        df = df.reset_index()
        rename_map = {"Date": "date", "Datetime": "date", "Open": "open", "High": "high", "Low": "low", "Close": "close", "Volume": "volume"}
        df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns}, inplace=True)
        if pd.api.types.is_datetime64_any_dtype(df['date']):
            if df['date'].dt.tz is not None:
                df['date'] = df['date'].dt.tz_convert('Asia/Taipei').dt.tz_localize(None)
            else:
                df['date'] = df['date'].dt.tz_localize('UTC').dt.tz_convert('Asia/Taipei').dt.tz_localize(None)
        return df
    except Exception as e:
        st.error(f"Failed to fetch data: {e}")
        return None

def process_data_for_chart(raw_df, interval, rolling_window):
    """Process data with smartmoneyconcepts for chart display."""
    df = raw_df.copy()
    df.set_index("date", inplace=True)
    resample_rules = {"1h": None, "4h": "4h", "1d": "D", "1wk": "W-MON"}
    rule = resample_rules.get(interval)
    if rule:
        df = df.resample(rule).agg({'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'}).dropna()
    df.reset_index(inplace=True)
    if len(df) < rolling_window:
        st.warning(f"Current data count ({len(df)}) is less than rolling window ({rolling_window})")
    df = prepare_data_for_chart(df, rolling_window=rolling_window)
    return df

# ── 圖表渲染 Fragment（切換時區不會觸發整頁 rerun）──
@st.fragment
def render_chart():
    """圖表 Fragment：內部自行讀取資料並建立 UI，確保 fragment rerun 時正常更新。"""
    raw_df = st.session_state.get("raw_df")
    if raw_df is None:
        st.info("Waiting for data to render chart...")
        return

    interval_map = {"1h (H1)": "1h", "4h (H4)": "4h", "1d (D1)": "1d", "1wk (W1)": "1wk"}
    chart_tf = st.selectbox("Chart Timeframe", list(interval_map.keys()), index=0, key="chart_tf")
    interval_option = interval_map[chart_tf]

    with st.spinner(f"Aggregating {chart_tf} timeframe and calculating SMC features..."):
        try:
            df = process_data_for_chart(raw_df, interval_option, cfg.rolling_window)
        except Exception as e:
            st.error(f"Data conversion or SMC calculation failed: {e}")
            return

    fig = plotly_go.Figure()
    if interval_option in ['1h', '4h']:
        df['date_str'] = df['date'].dt.strftime('%Y-%m-%d %H:%M')
    else:
        df['date_str'] = df['date'].dt.strftime('%Y-%m-%d')

    fig.add_trace(plotly_go.Candlestick(x=df['date_str'], open=df['open'], high=df['high'], low=df['low'], close=df['close'], name='Candlesticks'))

    # Old Highs / Old Lows
    if 'old_high' in df.columns:
        fig.add_trace(plotly_go.Scatter(x=df['date_str'], y=df['old_high'], mode='lines', name='Old High (BSL)', line=dict(color='red', width=1, dash='dash')))
        fig.add_trace(plotly_go.Scatter(x=df['date_str'], y=df['old_low'], mode='lines', name='Old Low (SSL)', line=dict(color='green', width=1, dash='dash')))

    # Order Block
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
            fig.add_trace(plotly_go.Scatter(x=ob_pos_x, y=ob_pos_y, fill='toself', fillcolor='rgba(0, 255, 0, 0.2)', mode='lines', line=dict(width=0), name='+ OB (Bullish)'))
        if ob_neg_x:
            fig.add_trace(plotly_go.Scatter(x=ob_neg_x, y=ob_neg_y, fill='toself', fillcolor='rgba(255, 0, 0, 0.2)', mode='lines', line=dict(width=0), name='- OB (Bearish)'))

    # FVG
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
            fig.add_trace(plotly_go.Scatter(x=fvg_pos_x, y=fvg_pos_y, fill='toself', fillcolor='rgba(0, 191, 255, 0.2)', mode='lines', line=dict(width=0), name='+ FVG (Bullish Gap)'))
        if fvg_neg_x:
            fig.add_trace(plotly_go.Scatter(x=fvg_neg_x, y=fvg_neg_y, fill='toself', fillcolor='rgba(255, 165, 0, 0.2)', mode='lines', line=dict(width=0), name='- FVG (Bearish Gap)'))

    # Liquidity
    if "liq_swept" in df.columns:
        liq_df = df[df['liq_swept'] != 0]
        if not liq_df.empty:
            fig.add_trace(plotly_go.Scatter(x=liq_df['date_str'], y=liq_df['high'] * 1.01, mode='markers', name='Liquidity Swept', marker=dict(symbol='x', color='purple', size=8)))

    # ── DRL 測試集交易標記 ──
    ret_data = st.session_state.get("model_ret", {})
    bt_data = ret_data.get("test_backtest", {}) if ret_data else {}
    bt_trades_df = bt_data.get("trades_df") if bt_data else None

    if bt_trades_df is not None and not bt_trades_df.empty:
        def _nearest_candle(trade_dt):
            """找到圖表中最接近交易時間的 K 線。"""
            t = pd.Timestamp(trade_dt)
            diffs = (df['date'] - t).abs()
            idx = diffs.idxmin()
            return df.loc[idx, 'date_str'], df.loc[idx, 'low'], df.loc[idx, 'high']

        # ── 所有 BUY 標記 ──
        buy_rows = bt_trades_df[bt_trades_df['type'] == 'BUY']
        if not buy_rows.empty:
            bx, by, bc = [], [], []
            for _, t in buy_rows.iterrows():
                ds, low, _ = _nearest_candle(t['datetime'])
                bx.append(ds); by.append(low * 0.995)
                bc.append([str(t['datetime'])[:16], f"{t['price']:,.2f}", f"{t['value']:,.0f}", f"{t['cost']:,.2f}"])
            fig.add_trace(plotly_go.Scatter(
                x=bx, y=by, mode='markers', name='BUY Trade',
                marker=dict(symbol='triangle-up', color='#00E676', size=10, line=dict(color='white', width=1)),
                customdata=bc,
                hovertemplate='<b>BUY</b><br>Time: %{customdata[0]}<br>Price: %{customdata[1]}<br>Value: %{customdata[2]}<br>Fee: %{customdata[3]}<extra></extra>',
            ))

        # ── 所有 SELL 標記 ──
        sell_rows = bt_trades_df[bt_trades_df['type'] == 'SELL']
        if not sell_rows.empty:
            sx, sy, sc = [], [], []
            for _, t in sell_rows.iterrows():
                ds, _, high = _nearest_candle(t['datetime'])
                sx.append(ds); sy.append(high * 1.005)
                sc.append([str(t['datetime'])[:16], f"{t['price']:,.2f}", f"{t['value']:,.0f}", f"{t['cost']:,.2f}"])
            fig.add_trace(plotly_go.Scatter(
                x=sx, y=sy, mode='markers', name='SELL Trade',
                marker=dict(symbol='triangle-down', color='#FF5252', size=10, line=dict(color='white', width=1)),
                customdata=sc,
                hovertemplate='<b>SELL</b><br>Time: %{customdata[0]}<br>Price: %{customdata[1]}<br>Value: %{customdata[2]}<br>Fee: %{customdata[3]}<extra></extra>',
            ))

        # ── 配對 BUY→SELL，計算 RRR，標記最佳/最差 ──
        pairs = []
        buy_entry = None
        for _, row in bt_trades_df.iterrows():
            if row['type'] == 'BUY' and buy_entry is None:
                buy_entry = row
            elif row['type'] == 'SELL' and buy_entry is not None:
                ret = row['price'] / buy_entry['price'] - 1.0
                pairs.append({'buy_dt': buy_entry['datetime'], 'buy_price': buy_entry['price'],
                              'sell_dt': row['datetime'], 'sell_price': row['price'], 'return': ret})
                buy_entry = None

        if len(pairs) >= 1:
            best = max(pairs, key=lambda p: p['return'])
            worst = min(pairs, key=lambda p: p['return'])
            highlights = [(best, 'Best RRR', 'gold')]
            if best is not worst:
                highlights.append((worst, 'Worst RRR', '#FF1744'))

            for pair, label, clr in highlights:
                bds, _, _ = _nearest_candle(pair['buy_dt'])
                sds, _, _ = _nearest_candle(pair['sell_dt'])
                fig.add_trace(plotly_go.Scatter(
                    x=[bds, sds], y=[pair['buy_price'], pair['sell_price']],
                    mode='markers+lines+text',
                    marker=dict(symbol='star', color=clr, size=16, line=dict(color='black', width=1.5)),
                    line=dict(color=clr, width=2, dash='dot'),
                    text=[f"BUY {pair['buy_price']:,.1f}", f"SELL {pair['sell_price']:,.1f}"],
                    textposition=['bottom center', 'top center'],
                    textfont=dict(color=clr, size=9),
                    customdata=[[label, f"BUY @ {pair['buy_price']:,.2f}", f"{pair['return']:.2%}"],
                                [label, f"SELL @ {pair['sell_price']:,.2f}", f"{pair['return']:.2%}"]],
                    hovertemplate='<b>%{customdata[0]}</b><br>%{customdata[1]}<br>Return: %{customdata[2]}<extra></extra>',
                    name=label,
                ))

    fig.update_layout(height=550, margin=dict(l=0, r=0, t=30, b=0), xaxis_rangeslider_visible=False, xaxis_type="category", title="SMC Price Action")
    fig.update_xaxes(type="category", nticks=10)
    st.plotly_chart(fig, use_container_width=True)


def _render_log_html(log_messages):
    """共用的 log HTML 渲染函式。"""
    display_text = "\n".join(log_messages)
    return f"""
    <div style="background-color: #F8F9FA; color: #1A1A2E; padding: 12px 16px; border-radius: 8px; font-family: 'SF Mono', Consolas, monospace; font-size: 13px; height: 280px; display: flex; flex-direction: column-reverse; overflow-y: auto; border: 1px solid #E0E0E0;">
        <div style="white-space: pre-wrap;">{display_text}</div>
    </div>
    """


def main():
    # ── Header ──
    st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
        html, body, [class*="css"] { font-family: 'Inter', sans-serif !important; }
    </style>
    <div style="padding: 1.2rem 0 0.6rem 0; border-bottom: 2px solid #E0E0E0; margin-bottom: 1.5rem; text-align: center;">
        <h1 style="margin: 0; font-size: 3.4rem; font-weight: 700; color: #1A1A2E; font-family: 'Inter', sans-serif;">
            SMC × DRL Trading Platform
        </h1>
        <p style="margin: 0.3rem 0 0 0; font-size: 0.95rem; color: #666; font-family: 'Inter', sans-serif;">
            Smart Money Concepts × Deep Reinforcement Learning — Multi-Timeframe Analysis & Strategy
        </p>
    </div>
    """, unsafe_allow_html=True)

    # 讓按鈕與輸入欄位垂直對齊
    st.markdown("""
    <style>
    div[data-testid="stHorizontalBlock"] > div:nth-child(4) button,
    div[data-testid="stHorizontalBlock"] > div:nth-child(5) button {
        margin-top: 1.65rem;
    }
    </style>
    """, unsafe_allow_html=True)

    # 計算今天與一年前的日期
    today = datetime.date.today()
    try:
        one_year_ago = today.replace(year=today.year - 1)
    except ValueError:
        # 處理 2/29 閏年問題
        one_year_ago = today.replace(year=today.year - 1, day=28)

    st.info(f"💡 Hint: To ensure data accuracy, the current search range is limited to the past year. ({one_year_ago.strftime('%Y/%m/%d')} ~ {today.strftime('%Y/%m/%d')})")

    col_input1, col_input2, col_input3, col_btn1, col_btn2 = st.columns([2.5, 1.5, 1.5, 0.7, 1.0])
    with col_input1:
        ticker = st.text_input("Ticker (e.g. AAPL, 2330.TW)", value="")
    with col_input2:
        default_start = st.session_state.get("start_date_input", one_year_ago)
        start_date = st.date_input("Start Date", value=default_start, min_value=one_year_ago, max_value=today)
    with col_input3:
        default_end = st.session_state.get("end_date_input", today)
        end_date = st.date_input("End Date", value=max(default_end, start_date), min_value=start_date, max_value=today)
    with col_btn1:
        start_btn = st.button("Fetch & Analyze")
    with col_btn2:
        reset_btn = st.button("Reset / Clear")

    if reset_btn:
        st.session_state.clear()
        st.rerun()

    if "fetch_warning" in st.session_state:
        st.warning(st.session_state.pop("fetch_warning"))

    st.divider()

    # ── 建立左右兩欄排版 ──
    main_col_left, main_col_right = st.columns([5.5, 4.5])

    # ── 左欄：SMC 圖表與其相關控制 ──
    with main_col_left:
        st.subheader("SMC Stock Chart")
        render_chart()

        st.divider()

    # ── 右欄：訓練與分析區塊 (所有控制、報表、Log) ──
    with main_col_right:
        title_col, btn_col = st.columns([5.5, 4.5])
        with title_col:
            st.subheader("DRL × SMC Analysis")
        
        with btn_col:
            # 訓練按鈕直接放在右側頂部，與標題同一列
            train_btn_placeholder = st.empty()
        
        # 可滾動的報表與 Log 區塊
        with st.container(height=800):
            report_placeholder = st.empty()
            
            st.divider()
            
            log_container = st.container()
            log_container.markdown("#### DQN Training Log")
            log_placeholder = log_container.empty()

    if start_btn:
        if not ticker:
            st.warning("Please enter a ticker symbol")
            return
        st.session_state.pop("model_ret", None)
        with st.spinner(f"Fetching 1H data from {start_date} to {end_date}..."):
            raw_df = load_data_raw(ticker, start_date, end_date)
            if raw_df is not None and not raw_df.empty:
                # 自動修正起始日期 (若歷史資料不足)
                actual_start = raw_df['date'].min().date()
                if actual_start > start_date:
                    st.session_state["start_date_input"] = actual_start
                    st.session_state["fetch_warning"] = f"⚠️ 自動修正：此標的歷史 1H 資料最早僅能追溯至 {actual_start.strftime('%Y/%m/%d')}，已為您調整起始日期。"
                    start_date = actual_start
                else:
                    st.session_state["start_date_input"] = start_date

                st.session_state["end_date_input"] = end_date
                st.session_state["raw_df"] = raw_df
                st.session_state["ticker"] = ticker
                st.session_state["start_date"] = str(start_date)
                st.session_state["end_date"] = str(end_date)
                st.rerun()
            else:
                st.error("Failed to fetch stock data, please check ticker or date range.")
                return

    if "raw_df" not in st.session_state:
        log_placeholder.info("Waiting for training...")
        report_placeholder.info("Waiting for model training...")
        return

    raw_df = st.session_state["raw_df"]
    ticker = st.session_state.get("ticker", "UNKNOWN")

    # ── Training ──
    # 將訓練按鈕繪製
    train_btn = train_btn_placeholder.button(f"🚀 DQN + SMC + MTF + RRR ({ticker})", use_container_width=True)

    if train_btn:
        log_status = log_container.empty()
        log_area = log_container.empty()
        log_status.info("Starting MTF DQN+SMC training...")

        st.session_state["train_log"] = []
        def update_log(msg):
            st.session_state["train_log"].append(msg)
            log_area.markdown(_render_log_html(st.session_state["train_log"]), unsafe_allow_html=True)

        try:
            train_cfg = Config()
            train_cfg.ticker = ticker
            train_cfg.start_date = st.session_state.get("start_date", cfg.start_date)
            train_cfg.end_date = st.session_state.get("end_date", cfg.end_date)
            ret = run_training_pipeline(train_cfg, progress_callback=update_log)
            st.session_state["model_ret"] = ret
            metrics = ret["metrics"]
            final_msg = f"Training Completed! Total Return: {metrics.get('total_return', 0)*100:.1f}% | Sharpe: {metrics.get('sharpe_ratio', 0):.2f}"
            update_log(final_msg)
            log_status.success("Training saved successfully!")
            st.rerun()
        except Exception as e:
            log_status.error(f"Error during training: {e}")
            import traceback
            update_log(traceback.format_exc())
            return

    # ── 恢復已保存的訓練 Log ──
    saved_log = st.session_state.get("train_log", [])
    if saved_log and not train_btn:
        log_area = log_container.empty()
        log_area.markdown(_render_log_html(saved_log), unsafe_allow_html=True)

    # ── Report ──
    ret = st.session_state.get("model_ret", {})
    if not ret and not train_btn:
        if not saved_log:
            log_placeholder.info("Waiting for new training...")
        report_placeholder.info("Waiting for model training...")
        return

    if not ret:
        return

    report_placeholder.info("Running strategy inference...")
    try:
        agent = ret["agent"]
        mtf_df = ret["mtf_df"]
        feature_mean = ret["feature_mean"]
        feature_std = ret["feature_std"]
        metrics = ret["metrics"]

        recommendation = recommend_strategy(
            agent=agent,
            latest_mtf_raw=mtf_df,
            cfg=cfg,
            feature_cols=FEATURE_COLUMNS,
            feature_mean=feature_mean,
            feature_std=feature_std,
        )

        rr = recommendation["risk_reward_plan"]
        snap = recommendation["mtf_snapshot"]

        # 清除 placeholder，重新渲染報表
        report_placeholder.empty()
        
        with report_placeholder.container():
            st.markdown("### 🎯 核心交易結論 (Action Plan)")
            m1, m2 = st.columns(2)
            m1.metric(label="✅ DRL 推薦動作", value=recommendation['best_action_name'], delta=recommendation['trade_direction'], delta_color="off")
            m2.metric(label="💼 建議倉位", value=f"{recommendation['target_position_ratio']:.0%}")
            
            m3, m4 = st.columns(2)
            if rr.get("risk_reward_valid"):
                m3.metric(label="⚖️ 預期盈虧比 (RR Ratio)", value=f"{rr['risk_reward_ratio']:.2f}")
            else:
                m3.metric(label="⚖️ 預期盈虧比 (RR Ratio)", value="N/A")
            m4.metric(label="💲 回測獲利", value=f"{metrics.get('total_return', 0)*100:.1f}%", delta=f"Sharpe: {metrics.get('sharpe_ratio', 0):.2f}")

            st.markdown("---")
            st.markdown("#### 📊 進階分析細節 (Detailed Breakdown)")
            
            with st.expander("🛡️ 風險控管水準 (Prices)", expanded=True):
                if rr.get("risk_reward_valid"):
                    st.markdown(f"""
                    * **當前價格 (Close)**: `{recommendation['latest_close']:,.2f}`
                    * **進場參考 (Entry)**: `{rr['entry_price']:,.2f}`
                    * **停損水位 (Stop Loss)**: `{rr['stop_loss_price']:,.2f}`
                    * **停利水位 (Take Profit)**: `{rr['take_profit_price']:,.2f}`
                    * **停利依據**: {rr.get('take_profit_basis', '')}
                    """)
                else:
                    st.markdown("*當前無有效的進出場風控計畫*")
            
            with st.expander("🧭 多時區 SMC 共識 (MTF Bias) & 📈 回測細節", expanded=True):
                st.markdown(f"""
                **MTF 偏誤**: W1 ({snap['w1_smc_bias']:.0f}) | D1 ({snap['d1_smc_bias']:.0f}) | H4 ({snap['h4_smc_bias']:.0f}) | H1 ({snap['h1_smc_bias']:.0f})
                * **共識分數**: `{snap['mtf_confluence_score']:.1f}` | **方向衝突**: `{'是 (Yes)' if snap['mtf_conflict'] else '否 (No)'}`
                
                **模型 Test Set Metrics**:
                * 累積報酬: `{metrics.get('total_return', 0)*100:.1f}%`
                * 最大回撤: `{metrics.get('max_drawdown', 0)*100:.1f}%`
                * 獲利因子: `{metrics.get('profit_factor', 0):.2f}`
                """)

    except Exception as e:
        report_placeholder.error(f"Inference failed: {e}")

if __name__ == '__main__':
    main()
