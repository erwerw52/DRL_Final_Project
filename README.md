# DRL Trading with Smart Money Concepts (SMC)

此專案結合了**深度強化學習 (Deep Reinforcement Learning, DRL)** 與 **聰明錢概念 (Smart Money Concepts, SMC)**，建立一個擁有圖形化互動介面與即時訓練進度追蹤的自動化交易輔助系統。系統透過分析市場的微觀結構特徵，輔助 DQN 代理人學習最佳交易策略（買入、賣出、持有）。

## ✨ 核心特色 (Features)

* **SMC 交易特徵分析**：
  使用 `smartmoneyconcepts` 函式庫解析市場盤面，將傳統 K 線轉化為專業交易員使用的 SMC 指標：
  * **流動性池 (Liquidity Pools / Swept)**
  * **前期高低點 (Old Highs / Old Lows)**
  * **合理價值缺口 (Fair Value Gaps, FVG)**
  * **訂單塊 (Order Blocks, OB)**
  * 溢價區/折價區 (Premium / Discount Zones)
* **DRL DQN 決策代理人**：
  * 觀測狀態維度 (State Dimension) 擴充至 12 維，讓模型能吸收盤面的價格位置與 SMC 的各種距離及強弱標記。
  * 動態分割預處理資料，並經過標準化的環境獎勵機制反覆迭代學習。
  * 具備完整的回測 (Backtesting) 評估計算，呈現真實的勝率與夏普值 (Sharpe Ratio)。
* **高互動 Plotly 繪圖分析戰情室**：
  * 無需繁瑣開關設定，一鍵渲染帶有各式 SMC 分區與邊界的互動式 K 線圖。
  * 透過點擊圖例 (Legend) 可以動態隱藏/顯示特定的 FVG 或 OB 區塊。
* **即時終端機等級的訓練日誌 UI**：
  * 具備自動下捲 (Auto-scroll到最底) 功能的網頁版擬真 Console，即時觀看 Reward 與 Loss。
  * 所有呈現的資金、損失格式全面支援千分位金額標示 (例如 `33,505.17`)，提升閱讀性。

## 📁 專案架構 (Project Structure)

```text
DRL_Final_Project/
├── app.py              # Streamlit 網頁主程式（SMC儀表板、互動繪圖、訓練追蹤）
├── train.py            # DRL 模型訓練邏輯與資料切分
├── config.py           # 系統與超參數配置設定檔
├── requirements.txt    # 依賴套件列表
├── agent/
│   ├── __init__.py
│   └── dqn_agent.py    # DQN 模型與代理人實作
├── env/
│   ├── __init__.py
│   └── trading_env.py  # 支援 SMC 觀測狀態的 Gym-like 交易環境
├── model/
│   ├── __init__.py
│   └── network.py      # 神經網路架構
└── utils/
    ├── __init__.py
    ├── data_utils.py   # 資料抓取與 SMC 前處理 (結合 yfinance 與 pandas)
    ├── metrics.py      # 回測成效與夏普值計算
    └── replay_buffer.py# 經驗回放池
```

## 🚀 快速開始 (Quick Start)

### 1. 安裝環境與依賴套件

請使用虛擬環境 (Virtual Environment) 進行安裝，避免套件衝突：

```bash
# 建立並啟動虛擬環境 (Windows 範例)
python -m venv venv
.\venv\Scripts\activate

# 安裝所需套件
pip install -r requirements.txt

# 安裝 SMC 指標分析庫
pip install smartmoneyconcepts
```

### 2. streamlit demo site
[https://drlfinalproject-whc4jiboyzx96fcyz9qmbt.streamlit.app/]

### 3. 操作介面

1. 於瀏覽器自動開啟的網頁中，輸入目標**股票代號**（例如 `AAPL`、`2330.TW`、或加密貨幣 `BTC-USD`）。
2. 選定您要擷取的歷史時間區間。
3. 點擊 **開始 SMC 分析**。
4. 網頁會自動繪製高互動性的 SMC 盤面結構，同時下方會即時串流顯示 DQN 模型最新的學習狀況（Episode, Reward, Loss）。
5. 訓練完成後，右側面板會自動生成基於當前最新一筆歷史 K 線特徵的**建議動作 (買入/持有/賣出)** 以及**模型綜合評估報告 (勝率、夏普值)**。
