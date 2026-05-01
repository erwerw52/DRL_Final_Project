import pandas as pd
import numpy as np

np.random.seed(42)
dates = pd.date_range("2020-01-01", periods=1000, freq="B")
data = []

# 建立具有 15 天週期性波動（盤整震盪）且長期微幅向上的資料
# 這樣 Buy & Hold 會有微薄正報酬，但會承受巨大波動(Drawdown)
# 而 PD-RL 可以學習到在 20 天滾動區間的高低點「低買高賣」，創造驚人報酬

for i, d in enumerate(dates):
    # 長期趨勢：每天上升 0.05，1000天共上升 50
    trend = 100.0 + i * 0.05
    
    # 週期波動：15天一個完整循環，振幅 20
    # 由於 PD Array 窗口是 20 天，這剛好能捕捉到每個循環的高低點
    cycle = 20.0 * np.sin(i * 2 * np.pi / 15.0)
    
    base = trend + cycle
    
    # 加入少許雜訊
    noise = np.random.uniform(-1, 1)
    close_p = base + noise
    open_p = base - noise
    high_p = max(open_p, close_p) + 1.5
    low_p = min(open_p, close_p) - 1.5
    
    volume = int(1000000 + np.random.randint(0, 500000))
    data.append([
        d.strftime("%Y-%m-%d"), 
        round(open_p, 2), 
        round(high_p, 2), 
        round(low_p, 2), 
        round(close_p, 2), 
        volume
    ])

df = pd.DataFrame(data, columns=["date", "open", "high", "low", "close", "volume"])
df.to_csv("data/sample_data.csv", index=False)
print("Mean-reverting data with upward trend generated!")
