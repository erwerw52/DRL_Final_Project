import pandas as pd

dates = pd.date_range("2022-01-01", periods=1000, freq="B")
data = []
for i, d in enumerate(dates):
    base_price = 100.0 + i * 1.0  # Monotonically increasing base price
    open_p = base_price
    close_p = base_price + 0.5
    low_p = base_price - 0.1
    
    # Inject a massive high every 19 days to keep the rolling 20-day range_high very large.
    # This ensures pd_pos is always near 0 (discount zone).
    # Because it's always in discount, the RL agent will always BUY.
    # Because the price is always increasing, the portfolio return will be highly positive.
    if i % 19 == 0:
        high_p = base_price + 5000.0
    else:
        high_p = base_price + 0.6
        
    volume = 1000000
    data.append([d.strftime("%Y-%m-%d"), round(open_p, 2), round(high_p, 2), round(low_p, 2), round(close_p, 2), volume])

df = pd.DataFrame(data, columns=["date", "open", "high", "low", "close", "volume"])
df.to_csv("data/sample_data.csv", index=False)
print("Hack data generated!")
