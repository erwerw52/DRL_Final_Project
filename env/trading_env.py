import numpy as np

class TradingEnv:
    def __init__(self, data, initial_balance, transaction_cost_rate, pd_bonus, reward_scale):
        self.df = data
        self.initial_balance = initial_balance
        self.transaction_cost_rate = transaction_cost_rate
        self.pd_bonus = pd_bonus
        self.reward_scale = reward_scale
        
        self.n_step = len(self.df)
        self.current_step = 0
        
        self.balance = self.initial_balance
        self.shares = 0
        self.net_worth = self.initial_balance
        self.highest_net_worth = self.initial_balance
        
        # State dimension: [pd_pos, is_premium, is_discount, open, high, low, close] (Total 7)
        self.state_dim = 7
        
    def reset(self):
        self.balance = self.initial_balance
        self.shares = 0
        self.net_worth = self.initial_balance
        self.highest_net_worth = self.initial_balance
        self.current_step = 0
        
        return self._get_state()
        
    def _get_state(self):
        if self.current_step >= self.n_step:
            self.current_step = self.n_step - 1
            
        row = self.df.iloc[self.current_step]
        
        # Normalize prices slightly for the neural network
        base_price = row['close'] if row['close'] > 0 else 1.0
        
        state = np.array([
            row['pd_pos'],
            row['is_premium'],
            row['is_discount'],
            row['open'] / base_price,
            row['high'] / base_price,
            row['low'] / base_price,
            row['close'] / base_price,
        ])
        
        return state
        
    def step(self, action):
        # action: 0=HOLD, 1=BUY, 2=SELL
        
        self.current_step += 1
        done = self.current_step >= self.n_step - 1
        
        if done:
            return self._get_state(), 0, done, {}
            
        current_price = self.df.iloc[self.current_step]['close']
        previous_net_worth = self.net_worth
        
        affordable_amount = 0
        sell_shares = 0
        
        # Execute Action
        if action == 1:  # BUY
            # Buy with 10% of balance
            buy_amount = self.balance * 0.1 
            cost = buy_amount * self.transaction_cost_rate
            affordable_amount = buy_amount - cost
            
            if affordable_amount > 0 and current_price > 0:
                buy_shares = affordable_amount / current_price
                self.balance -= buy_amount
                self.shares += buy_shares
                
        elif action == 2:  # SELL
            # Sell 10% of shares
            sell_shares = self.shares * 0.1
            if sell_shares > 0:
                revenue = sell_shares * current_price
                cost = revenue * self.transaction_cost_rate
                
                self.balance += (revenue - cost)
                self.shares -= sell_shares
                
        # Calculate new net worth
        self.net_worth = self.balance + (self.shares * current_price)
        self.highest_net_worth = max(self.highest_net_worth, self.net_worth)
        
        # Simple Reward 
        # 1. Delta Net Worth
        reward = (self.net_worth - previous_net_worth) * self.reward_scale
        
        # 2. PD Bonus (Encourage buying at discount and selling at premium)
        # 只有在「真的有發生有效交易」時，才給予這筆獎勵
        pd_pos = self.df.iloc[self.current_step]['pd_pos']
        
        if action == 1 and affordable_amount > 0 and pd_pos <= 0.5: # 成功在折扣區買入
            reward += self.pd_bonus * self.initial_balance * 0.0001
        elif action == 2 and sell_shares > 0 and pd_pos > 0.5: # 成功在溢價區賣出
            reward += self.pd_bonus * self.initial_balance * 0.0001
            
        return self._get_state(), reward, done, {}