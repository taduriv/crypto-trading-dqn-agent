import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

class TradingTestEnv:
    def __init__(self, data, window_size=10, transaction_fee=0.001):
        self.data = data.values
        self.window_size = window_size
        self.transaction_fee = transaction_fee
        self.initial_balance = 10000

    def reset(self):
        self.current_step = 0
        self.balance = self.initial_balance
        self.holdings = 0
        self.last_buy_price = None
        self.portfolio = []
        self.actions = []
        return np.expand_dims(self.data[self.current_step:self.current_step + self.window_size], axis=0)

    def step(self, action):
        price = self.data[self.current_step, 3]  # close price
        fee = price * self.transaction_fee

        if action == 0:  # Buy
            if self.balance >= price + fee:
                self.holdings += 1
                self.balance -= price + fee
                self.last_buy_price = price
                self.actions.append((self.current_step, 'Buy', price))
        elif action == 1 and self.holdings > 0:  # Sell
            self.balance += price - fee
            self.holdings -= 1
            self.actions.append((self.current_step, 'Sell', price))
            self.last_buy_price = None
        else:
            self.actions.append((self.current_step, 'Hold', price))

        self.portfolio.append(self.balance + self.holdings * price)
        self.current_step += 1
        done = self.current_step + self.window_size >= len(self.data)

        if done:
            next_state = np.zeros((1, self.window_size, 5))
        else:
            next_state = np.expand_dims(self.data[self.current_step:self.current_step + self.window_size], axis=0)
        return next_state, done

def test_best_model(data, model_path):
    env = TradingTestEnv(data)
    model = load_model(model_path)
    state = env.reset()
    done = False

    while not done:
        q_values = model.predict(state, verbose=0)[0]
        action = np.argmax(q_values)
        state, done = env.step(action)

    final_price = env.data[env.current_step - 1, 3]
    final_balance = env.balance + env.holdings * final_price
    profit = final_balance - env.initial_balance

    print(f"\nTEST SUMMARY")
    print(f"Final Balance: ${final_balance:.2f}")
    print(f"Profit: ${profit:.2f}")
    print(f"Total Actions: {len(env.actions)}")

    # Visualization
    buy_x, buy_y = zip(*[(s, p) for s, a, p in env.actions if a == 'Buy']) if any(a == 'Buy' for _, a, _ in env.actions) else ([], [])
    sell_x, sell_y = zip(*[(s, p) for s, a, p in env.actions if a == 'Sell']) if any(a == 'Sell' for _, a, _ in env.actions) else ([], [])
    close_prices = env.data[:len(env.portfolio), 3]

    plt.figure(figsize=(15,6))
    plt.plot(close_prices, label='Close Price')
    plt.plot(env.portfolio, label='Portfolio Value', linestyle='--')
    plt.scatter(buy_x, buy_y, marker='^', color='green', label='Buy')
    plt.scatter(sell_x, sell_y, marker='v', color='red', label='Sell')
    plt.title("DRL Agent Trading Simulation")
    plt.xlabel("Time Step")
    plt.ylabel("Price / Portfolio Value")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("trading_plot.png", dpi=300)
    plt.show()

# Load raw (non-normalized) OHLCV dataset and run
df = pd.read_csv("/home/taduriv/cryptobot/Simple/Bitfinex_ETHUSD_1h.csv")
df['date'] = pd.to_datetime(df['date'], format="%d-%m-%Y %H:%M", errors='coerce')
df = df.dropna(subset=['date']).sort_values('date').reset_index(drop=True)
df = df[(df['date'] >= "2018-05-15") & (df['date'] <= "2020-10-11")].reset_index(drop=True)
df = df[['open', 'high', 'low', 'close', 'Volume ETH']]

split = int(len(df) * 0.7)
test_data = df[split:]

test_best_model(test_data, "/home/taduriv/cryptobot/Simple/best_model_episode_5.keras")
