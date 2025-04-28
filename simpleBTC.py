import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
from collections import deque
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Flatten, Input
from tensorflow.keras.optimizers import Adam
import os

# Model
def create_model(input_shape, action_space):
    model = Sequential([
        Input(shape=input_shape),
        Flatten(),
        Dense(64, activation='relu'),
        Dense(32, activation='relu'),
        Dense(8, activation='relu'),
        Dense(action_space, activation='linear')
    ])
    model.compile(loss='mse', optimizer=Adam(learning_rate=0.001))
    return model

# Environment
class TradingEnv:
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
        self.buy_count = 0
        self.hold_count = 0
        self.trade_log = []
        self.balance_history = [self.balance]
        return np.expand_dims(self.data[self.current_step:self.current_step + self.window_size], axis=0)

    def step(self, action):
        current_price = self.data[self.current_step, 3]
        reward = 0
        fee = current_price * self.transaction_fee
        action_state = "Hold"

        if action == 0 and self.balance >= current_price + fee:
            self.holdings += 1
            self.balance -= (current_price + fee)
            self.last_buy_price = current_price
            self.buy_count += 1
            self.hold_count = 0
            action_state = "Buy"
            if self.buy_count > 20:
                reward -= 2

        elif action == 1 and self.holdings > 0:
            self.balance += (current_price - fee)
            self.holdings -= 1
            self.hold_count = 0
            self.buy_count = 0
            action_state = "Sell"
            if self.last_buy_price:
                profit_pct = (current_price - self.last_buy_price - fee) / self.last_buy_price
                reward = profit_pct * 100
                self.last_buy_price = None

        elif action == 2:
            self.hold_count += 1
            if self.hold_count > 20:
                reward -= 1
            if self.holdings > 0 and self.last_buy_price:
                price_diff = current_price - self.last_buy_price
                reward += (price_diff / self.last_buy_price) * 2

        reward = float(np.clip(reward, -10, 10))
        portfolio_value = self.balance + self.holdings * current_price
        self.trade_log.append((self.current_step, action_state, current_price, fee, self.balance))
        self.balance_history.append(portfolio_value)

        self.current_step += 1
        done = self.current_step + self.window_size >= len(self.data)
        next_state = (
            np.expand_dims(self.data[self.current_step:self.current_step + self.window_size], axis=0)
            if not done else np.zeros((1, self.window_size, 5))
        )
        return next_state, reward, done

# Agent
class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.995
        self.epsilon = 1.0
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.94
        self.model = create_model((state_size, 5), action_size)
        self.best_profit = float('-inf')

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        q_values = self.model.predict(state, verbose=0)
        return np.argmax(q_values[0])

    def replay(self, batch_size=64):
        minibatch = random.sample(self.memory, min(len(self.memory), batch_size))
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target += self.gamma * np.amax(self.model.predict(next_state, verbose=0)[0])
            target_f = self.model.predict(state, verbose=0)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

# Training
def train_dqn(data, episodes=100):
    env = TradingEnv(data)
    agent = DQNAgent(env.window_size, 3)

    for episode in range(episodes):
        state = env.reset()
        done = False
        buys = sells = holds = 0

        while not done:
            action = agent.act(state)
            next_state, reward, done = env.step(action)
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            if action == 0: buys += 1
            elif action == 1: sells += 1
            else: holds += 1

        agent.replay()
        final_price = env.data[env.current_step - 1, 3]
        profit = env.balance + env.holdings * final_price - env.initial_balance
        print(f"Episode {episode+1}: Profit=${profit:.2f}, Buys={buys}, Sells={sells}, Holds={holds}")

        if profit > agent.best_profit:
            agent.best_profit = profit
            agent.model.save("best_model.keras")

    return agent

# Testing
def test_agent(data, model_path):
    env = TradingEnv(data)
    model = load_model(model_path)

    state = env.reset()
    done = False
    buys = sells = holds = 0

    while not done:
        q_values = model.predict(state, verbose=0)
        action = np.argmax(q_values[0])
        next_state, reward, done = env.step(action)
        state = next_state
        if action == 0: buys += 1
        elif action == 1: sells += 1
        else: holds += 1

    final_balance = env.balance + env.holdings * env.data[env.current_step - 1, 3]
    profit = final_balance - env.initial_balance
    print("\nTEST RESULTS:")
    print(f"Final Balance: ${final_balance:.2f}")
    print(f"Profit: ${profit:.2f}")
    print(f"Actions Taken - Buys: {buys}, Sells: {sells}, Holds: {holds}")

# Main execution
if __name__ == "__main__":
    # Load dataset
    df = pd.read_csv("/home/taduriv/cryptobot/Simple/Bitfinex_BTCUSD_1h880d.csv")
    df['date'] = pd.to_datetime(df['date'], format="%d-%m-%Y %H:%M", errors='coerce')
    df = df.dropna(subset=['date'])
    df = df.sort_values('date').reset_index(drop=True)

    # Show actual date coveragec
    print(" Date range in dataset:", df['date'].min(), "to", df['date'].max())

    # Filter date range (2018-05-15 to 2020-10-11)
    df = df[(df['date'] >= "2018-05-15") & (df['date'] <= "2020-10-11")].reset_index(drop=True)

    if df.empty:
        raise ValueError("Dataset is empty after filtering by date. Check data source or date format.")

    # Select numeric columns (adjust if needed)
    df = df[['open', 'high', 'low', 'close', 'Volume BTC']]
    norm_data = (df - df.min()) / (df.max() - df.min())

    # Train-test split (70/30)
    split_index = int(len(norm_data) * 0.7)
    train_data = norm_data[:split_index]
    test_data = norm_data[split_index:]

    # Train the agent
    train_dqn(train_data, episodes=100)

    # Test best model
    test_agent(test_data, "best_model.keras")
