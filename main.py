import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_squared_error


# Load data
file_path = 'BTC-2020min.csv'
data = pd.read_csv(file_path)
data.columns = data.columns.str.strip()

# Preprocessing
data['date'] = pd.to_datetime(data['date'])
data.set_index('date', inplace=True)
data = data[['close', 'Volume BTC', 'Volume USD']].dropna()
data['Return'] = data['close'].pct_change()
data.dropna(inplace=True)

# Create lag features
window = 30
for i in range(1, window + 1):
    data[f'Lag_Return_{i}'] = data['Return'].shift(i)
    data[f'Lag_VolumeBTC_{i}'] = data['Volume BTC'].shift(i)
    data[f'Lag_VolumeUSD_{i}'] = data['Volume USD'].shift(i)
data.dropna(inplace=True)

# Prepare features and target
feature_cols = [f'Lag_Return_{i}' for i in range(1, window + 1)] + \
               [f'Lag_VolumeBTC_{i}' for i in range(1, window + 1)] + \
               [f'Lag_VolumeUSD_{i}' for i in range(1, window + 1)]
X = data[feature_cols]
y = np.where(data['Return'] >= 0, 1, -1)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, shuffle=False, test_size=0.2
)

# Fit model
model = RandomForestClassifier(
    n_estimators=50,
    max_depth=10,
    random_state=42,
    n_jobs=-1
)
model.fit(X_train, y_train)

# Predict directions
predicted_direction = model.predict(X_test)
accuracy = np.mean(predicted_direction == y_test)
print(f"\nDirection Accuracy: {accuracy * 100:.2f}%")

# Clamp extreme returns to avoid runaway values in simulation
returns_test = data['Return'].iloc[len(X_train):].values
test_prices = data.iloc[len(X_train):]['close'].values
max_r = 0.1
safe_returns = np.clip(returns_test, -max_r, max_r)

# Build predicted price series using sign-flip compounding
predicted_prices = [test_prices[0]]
for pred_dir, real_r in zip(predicted_direction, safe_returns):
    step_return = real_r if pred_dir == 1 else -real_r
    predicted_prices.append(predicted_prices[-1] * (1 + step_return))
predicted_prices = predicted_prices[1:]

# Calculate RMSE for the same period
rmse = np.sqrt(mean_squared_error(test_prices, predicted_prices))
print(f"RMSE: {rmse:.4f}")

# Optional: Smooth the predicted price for better visualization
predicted_prices = pd.Series(predicted_prices, index=data.iloc[len(X_train):].index).rolling(5).mean()

# Plot actual vs predicted prices
plt.figure(figsize=(10, 5))
plt.plot(data.iloc[len(X_train):].index, test_prices, label='Actual Price', color='blue', linewidth=2)
plt.plot(data.iloc[len(X_train):].index, predicted_prices, label='Predicted Price', color='red', linewidth=1.8)
plt.title(f"Bitcoin: Actual vs Predicted Price\n(Direction accuracy: {accuracy*100:.2f}% | RMSE: {rmse:.2f})", fontsize=13)
plt.xlabel("Date", fontsize=11)
plt.ylabel("Closing Price (USD)", fontsize=11)
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()
