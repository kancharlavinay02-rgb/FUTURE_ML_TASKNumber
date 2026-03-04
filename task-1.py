import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# --- 1. Generate Dummy Data ---
# In a real project, you would load this: data = pd.read_csv('sales_data.csv')
np.random.seed(42)
months = np.arange(1, 25).reshape(-1, 1)  # 24 months of data
sales = 100 + (months * 10).flatten() + np.random.normal(0, 15, 24) # Upward trend with noise

# Create a DataFrame
data = pd.DataFrame({'Month': months.flatten(), 'Sales': sales})
print("Sample Data (First 5 Rows):")
print(data.head())

# --- 2. Data Splitting ---
# We use past months to train, and we'll test on the last few months to see if it works
X = data[['Month']]
y = data['Sales']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- 3. Train the Model ---
model = LinearRegression()
model.fit(X_train, y_train)

# --- 4. Evaluate ---
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"\nModel Performance:")
print(f"Mean Squared Error: {mse:.2f}")
print(f"R^2 Score: {r2:.2f} (Closer to 1.0 is better)")

# --- 5. Forecast Future Demand (The "Quiet" Way) ---
# Create a DataFrame with the same column name used in training
future_months = pd.DataFrame({'Month': [47, 48, 49]})

future_sales = model.predict(future_months)

print("\nFuture Predictions:")
for month, sale in zip(future_months['Month'], future_sales):
    print(f"Month {month}: Predicted Sales = {sale:.2f}")

# --- 6. Visualization ---
plt.figure(figsize=(10, 5))
plt.scatter(X, y, color='blue', label='Actual Sales')
plt.plot(X, model.predict(X), color='red', linewidth=2, label='Trend Line')
plt.scatter(future_months, future_sales, color='green', marker='x', s=100, label='Future Forecast')
plt.xlabel('Month')
plt.ylabel('Sales Units')
plt.title('Sales Forecast (Linear Regression)')
plt.legend()
plt.grid(True)
plt.show()