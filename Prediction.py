import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
df = pd.read_csv("/content/housing.csv")  # Update path if needed

# Handle missing values 
df.dropna(inplace=True)

categorical_cols = ['mainroad', 'guestroom', 'basement', 'hotwaterheating',
                    'airconditioning', 'prefarea', 'furnishingstatus']
df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

#  Split features and target
X = df.drop(columns=['price'])     # Features
y = df['price']                    # Target

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(" Evaluation Metrics:")
print(f"MAE: {mae:.2f}")
print(f"MSE: {mse:.2f}")
print(f"R² Score: {r2:.2f}")

# Plot predictions vs actual 
plt.figure(figsize=(8, 5))
plt.scatter(X_test['area'], y_test, label='Actual', color='blue')
plt.scatter(X_test['area'], y_pred, label='Predicted', color='red', alpha=0.6)
plt.xlabel("Area")
plt.ylabel("Price")
plt.title("Actual vs Predicted Price based on Area")
plt.legend()
plt.show()

# Interpret coefficients
coeff_df = pd.DataFrame({'Feature': X.columns, 'Coefficient': model.coef_})
print("\n Feature Coefficients:")
print(coeff_df.sort_values(by='Coefficient', ascending=False))
