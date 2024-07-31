# Scripts for Exploratory Data Analysis

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

# Load data from a specific sheet
file_path = 'data.xlsx'
sheet_name = 'Follower_Mk1'  
data = pd.read_excel(file_path, sheet_name=sheet_name)

# Display the first few rows to confirm correct data loading
#print(data.head())

#print(data.describe())

plt.figure(figsize=(10, 5))
plt.plot(data['Date'], data["Leader's Price"], label='Leader Price')
plt.plot(data['Date'], data["Follower's Price"], label='Follower Price')
plt.title('Price Trends Over Time for Follower 1')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.show()

correlation = data[["Leader's Price", "Follower's Price"]].corr()
print("Correlation Matrix:\n", correlation)

# Linear Regression
lin_reg = LinearRegression()
lin_reg.fit(data[["Leader's Price"]], data["Follower's Price"])

# Polynomial Regression (degree 2)
poly_features = PolynomialFeatures(degree=2)
X_poly = poly_features.fit_transform(data[["Leader's Price"]])
poly_reg = LinearRegression()
poly_reg.fit(X_poly, data["Follower's Price"])

# Plotting the regression
X_fit = np.linspace(data["Leader's Price"].min(), data["Leader's Price"].max(), 100).reshape(-1, 1)
plt.scatter(data["Leader's Price"], data["Follower's Price"], color='lightblue')
plt.plot(X_fit, lin_reg.predict(X_fit), label='Linear Fit', color='red')
plt.plot(X_fit, poly_reg.predict(poly_features.transform(X_fit)), label='Polynomial Fit', color='green')
plt.title('Regression Analysis for Follower 1')
plt.xlabel("Leader's Price")
plt.ylabel("Follower's Price")
plt.legend()
plt.show()

data['Leader Price Change'] = data["Leader's Price"].diff()
data['Follower Price Change'] = data["Follower's Price"].diff()

plt.figure(figsize=(10, 5))
plt.plot(data['Date'], data['Leader Price Change'], label="Leader's Price Change")
plt.plot(data['Date'], data['Follower Price Change'], label="Follower's Price Change")
plt.title('Daily Price Changes for Follower 1')
plt.xlabel('Date')
plt.ylabel('Price Change')
plt.legend()
plt.show()

print("Descriptive Statistics for Price Changes:")
print(data[['Leader Price Change', 'Follower Price Change']].describe())