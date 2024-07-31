"""
A simple leader for the Stackelberg game using polynomial regression.
"""
from base_leader import Leader
import random
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import numpy as np

___author___ = "Rokas"

class SimpleLeader(Leader):
    def __init__(self, name):
        self.leader_prices = [] 
        self.follower_prices = []
        self.unit_cost = 1
        super().__init__(name)

    def calculate_daily_profit(self, leader_price):
        follower_price = self.coefficient * leader_price**2 + self.linear_coeff * leader_price + self.intercept
        demand = 2 - leader_price + 0.3 * follower_price
        profit = (leader_price - self.unit_cost) * demand
        return profit  

    def new_price(self, date: int) -> float:
        self.log(date)
        if date == 101:
            best_price = self.leader_strategy()
            best_price = float(best_price)
            return best_price
        elif date > 101:
            self.coefficient, self.linear_coeff, self.intercept = self.analyse_history(days=date-1)
            best_price = self.leader_strategy()
            best_price = float(best_price)
            return best_price

    def analyse_history(self, days=100):
        self.leader_prices = [] 
        self.follower_prices = []
        for day in range(1, days + 1):  
            if day == 4:
                continue
            leader_price, follower_price = self.get_price_from_date(day)
            self.leader_prices.append(leader_price)
            self.follower_prices.append(follower_price)

        X = np.array(self.leader_prices).reshape(-1, 1)
        y = np.array(self.follower_prices)

        # Generate polynomial features
        poly = PolynomialFeatures(degree=2)  # Adjust degree here as needed
        X_poly = poly.fit_transform(X)

        model = LinearRegression()
        model.fit(X_poly, y)

        # Extract the coefficients for the polynomial regression
        reaction_function_coefficient = model.coef_[2]  # Coefficient for x^2
        linear_coefficient = model.coef_[1]             # Coefficient for x
        reaction_function_intercept = model.intercept_

        print(f"Reaction Function Coefficient: {reaction_function_coefficient}")
        print(f"Linear Coefficient: {linear_coefficient}")
        print(f"Reaction Function Intercept: {reaction_function_intercept}")

        return reaction_function_coefficient, linear_coefficient, reaction_function_intercept

    def calculate_profit_derivative(self, leader_price):
        """ Numerically estimate the derivative of the profit function. """
        h = 0.01  # Small step for the derivative estimation
        profit = self.calculate_daily_profit(leader_price)
        profit_h = self.calculate_daily_profit(leader_price + h)
        return (profit_h - profit) / h

    def leader_strategy(self):
        current_price = max(1.01, self.unit_cost)  # Starting point slightly above unit cost
        tolerance = 0.001  # Tighter tolerance for stopping the iteration
        max_iterations = 1000  # Limit iterations to prevent infinite loops

        for iteration in range(max_iterations):
            profit_derivative = self.calculate_profit_derivative(current_price)
            # To calculate the second derivative more reliably, use central difference
            h = 0.01
            profit_derivative_plus = self.calculate_profit_derivative(current_price + h)
            profit_derivative_minus = self.calculate_profit_derivative(current_price - h)
            profit_second_derivative = (profit_derivative_plus - profit_derivative_minus) / (2 * h)

            if abs(profit_derivative) < tolerance:  # Check if derivative is close enough to zero
                break

            # Update current price based on Newton's method formula
            if profit_second_derivative == 0:
                break  # Avoid division by zero, indicating a possible issue with second derivative calculation
            step_size = profit_derivative / profit_second_derivative
            current_price -= step_size
            current_price = max(current_price, self.unit_cost)  # Ensure the price is never below the unit cost

            print(f"Iteration {iteration}: Price = {current_price}, Derivative = {profit_derivative}, Step Size = {step_size}")

        return current_price



    def start_simulation(self):
        self.log("Start of simulation")
        self.coefficient, self.linear_coeff, self.intercept = self.analyse_history()

    def end_simulation(self):
        total_profit = 0
        unit_cost = 1
        all_leader_prices = [] 
        all_follower_prices = []
        for day in range(101, 131):
            leader_price, follower_price = self.get_price_from_date(day)
            all_leader_prices.append(leader_price)
            all_follower_prices.append(follower_price)

        for i in range(len(all_leader_prices)):
            leader_price = all_leader_prices[i]
            follower_price = all_follower_prices[i]
            demand = 2 - leader_price + 0.3 * follower_price
            daily_profit = (leader_price - unit_cost) * demand
            total_profit += daily_profit

        self.log(f"End of simulation. Total profit: £{total_profit:.5f}")
        return total_profit

if __name__ == '__main__':
    SimpleLeader('17Polynomial2')
