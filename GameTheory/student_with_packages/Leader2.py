"""
A simple leader for the Stackelberg game.
"""
from base_leader import Leader
import random
from sklearn.linear_model import LinearRegression
import numpy as np

___author___ = "Rokas"


# Leader for follower 2, Profit of 18.83005(5d.p)

class SimpleLeader(Leader):
    def __init__(self, name):
        # If you want to initialize something here, do it before the super() call.
        self.leader_prices = []
        self.follower_prices = []
        self.unit_cost = 1
        self.profits = []
        super().__init__(name)

    def calculate_daily_profit(self, leader_price):
        follower_price = np.exp(self.coefficient * leader_price + self.intercept)  # Use exponential for the prediction
        demand = 2 - leader_price + 0.3 * follower_price
        profit = (leader_price - self.unit_cost) * demand
        return profit

    def new_price(self, date: int) -> float:
        self.log(date)
        if date == 101:
            best_price = self.leader_strategy()
            return float(best_price)
        elif date > 101:
            self.coefficient, self.intercept = self.analyse_history(days=date-1)
            best_price = self.leader_strategy()
            return float(best_price)

    def analyse_history(self, days=100, window_size=100):
        self.leader_prices = []
        self.follower_prices = []
        for day in range(1, days + 1):
            if day == 4:
                continue
            leader_price, follower_price = self.get_price_from_date(day)
            self.leader_prices.append(leader_price)
            self.follower_prices.append(np.log(follower_price))  # Log transform the follower prices

        # Lists to store the coefficients and intercepts from each window
        rolling_coefficients = []
        rolling_intercepts = []

        for start in range(days - window_size + 1):
            end = start + window_size
            X = np.array(self.leader_prices[start:end]).reshape(-1, 1)
            y = np.array(self.follower_prices[start:end])

            model = LinearRegression()
            model.fit(X, y)

            rolling_coefficients.append(model.coef_[0])
            rolling_intercepts.append(model.intercept_)

        decay_factor = 2.00 # Weight decay factor
        indices = np.arange(len(rolling_coefficients))
        weights = decay_factor ** indices
        weights = weights[::-1]  # Reverse to give more weight to recent observations

        weighted_average_coefficients = np.average(rolling_coefficients, weights=weights)
        weighted_average_intercepts = np.average(rolling_intercepts, weights=weights)

        print(f"Weighted average coefficient: {weighted_average_coefficients}")
        print(f"Weighted average intercept: {weighted_average_intercepts}")

        return weighted_average_coefficients, weighted_average_intercepts

    def calculate_profit_derivative(self, leader_price):
        h = 0.01
        profit = self.calculate_daily_profit(leader_price)
        profit_h = self.calculate_daily_profit(leader_price + h)
        return (profit_h - profit) / h
    
    def leader_strategy(self):
        current_price = max(1.01, self.unit_cost)
        tolerance = 0.001
        max_iterations = 1000

        for iteration in range(max_iterations):
            profit_derivative = self.calculate_profit_derivative(current_price)
            h = 0.01
            profit_derivative_plus = self.calculate_profit_derivative(current_price + h)
            profit_derivative_minus = self.calculate_profit_derivative(current_price - h)
            profit_second_derivative = (profit_derivative_plus - profit_derivative_minus) / (2 * h)

            if abs(profit_derivative) < tolerance:
                break
            if profit_second_derivative == 0:
                break
            step_size = profit_derivative / profit_second_derivative
            current_price -= step_size
            current_price = max(current_price, self.unit_cost)

            print(f"Iteration {iteration}: Price = {current_price}, Derivative = {profit_derivative}, Step Size = {step_size}")

        return current_price

    def start_simulation(self):
        self.log("Start of simulation")
        self.coefficient, self.intercept = self.analyse_history()

    def end_simulation(self):
        total_profit = 0
        all_leader_prices = []
        all_follower_prices = []
        for day in range(101, 131):
            leader_price, follower_price = self.get_price_from_date(day)
            all_leader_prices.append(leader_price)
            all_follower_prices.append(follower_price)

        for i in range(len(all_leader_prices)):
            leader_price = all_leader_prices[i]
            daily_profit = self.calculate_daily_profit(leader_price)
            total_profit += daily_profit

        self.log(f"End of simulation. Total profit: £{total_profit:.5f}")
        return total_profit

if __name__ == '__main__':
    SimpleLeader('17')