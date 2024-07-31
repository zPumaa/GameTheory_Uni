"""
A simple leader for the Stackelberg game.
"""
from base_leader import Leader
import random
from sklearn.linear_model import LinearRegression
import numpy as np

___author___ = "Rokas"



class SimpleLeader(Leader):
    def __init__(self, name):
        # If you want to initialize something here, do it before the super() call.
        self.leader_prices = [] 
        self.follower_prices = []
        self.unit_cost = 1
        super().__init__(name)

    def calculate_daily_profit(self, leader_price):
        follower_price = self.coefficient * leader_price + self.intercept
        demand = 2 - leader_price + 0.3 * follower_price
        profit = (leader_price - self.unit_cost) * demand
        return profit  

    def new_price(self, date: int) -> float:
        """
        A function for setting the new price of each day.
        :param date: date of the day to be updated
        :return: (float) price for the day
        """
        self.log(date)

        # if date <= len(self.leader_prices):
        #     return self.leader_prices[date - 1]
        # else:
        #     # We need to calculate the best price for today
        if date == 101:
            best_price = self.leader_strategy()
            best_price = float(best_price)
            return best_price
        elif date > 101:
            # Retrieve and append the follower's price for the current day
            self.coefficient, self.intercept = self.analyse_history(days=date-1)
            best_price = self.leader_strategy()
            best_price = float(best_price)
            return best_price

    
    def analyse_history(self, days=100):
        """
        Simulate the prices for the first `days` days and store them in separate arrays,
        excluding Day 4 due to identified noise issues.
        """
        self.leader_prices = [] 
        self.follower_prices = []
        for day in range(1, days + 1):
            #if day == 4:
            #    continue  # Skip Day 4
            leader_price, follower_price = self.get_price_from_date(day)
            self.leader_prices.append(leader_price)
            self.follower_prices.append(follower_price)

        # Reshape the data to fit the model
        X = np.array(self.leader_prices).reshape(-1, 1)
        y = np.array(self.follower_prices)

        # Initialize the Linear Regression model
        model = LinearRegression()

        # Fit the model with historical data
        model.fit(X, y)

        # The model's coefficients are the estimation of the reaction function
        reaction_function_coefficient = model.coef_[0]
        reaction_function_intercept = model.intercept_

        # Display the estimated reaction function
        print(f"Reaction Function Coefficient: {reaction_function_coefficient}")
        print(f"Reaction Function Intercept: {reaction_function_intercept}")

        return reaction_function_coefficient, reaction_function_intercept
    
    def calculate_profit_derivative(self, leader_price):
        """ Numerically estimate the derivative of the profit function. """
        h = 0.01
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
        """
        A function run at the beginning of the simulation.
        """
        self.log("Start of simulation")
        self.coefficient, self.intercept = self.analyse_history()
        # best_price = self.leader_strategy(coefficient, intercept)

    def end_simulation(self):
        """
        A function run at the end of the simulation.
        It calculates the total profit based on the leader and follower prices.
        """
        total_profit = 0
        unit_cost = 1
        all_leader_prices = [] 
        all_follower_prices = []
        for day in range(101, 131):  
            leader_price, follower_price = self.get_price_from_date(day)
            all_leader_prices.append(leader_price)
            all_follower_prices.append(follower_price)

        print(len(all_leader_prices))
        for i in range(len(all_leader_prices)):
            leader_price = all_leader_prices[i]
            follower_price = all_follower_prices[i]
            demand = 2 - leader_price + 0.3 * follower_price
            daily_profit = (leader_price - unit_cost) * demand
            total_profit += daily_profit
        
        self.log(f"End of simulation. Total profit: £{total_profit:.5f}")
        return total_profit



if __name__ == '__main__':
    # Make sure you set this to your group number!
    SimpleLeader('17Linear')
