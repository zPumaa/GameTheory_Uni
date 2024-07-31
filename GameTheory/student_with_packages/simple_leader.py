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
        self.profits = []
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
        
        if date == 101:
            # best_price = self.set_new_price()
            best_price = self.leader_strategy()
            best_price = float(best_price)
            return best_price
        elif date > 101:
            # Retrieve and append the follower's price for the current day
            self.coefficient, self.intercept = self.analyse_history(days=date-1)
            best_price = self.leader_strategy()
            # best_price = self.set_new_price()
            best_price = float(best_price)
            return best_price

    
    def analyse_history(self, days=100, window_size=100):
        """
        Analyse the prices for the first `days` days and store them in separate arrays.
        """
        self.leader_prices = [] 
        self.follower_prices = []
        for day in range(1, days + 1):  
            leader_price, follower_price = self.get_price_from_date(day)
            self.leader_prices.append(leader_price)
            self.follower_prices.append(follower_price)
            demand = 2 - leader_price + 0.3 * follower_price
            daily_profit = (leader_price - self.unit_cost) * demand
            self.profits.append(daily_profit)
        # print(len(self.leader_prices), self.leader_prices[99])

        # Lists to store the coefficients and intercepts from each window
        rolling_coefficients = []
        rolling_intercepts = []

        # Start the rolling window analysis
        for start in range(days - window_size + 1):
            end = start + window_size
            X = np.array(self.leader_prices[start:end]).reshape(-1, 1)
            y = np.array(self.follower_prices[start:end])

            model = LinearRegression()
            model.fit(X, y)

            # Store the coefficients and intercepts from each window
            rolling_coefficients.append(model.coef_[0])
            rolling_intercepts.append(model.intercept_)

        # Set up the decay factor for the weights, usually a small value like 0.9 or 0.95
        decay_factor = 0.9

        # Create an array of indices from oldest to most recent
        indices = np.arange(len(rolling_coefficients))

        # Compute the weights using an exponential decay formula
        weights = decay_factor ** indices

        # Reverse the weights array to have the highest weight for the most recent observation
        weights = weights[::-1]

        # Compute the weighted averages
        weighted_average_coefficients = np.average(rolling_coefficients, weights=weights)
        weighted_average_intercepts = np.average(rolling_intercepts, weights=weights)

        print(f"Weighted average coefficient: {weighted_average_coefficients}")
        print(f"Weighted average intercept: {weighted_average_intercepts}")

        # return weighted_average_coefficients, weighted_average_intercepts
        return rolling_coefficients[-1], rolling_intercepts[-1]

    
    def leader_strategy(self):
        strategy_space = (1.00, np.inf)
        upper_bound = 3.00
        potential_leader_prices = np.arange(1.00, upper_bound + 0.01, 0.01)

        # Find the leader's price that maximizes the daily profit
        max_profit = 0
        best_leader_price = 0

        for leader_price in potential_leader_prices:
            profit = self.calculate_daily_profit(leader_price)
            if profit > max_profit:
                max_profit = profit
                best_leader_price = leader_price
        
        return best_leader_price

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
        for day in range(1, 131):  
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
        
        self.log(f"End of simulation. Total profit: £{total_profit:.2f}")
        return total_profit



if __name__ == '__main__':
    # Make sure you set this to your group number!
    SimpleLeader('17')
