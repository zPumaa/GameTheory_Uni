"""
A simple leader for the Stackelberg game.
"""
from base_leader import Leader
import numpy as np
from sklearn.linear_model import LinearRegression

___author___ = "Rokas"

class SimpleLeader(Leader):
    def __init__(self, name):
        # Ensure proper initialisation of all member variables before calling super().__init__
        self.leader_prices = []
        self.follower_prices = []
        self.unit_cost = 1
        self.profits = []
        self.coefficient = 0
        self.intercept = 0
        super().__init__(name)

    def calculate_daily_profit(self, leader_price, follower_price):
        """
        Calculate profit for a given day using the leader and follower prices.
        """
        demand = 2 - leader_price + 0.3 * follower_price
        profit = (leader_price - self.unit_cost) * demand
        return profit


    def new_price(self, date: int) -> float:
        """
        Determine the new price for each day, dynamically updating strategy after day 101.
        """
        self.log(f"Calculating price for day {date}")
        if date == 101:
            self.coefficient, self.intercept = self.analyse_history()
        elif date > 101:
            self.coefficient, self.intercept = self.analyse_history(days=date-1)

        best_price = self.leader_strategy()
        return float(best_price)

    def analyse_history(self, days=100):
        """
        Analyse historical data to adapt the pricing model. Include profits as a feature to understand their impact on future pricing strategies.
        """
        self.leader_prices = []
        self.follower_prices = []
        self.profits = []

        # Fetch and process data
        for day in range(1, days + 1):  
            leader_price, follower_price = self.get_price_from_date(day)
            self.leader_prices.append(leader_price)
            self.follower_prices.append(follower_price)
            
            # Calculate profit using both leader_price and follower_price
            profit = self.calculate_daily_profit(leader_price, follower_price)
            self.profits.append(profit)

        # Prepare data for regression including profits as a feature
        features = np.column_stack((self.leader_prices, self.profits))
        y = np.array(self.follower_prices)

        # Linear regression
        model = LinearRegression()
        model.fit(features, y)
        self.coefficient = model.coef_
        self.intercept = model.intercept_

        return self.coefficient, self.intercept



    def leader_strategy(self):
        """
        Determine the optimal leader price by maximising the calculated daily profit,
        incorporating insights from past profits.
        """
        strategy_space = (1.00, np.inf)
        upper_bound = 3.00
        potential_leader_prices = np.arange(1.00, upper_bound + 0.01, 0.01)

        max_profit = 0
        best_leader_price = 1.00

        for leader_price in potential_leader_prices:
            # Assume last known profit or average as a placeholder for current profit influence
            last_profit = self.profits[-1] if self.profits else 0
            features = np.array([leader_price, last_profit]).reshape(1, -1)  

            # Calculate the follower price based on the current regression model including profit
            follower_price = np.dot(features, self.coefficient) + self.intercept

            # Calculate profit using both leader_price and calculated follower_price
            profit = self.calculate_daily_profit(leader_price, follower_price[0])
            
            if profit > max_profit:
                max_profit = profit
                best_leader_price = leader_price
        
        return best_leader_price


    
    def start_simulation(self):
        """
        Handle any setup necessary at the start of the simulation.
        """
        self.log("Start of simulation")
        self.coefficient, self.intercept = self.analyse_history()

    def end_simulation(self):
        """
        A function run at the end of the simulation.
        It calculates the total profit based on the leader and follower prices for the last 30 days.
        """
        total_profit = 0
        all_leader_prices = []
        all_follower_prices = []

        # Fetch all prices first to ensure data is complete before processing
        for day in range(101, 131):
            leader_price, follower_price = self.get_price_from_date(day)
            all_leader_prices.append(leader_price)
            all_follower_prices.append(follower_price)

        # Calculate profit for each day
        for i in range(len(all_leader_prices)):
            leader_price = all_leader_prices[i]
            follower_price = all_follower_prices[i]
            profit = self.calculate_daily_profit(leader_price, follower_price)
            total_profit += profit
            self.log(f"Day {101 + i} profit: £{profit:.2f}")

        self.log(f"Profit for last 30 days: £{total_profit:.2f}")

        def log(self, message):
            print(f"Log: {message}")

if __name__ == '__main__':
    # Make sure you set this to your group number!
    SimpleLeader('TEST')
